"""
CPU-offload load test using transformers + accelerate, extended to report
the full metric set requested for the GPU-bottleneck project:

  - Time to First Token (TTFT)
  - Throughput (tokens/sec)
  - GPU Utilization (%)
  - GPU Memory Usage
  - CPU Utilization (%)
  - Main (RAM) Memory Usage
  - CUDA Kernel Execution Time
  - Estimated PCIe data transferred between CPU <-> GPU

Install first:
    pip install transformers accelerate torch psutil nvidia-ml-py

Run directly:
    python offload_load_test.py

Run under nsys (same flags you used for vLLM) if you want the real,
ground-truth PCIe/CUDA kernel numbers instead of the estimates below:
    nsys profile \
      --trace=cuda,nvtx,osrt,cudnn,cublas \
      --sample=none --cpuctxsw=none --backtrace=none \
      --cuda-graph-trace=node \
      --output=qwen_profile_offload_hf \
      --force-overwrite=true \
      python offload_load_test.py
"""

import time
import threading
import statistics
import psutil
import torch
import torch.cuda.nvtx as nvtx
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False
    print("WARNING: pynvml not available -> GPU utilization / PCIe numbers "
          "will be skipped. Install with: pip install nvidia-ml-py")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# --- match these to your guidellm --data settings ---
PROMPT_TOKENS = 1024
OUTPUT_TOKENS = 128

# --- match these to your guidellm --profile settings ---
NUM_REQUESTS = 3           # start small to confirm it finishes, then scale up
CONCURRENCY = 1            # start with 1 - accelerate's CPU/GPU offload hooks are not
                           # safe/fast for true concurrency; raise this later once you
                           # know a single request's timing

# --- offload control ---
# None            -> let accelerate keep as much as possible on GPU (baseline-ish)
# {0: "5GiB", "cpu": "8GiB"} -> force part of the model onto CPU RAM
MAX_MEMORY = {0: "5GiB", "cpu": "8GiB"}


# ---------------------------------------------------------------------------
# Background system monitor: samples GPU util%, GPU mem, CPU util%, RAM, and
# PCIe throughput every 0.2s while requests are running.
# ---------------------------------------------------------------------------
class SystemMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self.cpu_util_samples = []
        self.ram_used_samples = []
        self.pcie_tx_bytes_total = 0
        self.pcie_rx_bytes_total = 0

    def _run(self):
        psutil.cpu_percent(interval=None)  # prime the counter
        while not self._stop.is_set():
            self.cpu_util_samples.append(psutil.cpu_percent(interval=None))
            self.ram_used_samples.append(psutil.virtual_memory().used / (1024 ** 3))

            if _HAS_NVML:
                util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
                self.gpu_util_samples.append(util.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                self.gpu_mem_samples.append(mem.used / (1024 ** 3))
                try:
                    tx_kbps = pynvml.nvmlDeviceGetPcieThroughput(
                        _GPU_HANDLE, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                    rx_kbps = pynvml.nvmlDeviceGetPcieThroughput(
                        _GPU_HANDLE, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                    self.pcie_tx_bytes_total += tx_kbps * 1024 * self.interval
                    self.pcie_rx_bytes_total += rx_kbps * 1024 * self.interval
                except Exception:
                    pass

            time.sleep(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def summary(self):
        def avg(lst):
            return statistics.mean(lst) if lst else 0.0

        def peak(lst):
            return max(lst) if lst else 0.0

        return {
            "gpu_util_avg_pct": avg(self.gpu_util_samples),
            "gpu_util_peak_pct": peak(self.gpu_util_samples),
            "gpu_mem_avg_gib": avg(self.gpu_mem_samples),
            "gpu_mem_peak_gib": peak(self.gpu_mem_samples),
            "cpu_util_avg_pct": avg(self.cpu_util_samples),
            "cpu_util_peak_pct": peak(self.cpu_util_samples),
            "ram_used_avg_gib": avg(self.ram_used_samples),
            "ram_used_peak_gib": peak(self.ram_used_samples),
            "pcie_tx_mib_est": self.pcie_tx_bytes_total / (1024 ** 2),
            "pcie_rx_mib_est": self.pcie_rx_bytes_total / (1024 ** 2),
        }


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        max_memory=MAX_MEMORY,
        dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def make_synthetic_prompt(tokenizer, n_tokens):
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tokenizer(filler, return_tensors="pt").input_ids[0][:n_tokens]
    if len(ids) < n_tokens:
        pad_id = tokenizer.eos_token_id
        ids = torch.cat([ids, torch.full((n_tokens - len(ids),), pad_id)])
    return ids.unsqueeze(0)


def run_one_request(model, input_ids, req_id):
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    nvtx.range_push(f"request_{req_id}")
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=OUTPUT_TOKENS,
            do_sample=False,
        )
    elapsed = time.time() - start
    nvtx.range_pop()

    generated = output.shape[1] - input_ids.shape[1]
    return elapsed, generated


def measure_ttft(model, tokenizer, input_ids):
    """Runs one generate() call with streaming so we can capture the
    wall-clock time until the FIRST new token appears."""
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=OUTPUT_TOKENS,
        do_sample=False,
        streamer=streamer,
    )

    start = time.time()
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    ttft = None
    for _ in streamer:
        if ttft is None:
            ttft = time.time() - start
            break
    thread.join()
    return ttft


def measure_cuda_kernel_time(model, input_ids):
    """Profiles a single generate() call with torch.profiler and reports
    total time actually spent executing CUDA kernels."""
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,  # short run, profiling has overhead
                do_sample=False,
            )

    total_cuda_us = sum(
        evt.self_cuda_time_total for evt in prof.key_averages()
        if evt.self_cuda_time_total > 0
    )
    return total_cuda_us / 1000.0  # -> ms


def run_load_test(model, tokenizer, label):
    prompt_ids = make_synthetic_prompt(tokenizer, PROMPT_TOKENS)

    print("\nWarming up...")
    run_one_request(model, prompt_ids, "warmup")

    print("Measuring Time to First Token (TTFT)...")
    ttft = measure_ttft(model, tokenizer, prompt_ids)

    print("Measuring CUDA kernel execution time (short profiling run)...")
    cuda_kernel_ms = None
    if torch.cuda.is_available():
        try:
            cuda_kernel_ms = measure_cuda_kernel_time(model, prompt_ids)
        except Exception as e:
            print(f"  (skipped, profiler error: {e})")

    monitor = SystemMonitor(interval=0.2)
    monitor.start()

    latencies = []
    tokens_per_request = []

    print(f"\nRunning load test: {NUM_REQUESTS} requests, concurrency={CONCURRENCY}")
    test_start = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [
            pool.submit(run_one_request, model, prompt_ids, i)
            for i in range(NUM_REQUESTS)
        ]
        for i, f in enumerate(futures):
            elapsed, generated = f.result()
            latencies.append(elapsed)
            tokens_per_request.append(generated)
            print(f"  request {i+1}/{NUM_REQUESTS} done in {elapsed:.1f}s "
                  f"({generated} tokens)")
    test_duration = time.time() - test_start

    monitor.stop()
    sys_stats = monitor.summary()

    total_tokens = sum(tokens_per_request)
    throughput = total_tokens / test_duration

    print(f"\n--- {label} ---")
    print(f"Requests:              {NUM_REQUESTS} (concurrency {CONCURRENCY})")
    print(f"Total duration:        {test_duration:.2f} sec")
    print(f"Total tokens out:      {total_tokens}")
    print(f"Throughput:            {throughput:.2f} tokens/sec")
    print(f"Avg request latency:   {statistics.mean(latencies):.2f} sec")
    if len(latencies) >= 2:
        print(f"P95 request latency:   {statistics.quantiles(latencies, n=20)[18]:.2f} sec")
    print(f"Time to First Token:   {ttft:.3f} sec" if ttft else "Time to First Token:   n/a")
    if cuda_kernel_ms is not None:
        print(f"CUDA kernel time (32 tok sample): {cuda_kernel_ms:.2f} ms")
    print(f"GPU utilization:       avg {sys_stats['gpu_util_avg_pct']:.1f}% / peak {sys_stats['gpu_util_peak_pct']:.1f}%")
    print(f"GPU memory used:       avg {sys_stats['gpu_mem_avg_gib']:.2f} GiB / peak {sys_stats['gpu_mem_peak_gib']:.2f} GiB")
    print(f"CPU utilization:       avg {sys_stats['cpu_util_avg_pct']:.1f}% / peak {sys_stats['cpu_util_peak_pct']:.1f}%")
    print(f"Main (RAM) memory used:avg {sys_stats['ram_used_avg_gib']:.2f} GiB / peak {sys_stats['ram_used_peak_gib']:.2f} GiB")
    print(f"Estimated PCIe transfer (during load test): TX {sys_stats['pcie_tx_mib_est']:.1f} MiB / RX {sys_stats['pcie_rx_mib_est']:.1f} MiB")
    print("(PCIe/CUDA kernel numbers here are estimates from polling; for exact")
    print(" ground-truth figures, run this same script under nsys as shown in")
    print(" the file's docstring, then read the .nsys-rep report.)")


if __name__ == "__main__":
    print(f"Loading model with MAX_MEMORY={MAX_MEMORY} ...")
    model, tokenizer = load_model()
    run_load_test(model, tokenizer, "CPU offload (transformers/accelerate)")