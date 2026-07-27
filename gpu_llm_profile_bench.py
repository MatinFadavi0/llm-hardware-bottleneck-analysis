#!/usr/bin/env python3
"""
gpu_llm_profile_bench.py
=========================

Automates the full workflow for Project 3 (Hardware Bottleneck Analysis
of LLM Inference on a single GPU):

  1. Launches the vLLM OpenAI-compatible server, optionally wrapped in
     `nsys profile` (Nsight Systems), with correct flags so it doesn't
     hang forever on "collecting data".
  2. Waits until the server's /health endpoint responds.
  3. Runs a configurable load of requests against the server and
     measures per-request Time-To-First-Token (TTFT) and throughput.
  4. In parallel, samples GPU utilization / GPU memory (nvidia-smi) and
     CPU utilization / RAM usage (psutil) at a fixed interval and saves
     them to a CSV.
  5. Cleanly stops the server (SIGINT -> SIGTERM -> SIGKILL fallback),
     which lets nsys flush the .nsys-rep file.
  6. If nsys was used, runs `nsys stats` to export CUDA API summaries
     (cudaMalloc, cudaMemcpy, cudaLaunchKernel, kernel execution time)
     to CSV files next to the report.

Usage examples
--------------

Baseline run (no CPU offload), with Nsight Systems profiling:

    python3 gpu_llm_profile_bench.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --gpu-memory-utilization 0.7 \
        --max-model-len 2048 \
        --out-dir runs/baseline \
        --use-nsys \
        --num-requests 20 --concurrency 4 --max-tokens 256

CPU offload run:

    python3 gpu_llm_profile_bench.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --gpu-memory-utilization 0.7 \
        --max-model-len 2048 \
        --cpu-offload-gb 4 \
        --out-dir runs/cpu_offload \
        --use-nsys \
        --num-requests 20 --concurrency 4 --max-tokens 256

Without nsys (just to sanity check server + benchmark logic):

    python3 gpu_llm_profile_bench.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --out-dir runs/quicktest --num-requests 5 --concurrency 1

Requirements
------------
    pip install requests psutil guidellm
    (nvidia-smi and nsys must be available on PATH if --use-nsys is set)

Benchmark tool
--------------
By default the load-generation step (project step 3) is delegated to
GuideLLM (`guidellm run ...`), which is what the assignment specifically
asks for. Pass --benchmark-tool custom to fall back to this script's own
simple ThreadPoolExecutor-based request sender instead.
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Missing dependency: pip install psutil", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DEFAULT_PROMPT = (
    "Explain, in a few paragraphs, how transformer-based language models "
    "process input tokens and generate text, focusing on the role of "
    "attention and how memory is used during inference."
)


@dataclass
class Config:
    model: str
    port: int
    gpu_memory_utilization: float
    max_model_len: int
    cpu_offload_gb: float
    out_dir: str
    use_nsys: bool
    nsys_duration: int
    num_requests: int
    concurrency: int
    max_tokens: int
    prompt: str
    sample_interval: float
    server_ready_timeout: int
    extra_vllm_args: list = field(default_factory=list)
    benchmark_tool: str = "guidellm"
    guidellm_profile: str = "sweep"
    guidellm_prompt_tokens: int = 256
    guidellm_output_tokens: int = 128
    guidellm_duration_seconds: int = 60
    guidellm_max_requests: int = 0
    guidellm_rate: str = ""
    guidellm_extra_args: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Server management
# --------------------------------------------------------------------------- #

def build_server_command(cfg: Config, nsys_report_path: str):
    """Builds the full command list to launch (optionally under nsys) the
    vLLM OpenAI API server."""

    vllm_cmd = [
        sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.model,
        "--port", str(cfg.port),
        "--max-model-len", str(cfg.max_model_len),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
    ]

    if cfg.cpu_offload_gb and cfg.cpu_offload_gb > 0:
        vllm_cmd += ["--cpu-offload-gb", str(cfg.cpu_offload_gb)]

    vllm_cmd += cfg.extra_vllm_args

    if not cfg.use_nsys:
        return vllm_cmd

    nsys_cmd = [
        "nsys", "profile",
        "-o", nsys_report_path,
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--trace-fork-before-exec=true",
        "--kill=none",
    ]
    if cfg.nsys_duration and cfg.nsys_duration > 0:
        nsys_cmd += ["--duration", str(cfg.nsys_duration)]

    return nsys_cmd + vllm_cmd


def start_server(cfg: Config, log_path: str, nsys_report_path: str):
    cmd = build_server_command(cfg, nsys_report_path)
    print(f"[server] launching: {' '.join(cmd)}")
    log_file = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # Detach stdin from the controlling terminal. Without this, creating
        # a new session via preexec_fn=os.setsid below puts the child in a
        # background process group relative to the terminal; any read of
        # stdin by the process (or by grandchildren it spawns, e.g. the
        # `file` binary invoked internally by the `cpuinfo` package that
        # vLLM imports at startup) then triggers SIGTTIN and hangs forever.
        stdin=subprocess.DEVNULL,
        # own process group so we can cleanly signal nsys + its vllm child
        preexec_fn=os.setsid,
        env={**os.environ},
    )
    return proc, log_file


def wait_for_server(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"[server] ready after {time.time()-start:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def stop_server(proc: subprocess.Popen, grace_seconds: int = 20):
    """Gracefully stop the server (and nsys, if wrapping it) so that
    nsys has a chance to flush the .nsys-rep report to disk."""
    if proc.poll() is not None:
        return

    pgid = os.getpgid(proc.pid)
    print("[server] sending SIGINT to process group (graceful stop)")
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=grace_seconds)
        print("[server] exited cleanly after SIGINT")
        return
    except subprocess.TimeoutExpired:
        pass

    print("[server] still alive, sending SIGTERM")
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=grace_seconds)
        print("[server] exited after SIGTERM")
        return
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass

    print("[server] force killing with SIGKILL")
    try:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass


# --------------------------------------------------------------------------- #
# Metrics collection (GPU / CPU / RAM) sampled in the background
# --------------------------------------------------------------------------- #

class MetricsCollector(threading.Thread):
    def __init__(self, csv_path: str, interval: float = 1.0):
        super().__init__(daemon=True)
        self.csv_path = csv_path
        self.interval = interval
        self._stop_event = threading.Event()
        self._rows = []

    def _read_gpu(self):
        """Returns (gpu_util_percent, gpu_mem_used_mb, gpu_mem_total_mb) for
        GPU 0 via nvidia-smi. Returns None values if nvidia-smi unavailable."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i", "0",
                ],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            util, mem_used, mem_total = [x.strip() for x in out.split(",")]
            return float(util), float(mem_used), float(mem_total)
        except Exception:
            return None, None, None

    def run(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "elapsed_s",
                "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb",
                "cpu_util_pct", "ram_used_mb", "ram_total_mb",
            ])
            f.flush()
            t0 = time.time()
            while not self._stop_event.is_set():
                gpu_util, gpu_mem_used, gpu_mem_total = self._read_gpu()
                cpu_util = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                row = [
                    datetime.now().isoformat(),
                    round(time.time() - t0, 2),
                    gpu_util, gpu_mem_used, gpu_mem_total,
                    cpu_util,
                    round(vm.used / (1024 * 1024), 1),
                    round(vm.total / (1024 * 1024), 1),
                ]
                writer.writerow(row)
                f.flush()
                self._rows.append(row)
                time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


# --------------------------------------------------------------------------- #
# Benchmark: send requests, measure TTFT + throughput
# --------------------------------------------------------------------------- #

@dataclass
class RequestResult:
    ttft: float          # seconds until first token/chunk arrived
    total_time: float    # seconds for the whole request
    completion_tokens: int
    ok: bool
    error: str = ""


def send_one_request(port: int, prompt: str, max_tokens: int) -> RequestResult:
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": "",  # server has a single loaded model; vLLM ignores/accepts blank or actual name
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    start = time.time()
    first_token_time = None
    completion_tokens = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if first_token_time is None:
                        first_token_time = time.time()
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text"):
                        completion_tokens += 1  # rough proxy; one chunk ~ one token piece
        total_time = time.time() - start
        ttft = (first_token_time - start) if first_token_time else total_time
        return RequestResult(ttft=ttft, total_time=total_time,
                              completion_tokens=completion_tokens, ok=True)
    except Exception as e:
        return RequestResult(ttft=0, total_time=time.time() - start,
                              completion_tokens=0, ok=False, error=str(e))


def run_benchmark(cfg: Config, csv_path: str):
    print(f"[bench] sending {cfg.num_requests} requests, "
          f"concurrency={cfg.concurrency}, max_tokens={cfg.max_tokens}")

    results = []
    wall_start = time.time()

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as executor:
        futures = [
            executor.submit(send_one_request, cfg.port, cfg.prompt, cfg.max_tokens)
            for _ in range(cfg.num_requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    wall_elapsed = time.time() - wall_start

    ok_results = [r for r in results if r.ok]
    failed = len(results) - len(ok_results)

    total_tokens = sum(r.completion_tokens for r in ok_results)
    avg_ttft = sum(r.ttft for r in ok_results) / len(ok_results) if ok_results else float("nan")
    throughput_tok_per_s = total_tokens / wall_elapsed if wall_elapsed > 0 else float("nan")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_idx", "ok", "ttft_s", "total_time_s", "completion_tokens", "error"])
        for i, r in enumerate(results):
            writer.writerow([i, r.ok, round(r.ttft, 4), round(r.total_time, 4), r.completion_tokens, r.error])

    summary = {
        "num_requests": cfg.num_requests,
        "concurrency": cfg.concurrency,
        "failed_requests": failed,
        "wall_clock_time_s": round(wall_elapsed, 3),
        "avg_ttft_s": round(avg_ttft, 4) if ok_results else None,
        "total_completion_tokens": total_tokens,
        "throughput_tokens_per_s": round(throughput_tok_per_s, 3) if ok_results else None,
    }

    print("[bench] summary:", json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# GuideLLM benchmark (project step 3): guidellm run ...
# --------------------------------------------------------------------------- #

def build_guidellm_command(cfg: Config, out_dir: str):
    """Builds the `guidellm run` command per the current GuideLLM CLI
    (registry-backed `--option kind=<type>,key=value,...` format)."""
    target = f"http://127.0.0.1:{cfg.port}"

    data_kv = f"kind=synthetic_text,prompt_tokens={cfg.guidellm_prompt_tokens}"
    if cfg.guidellm_output_tokens and cfg.guidellm_output_tokens > 0:
        data_kv += f",output_tokens={cfg.guidellm_output_tokens}"

    profile_kv = f"kind={cfg.guidellm_profile}"
    if cfg.guidellm_rate:
        profile_kv += f",rate={cfg.guidellm_rate}"
    if cfg.guidellm_profile == "concurrent" and cfg.concurrency:
        profile_kv += f",streams={cfg.concurrency}"
    if cfg.guidellm_profile == "throughput" and cfg.concurrency:
        profile_kv += f",max_concurrency={cfg.concurrency}"

    json_path = os.path.join(out_dir, "guidellm_benchmarks.json")
    csv_path = os.path.join(out_dir, "guidellm_benchmarks.csv")

    cmd = [
        "guidellm", "run",
        "--backend", f"kind=openai_http,target={target}",
        "--data", data_kv,
        "--profile", profile_kv,
        "--constraint", f"kind=max_duration,seconds={cfg.guidellm_duration_seconds}",
        "--output", f"kind=json,path={json_path}",
        "--output", f"kind=csv,path={csv_path}",
    ]
    if cfg.guidellm_max_requests and cfg.guidellm_max_requests > 0:
        cmd += ["--constraint", f"kind=max_requests,count={cfg.guidellm_max_requests}"]
    cmd += cfg.guidellm_extra_args

    return cmd, json_path, csv_path


def run_guidellm(cfg: Config, out_dir: str, log_path: str):
    cmd, json_path, csv_path = build_guidellm_command(cfg, out_dir)
    print(f"[guidellm] running: {' '.join(cmd)}")

    with open(log_path, "w") as log_file:
        try:
            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=cfg.guidellm_duration_seconds + 300,  # generous margin over the constraint
            )
            ok = proc.returncode == 0
        except FileNotFoundError:
            print("[guidellm] `guidellm` CLI not found on PATH. Install it with: pip install guidellm",
                  file=sys.stderr)
            return {"ok": False, "error": "guidellm not found on PATH"}
        except subprocess.TimeoutExpired:
            print(f"[guidellm] timed out after {cfg.guidellm_duration_seconds + 300}s", file=sys.stderr)
            return {"ok": False, "error": "guidellm run timed out"}

    summary = {
        "ok": ok,
        "log_path": log_path,
        "json_path": json_path if os.path.exists(json_path) else None,
        "csv_path": csv_path if os.path.exists(csv_path) else None,
    }
    if not ok:
        print(f"[guidellm] run failed (exit code {proc.returncode}); see {log_path}", file=sys.stderr)
    else:
        print(f"[guidellm] finished. Results: {json_path} / {csv_path}")
    return summary


# --------------------------------------------------------------------------- #
# nsys report post-processing
# --------------------------------------------------------------------------- #

def export_nsys_stats(nsys_report_path: str, out_dir: str):
    """Runs `nsys stats` on the produced .nsys-rep and dumps CUDA API /
    kernel summaries to CSV files for later analysis."""
    rep_file = nsys_report_path + ".nsys-rep"
    if not os.path.exists(rep_file):
        print(f"[nsys] report not found at {rep_file}, skipping stats export")
        return

    reports = [
        "cuda_api_sum",       # cudaMalloc / cudaMemcpy / cudaLaunchKernel timings
        "cuda_gpu_kern_sum",  # kernel execution time on GPU
        "cuda_gpu_mem_time_sum",  # memcpy time breakdown
    ]
    for report in reports:
        out_csv_prefix = os.path.join(out_dir, f"nsys_{report}")
        cmd = [
            "nsys", "stats",
            "--report", report,
            "--format", "csv",
            "--output", out_csv_prefix,
            rep_file,
        ]
        print(f"[nsys] exporting {report} -> {out_csv_prefix}*.csv")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        except subprocess.CalledProcessError as e:
            print(f"[nsys] failed to export {report}: {e.stderr}")
        except FileNotFoundError:
            print("[nsys] `nsys` not found on PATH, cannot export stats")
            return


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0,
                         help="Set >0 to enable vLLM's --cpu-offload-gb")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--use-nsys", action="store_true",
                         help="Wrap the server launch with `nsys profile`")
    parser.add_argument("--nsys-duration", type=int, default=0,
                         help="Optional hard cap (seconds) for nsys recording; 0 = unlimited "
                              "(recording stops when the server is stopped)")
    parser.add_argument("--num-requests", type=int, default=20,
                         help="Only used when --benchmark-tool custom")
    parser.add_argument("--concurrency", type=int, default=4,
                         help="Used by --benchmark-tool custom, and mapped into "
                              "GuideLLM's --profile streams/max_concurrency where applicable")
    parser.add_argument("--max-tokens", type=int, default=256,
                         help="Only used when --benchmark-tool custom")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sample-interval", type=float, default=1.0,
                         help="Seconds between GPU/CPU metric samples")
    parser.add_argument("--server-ready-timeout", type=int, default=300,
                         help="Max seconds to wait for the server /health endpoint")
    parser.add_argument("--extra-vllm-arg", action="append", default=[],
                         help="Additional raw arg to pass to vLLM server, can repeat")

    parser.add_argument("--benchmark-tool", choices=["guidellm", "custom"], default="guidellm",
                         help="guidellm (default, matches project step 3) or the built-in "
                              "simple request sender")
    parser.add_argument("--guidellm-profile", default="sweep",
                         choices=["sweep", "synchronous", "throughput", "concurrent", "constant", "poisson"],
                         help="GuideLLM --profile kind")
    parser.add_argument("--guidellm-prompt-tokens", type=int, default=256)
    parser.add_argument("--guidellm-output-tokens", type=int, default=128)
    parser.add_argument("--guidellm-duration-seconds", type=int, default=60,
                         help="GuideLLM max_duration constraint, per strategy")
    parser.add_argument("--guidellm-max-requests", type=int, default=0,
                         help="Optional GuideLLM max_requests constraint, per strategy (0 = unset)")
    parser.add_argument("--guidellm-rate", default="",
                         help="Rate value(s) for constant/poisson profiles, e.g. '10' or '16,32'")
    parser.add_argument("--guidellm-extra-arg", action="append", default=[],
                         help="Additional raw arg passed straight to `guidellm run`, can repeat")
    args = parser.parse_args()

    cfg = Config(
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        cpu_offload_gb=args.cpu_offload_gb,
        out_dir=args.out_dir,
        use_nsys=args.use_nsys,
        nsys_duration=args.nsys_duration,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        prompt=args.prompt,
        sample_interval=args.sample_interval,
        server_ready_timeout=args.server_ready_timeout,
        extra_vllm_args=args.extra_vllm_arg,
        benchmark_tool=args.benchmark_tool,
        guidellm_profile=args.guidellm_profile,
        guidellm_prompt_tokens=args.guidellm_prompt_tokens,
        guidellm_output_tokens=args.guidellm_output_tokens,
        guidellm_duration_seconds=args.guidellm_duration_seconds,
        guidellm_max_requests=args.guidellm_max_requests,
        guidellm_rate=args.guidellm_rate,
        guidellm_extra_args=args.guidellm_extra_arg,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    server_log_path = os.path.join(cfg.out_dir, "server.log")
    metrics_csv_path = os.path.join(cfg.out_dir, "gpu_cpu_metrics.csv")
    bench_csv_path = os.path.join(cfg.out_dir, "benchmark_requests.csv")
    guidellm_log_path = os.path.join(cfg.out_dir, "guidellm.log")
    summary_json_path = os.path.join(cfg.out_dir, "summary.json")
    nsys_report_path = os.path.join(cfg.out_dir, "llm_profile")  # nsys appends .nsys-rep

    proc = None
    collector = None
    try:
        proc, log_file = start_server(cfg, server_log_path, nsys_report_path)

        ready = wait_for_server(cfg.port, cfg.server_ready_timeout)
        if not ready:
            print(f"[error] server did not become ready within {cfg.server_ready_timeout}s. "
                  f"Check {server_log_path}", file=sys.stderr)
            stop_server(proc)
            sys.exit(1)

        collector = MetricsCollector(metrics_csv_path, interval=cfg.sample_interval)
        collector.start()

        # Small warmup request so first "real" measured requests aren't
        # skewed by CUDA graph capture / lazy kernel compilation.
        print("[bench] warmup request...")
        send_one_request(cfg.port, cfg.prompt, min(32, cfg.max_tokens))

        if cfg.benchmark_tool == "guidellm":
            bench_summary = run_guidellm(cfg, cfg.out_dir, guidellm_log_path)
        else:
            bench_summary = run_benchmark(cfg, bench_csv_path)

    finally:
        if collector is not None:
            collector.stop()
            collector.join(timeout=5)
        if proc is not None:
            stop_server(proc)

    if cfg.use_nsys:
        # give nsys a moment to finish writing the report after process exit
        time.sleep(3)
        export_nsys_stats(nsys_report_path, cfg.out_dir)

    full_summary = {
        "config": {
            "model": cfg.model,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "max_model_len": cfg.max_model_len,
            "cpu_offload_gb": cfg.cpu_offload_gb,
            "use_nsys": cfg.use_nsys,
            "benchmark_tool": cfg.benchmark_tool,
        },
        "benchmark": bench_summary,
        "files": {
            "server_log": server_log_path,
            "gpu_cpu_metrics_csv": metrics_csv_path,
            "benchmark_requests_csv": bench_csv_path if cfg.benchmark_tool == "custom" else None,
            "guidellm_log": guidellm_log_path if cfg.benchmark_tool == "guidellm" else None,
            "guidellm_json": bench_summary.get("json_path") if cfg.benchmark_tool == "guidellm" else None,
            "guidellm_csv": bench_summary.get("csv_path") if cfg.benchmark_tool == "guidellm" else None,
            "nsys_report": (nsys_report_path + ".nsys-rep") if cfg.use_nsys else None,
        },
    }
    with open(summary_json_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\n[done] all outputs written to: {cfg.out_dir}")
    print(f"       - server log:            {server_log_path}")
    print(f"       - GPU/CPU metrics CSV:    {metrics_csv_path}")
    if cfg.benchmark_tool == "guidellm":
        print(f"       - guidellm log:           {guidellm_log_path}")
        print(f"       - guidellm results JSON:  {bench_summary.get('json_path')}")
        print(f"       - guidellm results CSV:   {bench_summary.get('csv_path')}")
    else:
        print(f"       - benchmark requests CSV: {bench_csv_path}")
    print(f"       - summary JSON:           {summary_json_path}")
    if cfg.use_nsys:
        print(f"       - nsys report:            {nsys_report_path}.nsys-rep")
        print(f"       - nsys CSV summaries:     {cfg.out_dir}/nsys_*.csv")


if __name__ == "__main__":
    main()