import time
import csv
import psutil
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

with open("step4_hardware_metrics.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", 
        "gpu_util_pct", 
        "gpu_mem_used_mb", 
        "cpu_util_pct", 
        "ram_used_mb",
        "pcie_tx_mbps",  # میزان ارسال داده از GPU به CPU
        "pcie_rx_mbps"   # میزان دریافت داده توسط GPU از CPU (Offload)
    ])
    
    print("Monitoring step 4 hardware metrics... Press Ctrl+C to stop.")
    try:
        while True:
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
            cpu_util = psutil.cpu_percent(interval=None)
            ram_used = psutil.virtual_memory().used / (1024 ** 2)
            
            # سنجش پهنای باند PCIe (کیلوبایت بر ثانیه به مگابایت بر ثانیه)
            pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024
            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024
            
            writer.writerow([time.time(), gpu_util, gpu_mem, cpu_util, ram_used, pcie_tx, pcie_rx])
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Monitoring stopped.")