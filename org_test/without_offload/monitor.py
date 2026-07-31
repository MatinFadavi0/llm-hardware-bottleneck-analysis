import time
import csv
import psutil
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

with open("step3_hardware_metrics.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "gpu_util_pct", "gpu_mem_used_mb", "cpu_util_pct", "ram_used_mb"])
    
    print("Monitoring hardware metrics... Press Ctrl+C to stop.")
    try:
        while True:
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
            cpu_util = psutil.cpu_percent(interval=None)
            ram_used = psutil.virtual_memory().used / (1024 ** 2)
            
            writer.writerow([time.time(), gpu_util, gpu_mem, cpu_util, ram_used])
            time.sleep(0.5) # نمونه‌برداری هر نیم ثانیه
    except KeyboardInterrupt:
        print("Monitoring stopped.")