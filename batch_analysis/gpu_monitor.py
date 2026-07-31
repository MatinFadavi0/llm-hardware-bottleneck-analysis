import time
import csv
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetPcieThroughput,
    NVML_PCIE_UTIL_TX_BYTES,
    NVML_PCIE_UTIL_RX_BYTES,
    nvmlShutdown
)

def monitor(output_csv="gpu_metrics.csv", interval=1):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    
    headers = ["Timestamp", "GPU_Util_Percent", "VRAM_Used_MB", "PCIe_RX_MBs", "PCIe_TX_MBs"]
    
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        
        print(f"Monitoring started. Saving output to '{output_csv}'...")
        print(f"{'Time':<10} | {'GPU Util (%)':<12} | {'VRAM Used (MB)':<15} | {'PCIe RX (MB/s)':<15} | {'PCIe TX (MB/s)':<15}")
        print("-" * 75)
        
        try:
            while True:
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                rx = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES) / 1024 / 1024
                tx = nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES) / 1024 / 1024
                
                t_str = time.strftime("%H:%M:%S")
                vram_mb = round(mem.used / 1024 / 1024, 2)
                rx_mbs = round(rx, 2)
                tx_mbs = round(tx, 2)
                

                writer.writerow([t_str, util.gpu, vram_mb, rx_mbs, tx_mbs])
                csv_file.flush()
                
                print(f"{t_str:<10} | {util.gpu:<12} | {vram_mb:<15.1f} | {rx_mbs:<15.2f} | {tx_mbs:<15.2f}")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\nMonitoring stopped. Data saved successfully in '{output_csv}'.")
        finally:
            nvmlShutdown()

if __name__ == "__main__":
    monitor()