#!/usr/bin/env python3
"""Monitoring utilities for RunPod GPU training."""

import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime
from typing import Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUMonitor:
    """Real-time monitoring for GPU training sessions."""
    
    def __init__(self, log_file: str = "/workspace/results/training_monitor.log"):
        self.log_file = log_file
        self.start_time = time.time()
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "used_gb": psutil.virtual_memory().used / 1024**3,
                "available_gb": psutil.virtual_memory().available / 1024**3,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "used_gb": psutil.disk_usage('/').used / 1024**3,
                "free_gb": psutil.disk_usage('/').free / 1024**3,
                "percent": psutil.disk_usage('/').percent
            }
        }
        
        # GPU stats if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_stats = []
            for i in range(torch.cuda.device_count()):
                gpu_stat = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                    "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1024**3,
                    "max_memory_allocated_gb": torch.cuda.max_memory_allocated(i) / 1024**3
                }
                
                # Try to get utilization via nvidia-ml-py
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_stat.update({
                        "utilization_percent": gpu_util.gpu,
                        "memory_percent": (memory_info.used / memory_info.total) * 100,
                        "temperature": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    })
                    
                except ImportError:
                    # Fallback to nvidia-smi
                    pass
                
                gpu_stats.append(gpu_stat)
            
            stats["gpu"] = gpu_stats
        
        return stats
    
    def log_stats(self, stats: Dict[str, Any]):
        """Log statistics to file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
    
    def print_stats(self, stats: Dict[str, Any]):
        """Print statistics to console."""
        print(f"\n📊 System Monitor - {stats['timestamp']}")
        print(f"⏱️  Uptime: {stats['uptime']:.1f}s")
        print(f"🖥️  CPU: {stats['cpu_percent']:.1f}%")
        print(f"💾 RAM: {stats['memory']['used_gb']:.1f}GB/{stats['memory']['used_gb']+stats['memory']['available_gb']:.1f}GB ({stats['memory']['percent']:.1f}%)")
        print(f"💿 Disk: {stats['disk']['used_gb']:.1f}GB used, {stats['disk']['free_gb']:.1f}GB free")
        
        if "gpu" in stats:
            for gpu in stats["gpu"]:
                print(f"🎯 GPU {gpu['id']} ({gpu['name']}): {gpu['memory_allocated_gb']:.1f}GB allocated")
                if "utilization_percent" in gpu:
                    print(f"   Utilization: {gpu['utilization_percent']}%, Temp: {gpu.get('temperature', 'N/A')}°C")
    
    def monitor_continuous(self, interval: int = 30):
        """Run continuous monitoring."""
        print("🔄 Starting continuous monitoring...")
        print(f"📝 Logging to: {self.log_file}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                stats = self.get_system_stats()
                self.print_stats(stats)
                self.log_stats(stats)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def quick_check(self):
        """Quick system check."""
        stats = self.get_system_stats()
        self.print_stats(stats)
        self.log_stats(stats)
        return stats


class TrainingWatchdog:
    """Watchdog for training processes."""
    
    def __init__(self):
        self.processes = []
    
    def watch_process(self, process_name: str, max_idle_time: int = 300):
        """Watch a process and alert if it becomes idle."""
        print(f"👀 Watching process: {process_name}")
        
        # Implementation for process monitoring
        # This would track GPU utilization and alert if training stalls
        pass
    
    def check_training_health(self, log_dir: str = "/workspace/logs"):
        """Check health of training runs."""
        if not os.path.exists(log_dir):
            return {"status": "no_logs", "message": "No training logs found"}
        
        # Check for recent activity
        log_files = list(Path(log_dir).glob("*.log"))
        if not log_files:
            return {"status": "no_files", "message": "No log files found"}
        
        # Check most recent log
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        last_modified = time.time() - latest_log.stat().st_mtime
        
        if last_modified > 300:  # 5 minutes
            return {"status": "stale", "message": f"Last log update: {last_modified:.0f}s ago"}
        
        return {"status": "healthy", "message": "Training appears active"}


def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod GPU monitoring")
    parser.add_argument("--mode", choices=["check", "monitor", "watch"], default="check",
                       help="Monitoring mode")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    monitor = GPUMonitor()
    
    if args.mode == "check":
        print("🔍 Quick system check")
        monitor.quick_check()
    elif args.mode == "monitor":
        monitor.monitor_continuous(args.interval)
    elif args.mode == "watch":
        watchdog = TrainingWatchdog()
        health = watchdog.check_training_health()
        print(f"🏥 Training health: {health['status']} - {health['message']}")


if __name__ == "__main__":
    main()