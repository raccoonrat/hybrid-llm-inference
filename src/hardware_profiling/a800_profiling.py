# hybrid-llm-inference/src/hardware_profiling/a800_profiling.py
from .base_profiler import BaseProfiler
from toolbox.logger import get_logger
import pyjoules
import pynvml
import time

class A800Profiler(BaseProfiler):
    def __init__(self, config):
        """
        Initialize profiler for NVIDIA A800 GPU.
        
        Args:
            config (dict): Hardware configuration (e.g., idle_power, sample_interval).
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.device_id = config.get("device_id", 0)
        self.sample_interval = config.get("sample_interval", 200) / 1000.0  # ms to seconds
        self.idle_power = config.get("idle_power", 50.0)  # Watts, higher due to HBM2e
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self.logger.info(f"Initialized A800 profiler for device {self.device_id}")
        except pynvml.NVMLError as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            raise

    def measure(self, task, input_tokens, output_tokens):
        """
        Measure energy and runtime for a task on A800.
        
        Args:
            task (callable): Task to execute (e.g., model inference).
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
        
        Returns:
            dict: Metrics {"energy": float, "runtime": float, "throughput": float, "energy_per_token": float}.
        """
        try:
            # Start energy measurement
            energy_monitor = pyjoules.EnergyMonitor()
            energy_monitor.start()
            
            # Execute task
            start_time = time.time()
            task()
            runtime = time.time() - start_time
            
            # Stop energy measurement
            energy_monitor.stop()
            energy = energy_monitor.get_energy() / 1000.0  # Convert mJ to J
            energy -= self.idle_power * runtime  # Subtract idle power
            
            # Calculate metrics
            total_tokens = input_tokens + output_tokens
            throughput = total_tokens / runtime if runtime > 0 else 0
            energy_per_token = energy / total_tokens if total_tokens > 0 else 0
            
            metrics = {
                "energy": energy,
                "runtime": runtime,
                "throughput": throughput,
                "energy_per_token": energy_per_token
            }
            self.logger.debug(f"A800 metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Measurement failed: {e}")
            raise
        finally:
            pynvml.nvmlShutdown()
