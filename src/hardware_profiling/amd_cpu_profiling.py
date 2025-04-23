# hybrid-llm-inference/src/hardware_profiling/amd_cpu_profiling.py
import psutil
import time
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger

class AMDCPUProfiling(HardwareProfiler):
    def __init__(self, config):
        super().__init__(config)
        self.idle_power = config.get("idle_power", 15.0)  # Watts, for baseline subtraction
        self.power_per_core = config.get("power_per_core", 5.0)  # Watts/core, simplified model

    def measure(self, task, input_tokens, output_tokens):
        total_tokens = input_tokens + output_tokens
        start_time = time.time()
        cpu_usage = []
        
        # Monitor CPU usage during task
        try:
            while time.time() - start_time < 0.1:  # Initial sampling
                cpu_usage.append(psutil.cpu_percent(interval=0.1, percpu=True))
            task()
            while time.time() - start_time < 0.5:  # Post-task sampling
                cpu_usage.append(psutil.cpu_percent(interval=0.1, percpu=True))
        except Exception as e:
            self.logger.error(f"Task failed: {e}")
            raise
        
        runtime = time.time() - start_time
        
        # Simplified energy calculation (mock AMD μProf)
        avg_cpu_usage = sum(sum(usage) / len(usage) for usage in cpu_usage) / len(cpu_usage)
        active_cores = avg_cpu_usage / 100 * psutil.cpu_count()
        energy = (active_cores * self.power_per_core + self.idle_power) * runtime
        energy = max(energy - (self.idle_power * runtime), 0)  # Subtract idle power
        
        metrics = self._compute_metrics(energy, runtime, total_tokens)
        self.logger.info(f"AMD CPU metrics: {metrics}")
        return metrics
