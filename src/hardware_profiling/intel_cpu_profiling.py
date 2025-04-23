# hybrid-llm-inference/src/hardware_profiling/intel_cpu_profiling.py
from pyjoules.device.rapl import RAPLMonitor
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
import time

class IntelCPUProfiling(HardwareProfiler):
    def __init__(self, config):
        super().__init__(config)
        self.idle_power = config.get("idle_power", 10.0)  # Watts, for baseline subtraction
        try:
            self.monitor = RAPLMonitor()
        except Exception as e:
            self.logger.error(f"Failed to initialize RAPL: {e}")
            raise

    def measure(self, task, input_tokens, output_tokens):
        total_tokens = input_tokens + output_tokens
        self.monitor.start()
        start_time = time.time()
        
        try:
            task()
        except Exception as e:
            self.logger.error(f"Task failed: {e}")
            raise
        
        runtime = time.time() - start_time
        raw_energy = self.monitor.stop()
        
        # Subtract idle power
        energy = raw_energy - (self.idle_power * runtime)
        energy = max(energy, 0)  # Ensure non-negative
        
        metrics = self._compute_metrics(energy, runtime, total_tokens)
        self.logger.info(f"Intel CPU metrics: {metrics}")
        return metrics

