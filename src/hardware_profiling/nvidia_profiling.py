# hybrid-llm-inference/src/hardware_profiling/nvidia_profiling.py
from pyjoules.device.nvidia import NvidiaGPUMonitor
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
import time

class NvidiaProfiling(HardwareProfiler):
    def __init__(self, config):
        super().__init__(config)
        self.device_id = config.get("device_id", 0)
        try:
            self.monitor = NvidiaGPUMonitor(self.device_id)
        except Exception as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
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
        energy = self.monitor.stop()
        
        metrics = self._compute_metrics(energy, runtime, total_tokens)
        self.logger.info(f"NVIDIA GPU metrics: {metrics}")
        return metrics

