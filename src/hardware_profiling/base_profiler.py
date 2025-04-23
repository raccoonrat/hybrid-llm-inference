# hybrid-llm-inference/src/hardware_profiling/base_profiler.py
from abc import ABC, abstractmethod
from toolbox.logger import get_logger
import time

class HardwareProfiler(ABC):
    def __init__(self, config):
        """
        Initialize HardwareProfiler base class.
        
        Args:
            config (dict): Hardware configuration.
        """
        self.config = config
        self.logger = get_logger(__name__)

    @abstractmethod
    def measure(self, task, input_tokens, output_tokens):
        """
        Measure energy and performance for a task.
        
        Args:
            task (callable): Task to measure (e.g., LLM inference).
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
        
        Returns:
            dict: Metrics including energy (Joules), runtime (sec), throughput (tokens/sec),
                  energy_per_token (Joules/token).
        """
        pass

    def _compute_metrics(self, energy, runtime, total_tokens):
        """Compute standard metrics."""
        throughput = total_tokens / runtime if runtime > 0 else 0
        energy_per_token = energy / total_tokens if total_tokens > 0 else 0
        return {
            "energy": energy,
            "runtime": runtime,
            "throughput": throughput,
            "energy_per_token": energy_per_token
        }

