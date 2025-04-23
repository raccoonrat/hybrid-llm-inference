# hybrid-llm-inference/src/optimization_engine/cost_function.py
from toolbox.logger import get_logger
from hardware_profiling import get_profiler

class CostFunction:
    def __init__(self, lambda_param=0.5, hardware_config=None):
        """
        Initialize CostFunction for computing U(m, n, s) = λE + (1-λ)R.
        
        Args:
            lambda_param (float): Tradeoff parameter (0 <= λ <= 1).
            hardware_config (dict): Hardware configuration for profiling.
        """
        self.lambda_param = lambda_param
        self.hardware_config = hardware_config or {}
        self.logger = get_logger(__name__)
        self.profilers = {key: get_profiler(key, cfg) for key, cfg in self.hardware_config.items()}

    def compute(self, task, input_tokens, output_tokens, system):
        """
        Compute cost for a task on a specific system.
        
        Args:
            task (callable): Task to measure (e.g., LLM inference).
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
            system (str): Hardware system (e.g., 'm1_pro', 'a100').
        
        Returns:
            float: Cost value (λE + (1-λ)R).
        """
        if system not in self.profilers:
            self.logger.error(f"System {system} not supported")
            raise ValueError(f"System {system} not supported")

        profiler = self.profilers[system]
        metrics = profiler.measure(task, input_tokens, output_tokens)
        
        energy = metrics["energy"]  # Joules
        runtime = metrics["runtime"]  # Seconds
        cost = self.lambda_param * energy + (1 - self.lambda_param) * runtime
        
        self.logger.debug(f"Cost for {system}: {cost} (E={energy}, R={runtime}, λ={self.lambda_param})")
        return cost
