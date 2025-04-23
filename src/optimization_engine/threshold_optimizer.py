# hybrid-llm-inference/src/optimization_engine/threshold_optimizer.py
import pickle
import numpy as np
from pathlib import Path
from toolbox.logger import get_logger
from .cost_function import CostFunction
from model_zoo import get_model

class ThresholdOptimizer:
    def __init__(self, token_distribution_path, hardware_config, model_config):
        """
        Initialize ThresholdOptimizer for computing optimal T_in and T_out.
        
        Args:
            token_distribution_path (str): Path to token_distribution.pkl.
            hardware_config (dict): Hardware configuration.
            model_config (dict): Model configuration.
        """
        self.token_distribution_path = Path(token_distribution_path)
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.logger = get_logger(__name__)
        self.distribution = None
        self.models = {name: get_model(name, cfg.get("mode", "local"), cfg) 
                      for name, cfg in model_config["models"].items()}
        
    def load_distribution(self):
        """Load token distribution from file."""
        if not self.token_distribution_path.exists():
            self.logger.error(f"Token distribution not found at {self.token_distribution_path}")
            raise FileNotFoundError(f"Token distribution not found")
        
        with open(self.token_distribution_path, 'rb') as f:
            data = pickle.load(f)
        self.distribution = data.get('distribution', {})
        self.logger.info("Loaded token distribution")

    def optimize(self, lambda_param=0.5, model_name="llama3"):
        """
        Optimize scheduling thresholds T_in and T_out.
        
        Args:
            lambda_param (float): Tradeoff parameter for cost function.
            model_name (str): Model to use for inference.
        
        Returns:
            dict: Optimal thresholds {"T_in": int, "T_out": int}.
        """
        if self.distribution is None:
            self.load_distribution()

        cost_function = CostFunction(lambda_param, self.hardware_config)
        model = self.models.get(model_name)
        if not model:
            self.logger.error(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")

        # Define search space (based on paper: 8 to 2048 for input, 8 to 4096 for output)
        input_thresholds = np.logspace(np.log10(8), np.log10(2048), num=10, dtype=int)
        output_thresholds = np.logspace(np.log10(8), np.log10(4096), num=10, dtype=int)
        
        best_cost = float('inf')
        best_thresholds = {"T_in": 32, "T_out": 32}  # Default from paper
        
        # Grid search over thresholds
        for T_in in input_thresholds:
            for T_out in output_thresholds:
                total_cost = 0
                for input_tokens, input_freq in self.distribution['input_distribution'].items():
                    for output_tokens, output_freq in self.distribution['output_distribution'].items():
                        # Simulate task
                        task = lambda: model.infer("Sample prompt")  # Simplified task
                        system = "m1_pro" if input_tokens <= T_in and output_tokens <= T_out else "a100"
                        cost = cost_function.compute(task, input_tokens, output_tokens, system)
                        total_cost += cost * input_freq * output_freq
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_thresholds = {"T_in": T_in, "T_out": T_out}
                    self.logger.debug(f"New best thresholds: {best_thresholds}, Cost: {best_cost}")

        self.logger.info(f"Optimal thresholds: {best_thresholds}, Total cost: {best_cost}")
        return best_thresholds


