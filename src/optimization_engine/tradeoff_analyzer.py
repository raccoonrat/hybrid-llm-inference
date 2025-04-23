# hybrid-llm-inference/src/optimization_engine/tradeoff_analyzer.py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from toolbox.logger import get_logger
from .cost_function import CostFunction
from model_zoo import get_model

class TradeoffAnalyzer:
    def __init__(self, token_distribution_path, hardware_config, model_config, output_dir="data/processed"):
        """
        Initialize TradeoffAnalyzer for analyzing energy-runtime tradeoffs.
        
        Args:
            token_distribution_path (str): Path to token_distribution.pkl.
            hardware_config (dict): Hardware configuration.
            model_config (dict): Model configuration.
            output_dir (str): Directory to save tradeoff results.
        """
        self.token_distribution_path = Path(token_distribution_path)
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self.models = {name: get_model(name, cfg.get("mode", "local"), cfg) 
                      for name, cfg in model_config["models"].items()}

    def analyze(self, model_name="llama3"):
        """
        Analyze energy-runtime tradeoffs for different λ values.
        
        Args:
            model_name (str): Model to use for inference.
        
        Returns:
            dict: Tradeoff results {lambda: {"energy": float, "runtime": float}}.
        """
        if not self.token_distribution_path.exists():
            self.logger.error(f"Token distribution not found at {self.token_distribution_path}")
            raise FileNotFoundError(f"Token distribution not found")

        with open(self.token_distribution_path, 'rb') as f:
            distribution = pickle.load(f).get('distribution', {})

        model = self.models.get(model_name)
        if not model:
            self.logger.error(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")

        lambda_values = np.arange(0.0, 1.1, 0.1)
        results = {}

        for lambda_param in lambda_values:
            cost_function = CostFunction(lambda_param, self.hardware_config)
            total_energy = 0
            total_runtime = 0
            total_tasks = 0

            for input_tokens, input_freq in distribution['input_distribution'].items():
                for output_tokens, output_freq in distribution['output_distribution'].items():
                    # Simulate task
                    task = lambda: model.infer("Sample prompt")  # Simplified task
                    # Choose system based on paper's default thresholds (T_in=32, T_out=32)
                    system = "m1_pro" if input_tokens <= 32 and output_tokens <= 32 else "a100"
                    metrics = cost_function.compute(task, input_tokens, output_tokens, system)
                    total_energy += metrics["energy"] * input_freq * output_freq
                    total_runtime += metrics["runtime"] * input_freq * output_freq
                    total_tasks += input_freq * output_freq

            results[lambda_param] = {
                "energy": total_energy / total_tasks if total_tasks > 0 else 0,
                "runtime": total_runtime / total_tasks if total_tasks > 0 else 0
            }
            self.logger.debug(f"λ={lambda_param}: Energy={results[lambda_param]['energy']}, Runtime={results[lambda_param]['runtime']}")

        # Save results
        result_path = self.output_dir / 'tradeoff_results.json'
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved tradeoff results to {result_path}")

        # Visualize tradeoff curve
        self._visualize_tradeoff(results)
        return results

    def _visualize_tradeoff(self, results):
        """Generate and save energy-runtime tradeoff curve (similar to Figure 4)."""
        lambdas = list(results.keys())
        energies = [results[l]["energy"] for l in lambdas]
        runtimes = [results[l]["runtime"] for l in lambdas]

        plt.figure(figsize=(8, 6))
        plt.plot(runtimes, energies, marker='o')
        for i, l in enumerate(lambdas):
            plt.annotate(f"λ={l:.1f}", (runtimes[i], energies[i]))
        plt.xlabel('Average Runtime (seconds)')
        plt.ylabel('Average Energy (Joules)')
        plt.title('Energy-Runtime Tradeoff')
        plt.grid(True)

        plot_path = self.output_dir / 'tradeoff_curve.png'
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved tradeoff curve to {plot_path}")

