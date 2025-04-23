# hybrid-llm-inference/src/benchmarking/model_benchmarking.py
import pandas as pd
import json
from pathlib import Path
from toolbox.logger import get_logger
from dataset_manager.alpaca_loader import AlpacaLoader
from model_zoo import get_model
from hardware_profiling import get_profiler

class ModelBenchmarking:
    def __init__(self, dataset_path, hardware_config, model_config, output_dir="data/benchmarks"):
        """
        Initialize ModelBenchmarking for evaluating model performance on different hardware.
        
        Args:
            dataset_path (str): Path to Alpaca dataset.
            hardware_config (dict): Hardware configuration.
            model_config (dict): Model configuration.
            output_dir (str): Directory to save benchmark results.
        """
        self.logger = get_logger(__name__)
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            self.logger.error(f"Dataset not found: {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = AlpacaLoader(dataset_path)
        self.profilers = {
            key: get_profiler(key, cfg) for key, cfg in hardware_config.items()
        }
        self.logger.info("ModelBenchmarking initialized")

    def run_benchmarks(self, sample_size=1000):
        """
        Run benchmarking experiments for each model on each hardware platform.
        
        Args:
            sample_size (int): Number of tasks to sample from dataset.
        
        Returns:
            dict: Benchmark results {model: {hardware: {"metrics": list, "summary": dict}}}.
        """
        if sample_size <= 0:
            self.logger.error("Sample size must be positive")
            raise ValueError("Sample size must be positive")
        
        self.logger.info(f"Starting model benchmarks with sample size {sample_size}")
        data = self.loader.load()
        if data.empty:
            self.logger.error("Dataset is empty")
            raise ValueError("Dataset is empty")
        
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
        
        results = {}
        
        for model_name, model_cfg in self.model_config["models"].items():
            results[model_name] = {}
            model = get_model(model_name, model_cfg.get("mode", "local"), model_cfg)
            
            # Prepare token data
            token_data = []
            for _, row in data.iterrows():
                try:
                    input_tokens = model.get_token_count(row["prompt"])
                    output_tokens = model.get_token_count(row["response"]) if row["response"] else 0
                    token_data.append({
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to process row for {model_name}: {e}")
                    continue
            
            if not token_data:
                self.logger.warning(f"No valid token data for {model_name}")
                continue
            
            # Benchmark on each hardware
            for hardware, profiler in self.profilers.items():
                self.logger.info(f"Benchmarking {model_name} on {hardware}")
                metrics_list = []
                
                for query in token_data:
                    prompt = query["prompt"]
                    input_tokens = query["input_tokens"]
                    output_tokens = query["output_tokens"]
                    
                    try:
                        task = lambda: model.infer(prompt)
                        metrics = profiler.measure(task, input_tokens, output_tokens)
                        metrics_list.append(metrics)
                        self.logger.debug(f"Metrics for {model_name} on {hardware}: {metrics}")
                    except Exception as e:
                        self.logger.warning(f"Failed to benchmark query on {hardware}: {e}")
                        continue
                
                results[model_name][hardware] = {
                    "metrics": metrics_list,
                    "summary": self._compute_summary(metrics_list)
                }
                self.logger.info(f"Completed benchmark for {model_name} on {hardware}")
        
        # Save results
        results_path = self.output_dir / "model_benchmarks.json"
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved model benchmark results to {results_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model benchmark results: {e}")
            raise
        
        return results
    
    def _compute_summary(self, metrics_list):
        """Compute summary statistics for benchmark results."""
        if not metrics_list:
            self.logger.warning("No metrics to summarize")
            return {}
        
        metrics_df = pd.DataFrame(metrics_list)
        return {
            "avg_energy": metrics_df["energy"].mean() if not metrics_df.empty else 0.0,
            "avg_runtime": metrics_df["runtime"].mean() if not metrics_df.empty else 0.0,
            "avg_throughput": metrics_df["throughput"].mean() if not metrics_df.empty else 0.0,
            "avg_energy_per_token": metrics_df["energy_per_token"].mean() if not metrics_df.empty else 0.0,
            "total_tasks": len(metrics_df)
        }
