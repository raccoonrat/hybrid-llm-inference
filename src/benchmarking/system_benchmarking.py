# hybrid-llm-inference/src/benchmarking/system_benchmarking.py
import pandas as pd
import numpy as np
from pathlib import Path
from toolbox.logger import get_logger
from scheduling.token_based_scheduler import TokenBasedScheduler
from scheduling.task_allocator import TaskAllocator
from dataset_manager.alpaca_loader import AlpacaLoader
from model_zoo import get_model

class SystemBenchmarking:
    def __init__(self, dataset_path, hardware_config, model_config, scheduler_config, output_dir="data/benchmarks"):
        """
        Initialize SystemBenchmarking for evaluating system performance.
        
        Args:
            dataset_path (str): Path to Alpaca dataset.
            hardware_config (dict): Hardware configuration.
            model_config (dict): Model configuration.
            scheduler_config (dict): Scheduler configuration.
            output_dir (str): Directory to save benchmark results.
        """
        self.logger = get_logger(__name__)
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            self.logger.error(f"Dataset not found: {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = AlpacaLoader(dataset_path)
        self.allocator = TaskAllocator(hardware_config, model_config)
        self.logger.info("SystemBenchmarking initialized")

    def run_benchmarks(self, thresholds, model_name="llama3", sample_size=1000):
        """
        Run benchmarking experiments for hybrid and homogeneous scheduling.
        
        Args:
            thresholds (dict): Scheduling thresholds {"T_in": int, "T_out": int}.
            model_name (str): Model to use for inference.
            sample_size (int): Number of tasks to sample from dataset.
        
        Returns:
            dict: Benchmark results {strategy: {"metrics": list, "summary": dict}}.
        """
        if not thresholds or "T_in" not in thresholds or "T_out" not in thresholds:
            self.logger.error("Invalid thresholds provided")
            raise ValueError("Thresholds must include T_in and T_out")
        if thresholds["T_in"] <= 0 or thresholds["T_out"] <= 0:
            self.logger.error("Thresholds must be positive")
            raise ValueError("Thresholds must be positive")
        
        if model_name not in self.model_config["models"]:
            self.logger.error(f"Model {model_name} not found in configuration")
            raise ValueError(f"Model {model_name} not found")
        
        self.logger.info(f"Starting benchmarks with model {model_name}, sample size {sample_size}")
        data = self.loader.load()
        if data.empty:
            self.logger.error("Dataset is empty")
            raise ValueError("Dataset is empty")
        
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
        
        model = get_model(model_name, self.model_config["models"][model_name].get("mode", "local"), 
                         self.model_config["models"][model_name])
        
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
                self.logger.warning(f"Failed to process row: {e}")
                continue
        
        if not token_data:
            self.logger.error("No valid token data generated")
            raise ValueError("No valid token data generated")
        
        results = {}
        
        # Hybrid scheduling
        try:
            scheduler = TokenBasedScheduler(thresholds, self.scheduler_config)
            allocations = scheduler.schedule(token_data)
            hybrid_results = self.allocator.allocate(allocations, model_name)
            results["hybrid"] = {
                "metrics": [r["metrics"] for r in hybrid_results],
                "summary": self._compute_summary(hybrid_results)
            }
            self.logger.info("Completed hybrid scheduling benchmark")
        except Exception as e:
            self.logger.error(f"Hybrid scheduling benchmark failed: {e}")
            raise
        
        # Homogeneous scheduling for each hardware
        for hardware in self.hardware_config.keys():
            try:
                homogeneous_allocations = [{"query": q, "hardware": hardware} for q in token_data]
                hw_results = self.allocator.allocate(homogeneous_allocations, model_name)
                results[hardware] = {
                    "metrics": [r["metrics"] for r in hw_results],
                    "summary": self._compute_summary(hw_results)
                }
                self.logger.info(f"Completed {hardware} homogeneous benchmark")
            except Exception as e:
                self.logger.error(f"{hardware} homogeneous benchmark failed: {e}")
                raise
        
        return results
    
    def _compute_summary(self, results):
        """Compute summary statistics for benchmark results."""
        if not results:
            self.logger.warning("No results to summarize")
            return {}
        
        metrics_df = pd.DataFrame([r["metrics"] for r in results])
        return {
            "avg_energy": metrics_df["energy"].mean() if not metrics_df.empty else 0.0,
            "avg_runtime": metrics_df["runtime"].mean() if not metrics_df.empty else 0.0,
            "avg_throughput": metrics_df["throughput"].mean() if not metrics_df.empty else 0.0,
            "avg_energy_per_token": metrics_df["energy_per_token"].mean() if not metrics_df.empty else 0.0,
            "total_tasks": len(metrics_df)
        }

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
