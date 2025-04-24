# hybrid-llm-inference/src/benchmarking/system_benchmarking.py
import pandas as pd
import numpy as np
from pathlib import Path
from toolbox.logger import get_logger
from scheduling.token_based_scheduler import TokenBasedScheduler
from scheduling.task_allocator import TaskAllocator
from dataset_manager.alpaca_loader import AlpacaLoader
from model_zoo import get_model
import logging
from typing import Dict, Any, List
from model_zoo.base_model import BaseModel
from hardware_profiling import get_profiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemBenchmarking:
    """系统基准测试类。"""
    
    def __init__(
        self,
        dataset_path: Path,
        hardware_config: Dict[str, Any],
        model_config: Dict[str, Any],
        scheduler_config: Dict[str, Any] = None,
        output_dir: Path = Path("data/benchmarks")
    ):
        """
        初始化系统基准测试。

        Args:
            dataset_path: 数据集路径
            hardware_config: 硬件配置
            model_config: 模型配置
            scheduler_config: 调度器配置（可选）
            output_dir: 输出目录（可选）
        """
        self.logger = get_logger(__name__)
        self.dataset_path = dataset_path
        if not self.dataset_path.exists():
            self.logger.error(f"Dataset not found: {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.scheduler_config = scheduler_config or {
            "max_batch_size": 4,
            "max_wait_time": 1.0,
            "scheduling_strategy": "token_based"
        }
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = AlpacaLoader(dataset_path)
        self.allocator = TaskAllocator(hardware_config, model_config)
        self.logger.info("SystemBenchmarking initialized")

        self.scheduler = TokenBasedScheduler(self.scheduler_config)
        self.models = {
            name: get_model(name, cfg.get("mode", "local"), cfg)
            for name, cfg in self.model_config["models"].items()
        }

        self.profiler = get_profiler(config=hardware_config, skip_nvml=True)

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
        
        model = self.models[model_name]
        
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

    def benchmark(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对任务进行基准测试。

        Args:
            tasks: 要测试的任务列表

        Returns:
            Dict[str, Any]: 基准测试结果
        """
        if not tasks:
            logger.warning("没有任务需要测试")
            return {
                "energy": 0.0,
                "runtime": 0.0,
                "throughput": 0.0,
                "energy_per_token": 0.0
            }

        total_energy = 0.0
        total_runtime = 0.0
        total_tokens = 0

        for task in tasks:
            try:
                # 获取任务指标
                metrics = self._measure_task(task)
                total_energy += metrics.get("energy", 0.0)
                total_runtime += metrics.get("runtime", 0.0)
                total_tokens += metrics.get("tokens", 0)
            except Exception as e:
                logger.error(f"任务测试失败: {e}")
                continue

        if total_tokens == 0:
            return {
                "energy": total_energy,
                "runtime": total_runtime,
                "throughput": 0.0,
                "energy_per_token": 0.0
            }

        return {
            "energy": total_energy,
            "runtime": total_runtime,
            "throughput": total_tokens / total_runtime if total_runtime > 0 else 0.0,
            "energy_per_token": total_energy / total_tokens
        }

    def _measure_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        测量单个任务的性能指标。

        Args:
            task: 要测量的任务

        Returns:
            Dict[str, Any]: 任务性能指标
        """
        hardware = task.get("hardware")
        model_name = task.get("model")
        
        if not hardware or hardware not in self.hardware_config:
            logger.error(f"无效的硬件配置: {hardware}")
            return {
                "energy": 0.0,
                "runtime": 0.0,
                "tokens": 0
            }
            
        if not model_name or model_name not in self.models:
            logger.error(f"无效的模型名称: {model_name}")
            return {
                "energy": 0.0,
                "runtime": 0.0,
                "tokens": 0
            }

        model = self.models[model_name]
        try:
            # 执行推理
            start_time = pd.Timestamp.now()
            result = model.infer(task.get("query", ""))
            end_time = pd.Timestamp.now()
            
            # 计算指标
            runtime = (end_time - start_time).total_seconds()
            input_tokens = model.get_token_count(task.get("query", ""))
            output_tokens = model.get_token_count(result)
            total_tokens = input_tokens + output_tokens
            
            # 获取硬件指标
            profiler = get_profiler(hardware, self.hardware_config[hardware])
            power = profiler.measure_power()
            energy = power * runtime
            
            return {
                "energy": energy,
                "runtime": runtime,
                "tokens": total_tokens
            }
        except Exception as e:
            logger.error(f"任务测量失败: {e}")
            return {
                "energy": 0.0,
                "runtime": 0.0,
                "tokens": 0
            }
    
    def cleanup(self) -> None:
        """
        清理资源。
        """
        for model in self.models.values():
            model.cleanup()
        if hasattr(self, "allocator"):
            self.allocator.cleanup()
        if hasattr(self, "profiler"):
            self.profiler.cleanup()

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
