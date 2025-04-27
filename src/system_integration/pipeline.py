"""系统集成管道模块。"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..dataset_manager.alpaca_loader import AlpacaLoader
from ..data_processing.token_processing import TokenProcessing
from ..optimization_engine.threshold_optimizer import ThresholdOptimizer
from ..optimization_engine.tradeoff_analyzer import TradeoffAnalyzer
from ..scheduling.token_based_scheduler import TokenBasedScheduler
from ..scheduling.task_allocator import TaskAllocator
from ..benchmarking.system_benchmarking import SystemBenchmarking
from ..benchmarking.report_generator import ReportGenerator
from ..toolbox.config_manager import ConfigManager
from src.model_inference.hybrid_inference import HybridInference
from src.hardware_profiling import get_profiler
from src.model_zoo.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemPipeline:
    """系统集成管道类，用于协调整个系统的运行。"""
    
    def __init__(
        self,
        data_path: Path,
        distribution_path: Path,
        output_dir: Path,
        model_name: str,
        model_path: str,
        mode: str = "local",
        config_dir: Optional[Path] = None
    ):
        """初始化系统管道。
        
        Args:
            data_path: 数据文件路径
            distribution_path: 分布文件路径
            output_dir: 输出目录路径
            model_name: 模型名称
            model_path: 模型路径
            mode: 运行模式（local/remote）
            config_dir: 配置目录路径
        """
        self.data_path = data_path
        self.distribution_path = distribution_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.model_path = model_path
        self.mode = mode
        self.config_dir = config_dir
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化系统组件。"""
        # 加载分布数据
        with open(self.distribution_path, "rb") as f:
            self.distribution = pickle.load(f)
        
        # 初始化硬件分析器
        self.profiler = get_profiler("rtx4050", {
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        })
        
        # 初始化混合推理器
        self.hybrid_inference = HybridInference([{
            "name": self.model_name,
            "size": "1.1B",  # 假设是 TinyLlama
            "precision": "float16"
        }])
    
    def run(self) -> Dict[str, Any]:
        """运行系统管道。
        
        Returns:
            Dict[str, Any]: 运行结果
        """
        # 加载数据
        with open(self.data_path) as f:
            data = json.load(f)
        
        # 执行推理任务
        results = []
        for item in data:
            result = self.hybrid_inference.infer({
                "input": item["prompt"],
                "max_tokens": 100
            })
            results.append(result)
        
        # 计算总体指标
        total_energy = sum(r["metrics"]["energy"] for r in results)
        total_runtime = sum(r["metrics"]["runtime"] for r in results)
        
        # 保存结果
        with open(self.output_dir / "results.json", "w") as f:
            json.dump({
                "results": results,
                "total_energy": total_energy,
                "total_runtime": total_runtime
            }, f, indent=2)
        
        return {
            "energy": total_energy,
            "runtime": total_runtime,
            "metrics": {
                "avg_energy": total_energy / len(results),
                "avg_runtime": total_runtime / len(results)
            },
            "config": {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "mode": self.mode
            },
            "distribution": self.distribution
        }
    
    def cleanup(self):
        """清理资源。"""
        self.profiler.cleanup()
        self.hybrid_inference.cleanup()

    def _init_components(self):
        """初始化系统组件。"""
        # 加载分布数据
        with open(self.distribution_path, "rb") as f:
            self.distribution = pickle.load(f)
        
        # 初始化硬件分析器
        self.profiler = get_profiler("rtx4050", {
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        })
        
        # 初始化混合推理器
        self.hybrid_inference = HybridInference([{
            "name": self.model_name,
            "size": "1.1B",  # 假设是 TinyLlama
            "precision": "float16"
        }])
        
        # 初始化组件
        self.config_manager = ConfigManager()
        self.token_processor = TokenProcessing()
        self.scheduler = TokenBasedScheduler({
            "max_batch_size": 4,
            "max_wait_time": 1.0,
            "scheduling_strategy": "token_based"
        })
        self.allocator = TaskAllocator(
            hardware_config={
                "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
                "a100": {"type": "gpu", "device_id": 0}
            },
            model_config={
                "models": {
                    "llama3": {"model_name": "meta-llama/Llama-3-8B", "mode": "local", "max_length": 512}
                }
            }
        )
        self.benchmarking = SystemBenchmarking(
            dataset_path=self.data_path,
            hardware_config={
                "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
                "a100": {"type": "gpu", "device_id": 0}
            },
            model_config={
                "models": {
                    "llama3": {"model_name": "meta-llama/Llama-3-8B", "mode": "local", "max_length": 512}
                }
            },
            scheduler_config={
                "max_batch_size": 4,
                "max_wait_time": 1.0,
                "scheduling_strategy": "token_based"
            },
            output_dir=self.output_dir
        )
        self.report_generator = ReportGenerator()
        
    def run(self) -> Dict[str, Any]:
        """
        运行系统管道。

        Returns:
            Dict[str, Any]: 运行结果
        """
        try:
            # 加载数据
            data_loader = AlpacaLoader(self.data_path)
            data = data_loader.load()
            
            # 处理数据
            processed_data = self.token_processor.process(data)
            
            # 优化阈值
            optimizer = ThresholdOptimizer()
            thresholds = optimizer.optimize(processed_data)
            
            # 分析权衡
            analyzer = TradeoffAnalyzer()
            tradeoffs = analyzer.analyze(processed_data, thresholds)
            
            # 调度任务
            scheduled_tasks = self.scheduler.schedule(processed_data, tradeoffs)
            
            # 分配任务
            allocated_tasks = self.allocator.allocate(scheduled_tasks, self.model_name or "llama3")
            
            # 基准测试
            metrics = self.benchmarking.benchmark(allocated_tasks)
            
            # 生成报告
            report = self.report_generator.generate(metrics)
            
            return {
                "energy": metrics.get("energy", 0.0),
                "runtime": metrics.get("runtime", 0.0),
                "metrics": metrics,
                "config": self.config_manager.get_config(),
                "distribution": self.distribution_path
            }
            
        except Exception as e:
            logger.error(f"运行系统管道时出错: {e}")
            raise 