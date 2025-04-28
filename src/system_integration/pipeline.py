"""系统集成管道模块。"""

import os
import json
import pickle
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..dataset_manager.alpaca_loader import AlpacaLoader
from src.data_processing.token_processing import TokenProcessing
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
    
    def __init__(self, model_name: str, data_path: str, output_dir: str,
                 model_path: Optional[str] = None, distribution_path: Optional[str] = None,
                 mode: str = "hybrid", config_dir: Optional[str] = None):
        """初始化系统管道。
        
        Args:
            model_name: 模型名称
            data_path: 数据路径
            output_dir: 输出目录
            model_path: 模型路径（可选）
            distribution_path: 分布数据路径（可选）
            mode: 运行模式，默认为"hybrid"
            config_dir: 配置目录路径（可选）
        """
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.distribution_path = distribution_path
        self.mode = mode
        self.config_dir = config_dir
        
        # 初始化配置
        self.model_config = None
        self.hardware_config = None
        self.scheduler_config = None
        
        # 初始化组件
        self._init_components()
        
        logger.info("系统管道初始化完成")
    
    def _load_config(self, config_name: str) -> Dict[str, Any]:
        """从配置目录加载配置文件。
        
        Args:
            config_name: 配置文件名称
            
        Returns:
            Dict[str, Any]: 配置数据
        """
        if not self.config_dir:
            return {}
            
        if isinstance(self.config_dir, dict):
            return self.config_dir.get(config_name, {})
            
        config_path = Path(self.config_dir) / f"{config_name}.yaml"
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            return {}
            
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _init_components(self) -> None:
        """初始化组件。"""
        try:
            # 加载分布数据
            if self.distribution_path and os.path.exists(self.distribution_path):
                with open(self.distribution_path, "rb") as f:
                    self.distribution = pickle.load(f)
            else:
                # 使用默认分布
                self.distribution = {
                    "input_distribution": {10: 100, 20: 50},
                    "output_distribution": {30: 80, 40: 70}
                }
                if self.distribution_path:
                    # 保存默认分布
                    os.makedirs(os.path.dirname(self.distribution_path), exist_ok=True)
                    with open(self.distribution_path, "wb") as f:
                        pickle.dump(self.distribution, f)
            
            # 加载配置文件
            self.model_config = self._load_config("model_config")
            self.hardware_config = self._load_config("hardware_config")
            self.scheduler_config = self._load_config("scheduler_config")
            
            # 如果没有配置文件，使用默认配置
            if not self.model_config or "models" not in self.model_config:
                self.model_config = {
                    "models": {
                        self.model_name: {
                            "model_name": self.model_name,
                            "model_path": str(self.model_path),
                            "mode": self.mode,
                            "max_length": 512,
                            "batch_size": 1,
                            "device": "cuda",
                            "dtype": "float32"
                        }
                    }
                }
            elif self.model_name not in self.model_config["models"]:
                self.model_config["models"][self.model_name] = {
                    "model_name": self.model_name,
                    "model_path": str(self.model_path),
                    "mode": self.mode,
                    "max_length": 512,
                    "batch_size": 1,
                    "device": "cuda",
                    "dtype": "float32"
                }
            else:
                model_config = self.model_config["models"][self.model_name]
                if "batch_size" not in model_config:
                    model_config["batch_size"] = 1
                if "device" not in model_config:
                    model_config["device"] = "cuda"
                if "dtype" not in model_config:
                    model_config["dtype"] = "float32"
                model_config["model_path"] = str(self.model_path)
            
            if not self.hardware_config:
                self.hardware_config = {
                    "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
                    "a100": {"type": "gpu", "device_id": 0}
                }
            
            if not self.scheduler_config:
                self.scheduler_config = {
                    "scheduler_type": "token_based",
                    "max_batch_size": 4,
                    "max_queue_size": 100,
                    "max_wait_time": 1.0,
                    "scheduling_strategy": "token_based"
                }
            
            # 初始化硬件分析器
            self.profiler = get_profiler("rtx4050", {
                "device_id": 0,
                "idle_power": 15.0,
                "sample_interval": 200
            })
            
            # 初始化混合推理器
            self.hybrid_inference = HybridInference({
                "models": [{
                    "name": self.model_name,
                    "model_config": {
                        "model_path": str(self.model_path),
                        "device": self.model_config["models"][self.model_name].get("device", "cuda"),
                        "mode": self.mode,
                        "dtype": "float32",
                        "max_memory": None,
                        "batch_size": self.model_config["models"][self.model_name]["batch_size"]
                    }
                }],
                "scheduler_config": {"hardware_config": self.hardware_config}
            })
            
            # 初始化其他组件
            config = {
                "dataset_path": str(self.data_path),
                "output_dir": str(self.output_dir),
                "model_config": {
                    "model_name": self.model_name,
                    "model_path": str(self.model_path),
                    "batch_size": self.model_config["models"][self.model_name]["batch_size"],
                    "mode": self.mode
                },
                "hardware_config": self.hardware_config,
                "scheduler_config": self.scheduler_config,
                "batch_size": self.model_config["models"][self.model_name]["batch_size"]
            }
            
            self.config_manager = ConfigManager(config)
            self.token_processor = TokenProcessing(model_path=str(self.model_path))
            self.scheduler = TokenBasedScheduler(self.scheduler_config)
            self.allocator = TaskAllocator({
                "hardware_config": self.hardware_config,
                "model_config": self.model_config,
                "scheduler_config": self.scheduler_config
            })
            
            self.benchmarking = SystemBenchmarking({
                "model_path": self.model_path,
                "model_name": self.model_name,
                "batch_size": self.model_config["models"][self.model_name]["batch_size"],
                "dataset_path": str(self.data_path),
                "output_dir": str(self.output_dir),
                "scheduler_config": self.scheduler_config,
                "hardware_config": self.hardware_config,
                "model_config": self.model_config
            })
            
            logger.info("系统管道组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化组件时出错: {e}")
            raise
            
    def run(self) -> Dict[str, Any]:
        """运行系统管道。
        
        Returns:
            Dict[str, Any]: 运行结果
        """
        try:
            # 加载数据
            data_loader = AlpacaLoader(self.data_path)
            data = data_loader.load()
            
            # 处理数据
            processed_data = self.token_processor.process_tokens(data)
            
            # 优化阈值
            optimizer = ThresholdOptimizer({
                "model_config": self.model_config,
                "measure_fn": lambda input_tokens, output_tokens, device_id: {
                    "energy": 10.0,
                    "runtime": 1.0,
                    "throughput": (input_tokens + output_tokens) / 1.0,
                    "energy_per_token": 10.0 / (input_tokens + output_tokens)
                },
                "min_threshold": 100,
                "max_threshold": 1000,
                "step_size": 100
            })
            thresholds = optimizer.optimize(processed_data)
            
            # 获取模型配置
            model_config = self.model_config.get("models", {}).get(self.model_name, {})
            model_device = model_config.get("device", "cuda")
            model_dtype = model_config.get("dtype", "float32")
            
            # 分析权衡
            analyzer = TradeoffAnalyzer(
                token_distribution_path=self.distribution_path,
                hardware_config=self.hardware_config,
                model_config={
                    "models": {
                        self.model_name: {
                            "model_path": str(self.model_path),
                            "device": "cuda",
                            "dtype": model_dtype,
                            "model_name": self.model_name,
                            "batch_size": model_config.get("batch_size", 1)
                        }
                    }
                }
            )
            tradeoffs = analyzer.analyze()
            
            # 调度任务
            scheduled_tasks = self.scheduler.schedule(processed_data)
            
            # 分配任务
            allocated_tasks = self.allocator.allocate(scheduled_tasks, self.model_name)
            
            # 基准测试
            metrics = self.benchmarking.benchmark(allocated_tasks)
            
            # 生成报告
            report = self.report_generator.generate(metrics)
            
            return {
                "energy": metrics.get("energy", 0.0),
                "runtime": metrics.get("runtime", 0.0),
                "metrics": metrics,
                "config": {
                    "model_config": self.model_config,
                    "hardware_config": self.hardware_config,
                    "scheduler_config": self.scheduler_config
                },
                "distribution": self.distribution
            }
            
        except Exception as e:
            logger.error(f"运行系统管道时出错: {e}")
            raise
            
    def cleanup(self):
        """清理资源。"""
        self.profiler.cleanup()
        self.hybrid_inference.cleanup() 