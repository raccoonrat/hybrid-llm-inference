"""系统集成管道模块。"""

import os
import json
import pickle
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

from ..dataset_manager.alpaca_loader import AlpacaLoader
from ..data_processing.token_processing import TokenProcessing
from ..optimization_engine.threshold_optimizer import ThresholdOptimizer
from ..optimization_engine.tradeoff_analyzer import TradeoffAnalyzer
from ..scheduling.token_based_scheduler import TokenBasedScheduler
from ..scheduling.task_allocator import TaskAllocator
from ..benchmarking.system_benchmarking import SystemBenchmarking
from ..benchmarking.report_generator import ReportGenerator
from ..toolbox.config_manager import ConfigManager
from ..model_inference.hybrid_inference import HybridInference
from ..hardware_profiling import get_profiler
from ..model_zoo.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemPipeline:
    """系统集成管道类，用于协调整个系统的运行。"""
    
    def __init__(self, model_name: str, data_path: str, output_dir: str,
                 model_path: str, distribution_path: Optional[str] = None,
                 mode: str = "hybrid", config_dir: Optional[Union[str, Dict[str, Any]]] = None):
        """初始化系统管道。
        
        Args:
            model_name: 模型名称（必需）
            data_path: 数据路径（必需）
            output_dir: 输出目录（必需）
            model_path: 模型路径（必需）
            distribution_path: 分布数据路径（可选）
            mode: 运行模式，默认为"hybrid"
            config_dir: 配置目录路径或配置字典（可选）
            
        Raises:
            ValueError: 当必需参数为空或无效时
            FileNotFoundError: 当必需的文件或目录不存在时
            RuntimeError: 当初始化过程中发生错误时
        """
        # 验证必需参数
        if not all([model_name, data_path, output_dir, model_path]):
            raise ValueError("必需参数不能为空")
            
        # 验证文件路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
            
        # 初始化基本属性
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.distribution_path = distribution_path
        self.mode = mode
        self.config_dir = config_dir
        
        # 初始化状态标志
        self.initialized = False
        self.components = {}
        
        try:
            # 加载配置
            self._load_configurations()
            
            # 初始化组件
            self._init_components()
            
            # 初始化报告生成器
            self._init_report_generator()
            
            self.initialized = True
            logger.info("系统管道初始化完成")
            
        except Exception as e:
            logger.error(f"初始化系统管道时出错: {str(e)}")
            self.cleanup()
            raise RuntimeError(f"初始化失败: {str(e)}")
    
    def _load_configurations(self) -> None:
        """加载和验证配置。
        
        Raises:
            ValueError: 当配置无效时
        """
        # 加载基础配置
        base_config = self._load_config("base_config") if isinstance(self.config_dir, str) else {}
        
        # 合并配置
        if isinstance(self.config_dir, dict):
            self.config = self.config_dir
        else:
            self.config = {
                "model": {
                    "model_name": self.model_name,
                    "model_path": str(self.model_path),
                    "mode": self.mode,
                    "max_length": 512,
                    "batch_size": 1,
                    "device": "cuda",
                    "dtype": "float32"
                },
                "scheduler": {
                    "scheduler_type": "token_based",
                    "max_batch_size": 4,
                    "max_queue_size": 100,
                    "max_wait_time": 1.0
                },
                "hardware": {
                    "rtx4050": {
                        "device_type": "gpu",
                        "device_id": 0,
                        "idle_power": 15.0,
                        "sample_interval": 200
                    }
                }
            }
            self.config.update(base_config)
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置的完整性和有效性。
        
        Raises:
            ValueError: 当配置无效时
        """
        required_sections = ["model", "scheduler", "hardware"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置缺少必需部分: {section}")
        
        # 验证模型配置
        model_config = self.config["model"]
        required_model_fields = ["batch_size"]  # 只保留必需的字段
        for field in required_model_fields:
            if field not in model_config:
                raise ValueError(f"模型配置缺少必需字段: {field}")
        
        # 验证调度器配置
        scheduler_config = self.config["scheduler"]
        required_scheduler_fields = ["scheduler_type", "max_batch_size", "max_queue_size"]
        for field in required_scheduler_fields:
            if field not in scheduler_config:
                raise ValueError(f"调度器配置缺少必需字段: {field}")
        
        # 验证硬件配置
        hardware_config = self.config["hardware"]
        if not isinstance(hardware_config, dict) or not hardware_config:
            raise ValueError("硬件配置无效")
    
    def _init_components(self) -> None:
        """初始化系统组件。
        
        按照依赖关系顺序初始化各个组件。
        
        Raises:
            RuntimeError: 当组件初始化失败时
        """
        try:
            # 1. 初始化基础组件
            self.hybrid_inference = HybridInference(
                config={
                    "model_name": self.model_name,
                    "model_path": self.model_path,
                    "device": self.config["model"].get("device", "cuda"),
                    "mode": self.config["model"].get("mode", "local"),
                    "batch_size": self.config["model"].get("batch_size", 1),
                    "dtype": self.config["model"].get("dtype", "float32"),
                    "scheduler_config": {
                        "hardware_config": self.config["hardware"]
                    }
                },
                test_mode=os.environ.get("TEST_MODE", "0") == "1"
            )
            self.components["hybrid_inference"] = self.hybrid_inference
            
            # 2. 初始化令牌处理器
            self.token_processor = TokenProcessing(
                model_name=self.model_name,
                model_config=self.config["model"]
            )
            self.components["token_processor"] = self.token_processor
            
            # 3. 初始化调度器
            self.scheduler = TokenBasedScheduler(self.config["scheduler"])
            self.components["scheduler"] = self.scheduler
            
            # 4. 初始化性能分析器
            if "rtx4050" in self.config["hardware"]:
                self.profiler = get_profiler("rtx4050", self.config["hardware"]["rtx4050"])
            else:
                self.profiler = None
            self.components["profiler"] = self.profiler
            
            # 5. 初始化基准测试
            # 转换为标准格式的配置
            standard_model_config = {
                "models": {
                    self.model_name: {
                        "model_name": self.model_name,
                        "model_path": self.model_path,
                        "device": self.config["model"].get("device", "cuda"),
                        "dtype": self.config["model"].get("dtype", "float32"),
                        "mode": self.config["model"].get("mode", "local"),
                        "max_length": self.config["model"].get("max_length", 512),
                        "mixed_precision": self.config["model"].get("mixed_precision", "fp16"),
                        "device_placement": self.config["model"].get("device_placement", True),
                        "batch_size": self.config["model"].get("batch_size", 1)
                    }
                }
            }

            standard_hardware_config = {
                "devices": {
                    "rtx4050": {
                        "device_type": self.config["hardware"].get("device_type", "gpu"),
                        "device_id": self.config["hardware"].get("device_id", 0),
                        "idle_power": self.config["hardware"].get("idle_power", 15.0),
                        "memory_limit": self.config["hardware"].get("memory_limit", 6144),
                        "compute_capability": self.config["hardware"].get("compute_capability", 8.9),
                        "priority": self.config["hardware"].get("priority", 1),
                        "sample_interval": self.config["hardware"].get("sample_interval", 200)
                    }
                }
            }

            standard_scheduler_config = {
                "scheduler": {
                    "scheduler_type": self.config["scheduler"].get("scheduler_type", "token_based"),
                    "max_batch_size": self.config["scheduler"].get("max_batch_size", 4),
                    "max_queue_size": self.config["scheduler"].get("max_queue_size", 100),
                    "max_wait_time": self.config["scheduler"].get("max_wait_time", 1.0),
                    "token_threshold": self.config["scheduler"].get("token_threshold", 512),
                    "dynamic_threshold": self.config["scheduler"].get("dynamic_threshold", True),
                    "batch_processing": self.config["scheduler"].get("batch_processing", True),
                    "device_priority": ["rtx4050"],
                    "monitoring": {
                        "sample_interval": self.config["hardware"].get("sample_interval", 200),
                        "metrics": ["power_usage", "memory_usage"]
                    }
                }
            }

            benchmarking_config = {
                "model_name": self.model_name,
                "model_config": standard_model_config,
                "batch_size": self.config["scheduler"].get("max_batch_size", 4),
                "dataset_path": self.data_path,
                "scheduler_config": standard_scheduler_config,
                "hardware_config": standard_hardware_config,
                "model_path": self.model_path,
                "output_dir": self.output_dir
            }
            self.benchmarking = SystemBenchmarking(benchmarking_config)
            self.components["benchmarking"] = self.benchmarking
            
            logger.info("系统管道组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化组件时出错: {str(e)}")
            self.cleanup()
            raise RuntimeError(f"组件初始化失败: {str(e)}")
    
    def _init_report_generator(self) -> None:
        """初始化报告生成器。"""
        try:
            from src.benchmarking.report_generator import ReportGenerator
            self.report_generator = ReportGenerator(self.output_dir)
            self.components["report_generator"] = self.report_generator
        except Exception as e:
            logger.error(f"初始化报告生成器时出错: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """清理资源。"""
        if hasattr(self, 'profiler') and self.profiler is not None:
            self.profiler.cleanup()
        
        # 遍历所有组件并清理
        for component_name, component in self.components.items():
            if component is not None and hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    logger.warning(f"清理组件 {component_name} 时出错: {str(e)}")
        
        # 重置状态
        self.initialized = False
    
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
    
    def _measure_performance(self, input_tokens: int, output_tokens: int, device_id: str) -> Dict[str, float]:
        """测量给定参数下的性能。
        
        Args:
            input_tokens: 输入令牌数量
            output_tokens: 输出令牌数量
            device_id: 设备ID
            
        Returns:
            Dict[str, float]: 性能指标，包括延迟和能耗
        """
        try:
            # 创建测试任务
            test_task = {
                "input": "测试输入",
                "max_tokens": output_tokens,
                "decoded_text": "测试输入",
                "input_tokens": [input_tokens],
                "input_tokens_count": input_tokens,
                "output_tokens_count": output_tokens
            }
            
            # 开始测量
            self.profiler.start()
            
            # 执行任务
            result = self.hybrid_inference.generate(
                test_task["input"],
                test_task["max_tokens"]
            )
            
            # 停止测量
            metrics = self.profiler.stop()
            
            return {
                "latency": metrics.get("execution_time", 0.0),
                "energy": metrics.get("energy", 0.0)
            }
            
        except Exception as e:
            logger.error(f"测量性能时出错: {e}")
            return {
                "latency": float("inf"),
                "energy": float("inf")
            }
            
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
            
            # 添加令牌计数到处理后的数据
            for item in processed_data:
                item["input_tokens"] = len(item.get("input", "").split())
                item["output_tokens"] = len(item.get("output", "").split())
            
            # 创建优化器
            optimizer = ThresholdOptimizer(
                search_range=(0.1, 0.9),
                num_points=10,
                device_id="cuda:0",
                measure_fn=self._measure_performance
            )
            
            # 创建示例任务
            example_task = {
                "input": "测试输入",
                "max_tokens": 100,
                "decoded_text": "测试输入",
                "input_tokens": [50],
                "input_tokens_count": 50,
                "output_tokens_count": 100
            }
            thresholds = optimizer.optimize(example_task)
            
            # 分析权衡
            analyzer = TradeoffAnalyzer(
                token_distribution_path=str(self.distribution_path),
                hardware_config=self.config["hardware"],
                model_config=self.config["model"],
                output_dir=str(self.output_dir)
            )
            analyzer.analyze(self.model_name)
            
            # 创建任务
            task = {
                "input": "测试输入",
                "max_tokens": 100,
                "decoded_text": "测试输入",
                "input_tokens": [50],
                "input_tokens_count": 50,
                "output_tokens_count": 100
            }
            
            # 将单个任务转换为任务列表
            tasks = [task]
            
            # 调度任务
            scheduled_tasks = self.scheduler.schedule(tasks)
            
            # 执行任务
            results = []
            for task in scheduled_tasks:
                result = self.hybrid_inference.infer({
                    "input": task["input"],
                    "max_tokens": task["max_tokens"]
                })
                results.append(result)
            
            # 获取模型配置
            model_config = self.config["model"]
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
            benchmark_results = self.benchmarking.run_benchmarks(allocated_tasks)
            
            # 将基准测试结果转换为报告格式
            metrics = {
                "metrics": {
                    "latency": 0.0,
                    "energy": 0.0,
                    "throughput": 0.0,
                    "runtime": 0.0
                }
            }
            
            # 从基准测试结果中提取指标
            for task_id, task_result in benchmark_results.items():
                result = task_result["result"]
                metrics["metrics"]["latency"] += result.get("execution_time", 0.0)
                metrics["metrics"]["energy"] += result.get("energy", 0.0)
                metrics["metrics"]["throughput"] += result.get("tokens", 0) / result.get("execution_time", 1.0)
                metrics["metrics"]["runtime"] += result.get("execution_time", 0.0)
            
            # 计算平均值
            num_tasks = len(benchmark_results)
            if num_tasks > 0:
                metrics["metrics"]["latency"] /= num_tasks
                metrics["metrics"]["energy"] /= num_tasks
                metrics["metrics"]["throughput"] /= num_tasks
                metrics["metrics"]["runtime"] /= num_tasks
            
            # 生成报告
            report_path = self.report_generator.generate_report(metrics)
            
            # 保存处理后的数据
            processed_df = pd.DataFrame(processed_data)
            processed_df.to_csv(Path(self.output_dir) / "processed_data.csv", index=False)
            
            return {
                "results": results,
                "thresholds": thresholds,
                "energy": metrics["metrics"]["energy"],
                "runtime": metrics["metrics"]["runtime"],
                "metrics": metrics["metrics"],
                "config": {
                    "model_config": self.model_config,
                    "hardware_config": self.hardware_config,
                    "scheduler_config": self.scheduler_config
                },
                "distribution": self.distribution,
                "report_path": report_path
            }
            
        except Exception as e:
            logger.error(f"运行系统管道时出错: {str(e)}")
            raise
            
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个任务。

        Args:
            task: 任务字典，包含以下字段：
                - input: 输入文本
                - max_length: 最大生成长度（可选）
                - additional_config: 额外的配置参数（可选）

        Returns:
            Dict[str, Any]: 处理结果，包含生成结果和性能指标
        """
        try:
            if not self.initialized:
                raise RuntimeError("系统管道未初始化")

            # 预处理任务数据
            processed_task = self.token_processor.process_tokens([task])[0]
            # 添加 token 计数信息
            processed_task["input_token_count"] = len(processed_task.get("input", "").split())
            processed_task["output_token_count"] = len(processed_task.get("output", "").split())

            # 使用阈值优化器
            optimizer = ThresholdOptimizer(
                search_range=(0.1, 0.9),
                num_points=10,
                device_id="cuda:0",
                measure_fn=self._measure_performance
            )
            thresholds = optimizer.optimize(processed_task)

            # 分析性能权衡
            if self.distribution_path:
                analyzer = TradeoffAnalyzer(
                    token_distribution_path=str(self.distribution_path),
                    hardware_config=self.config["hardware"],
                    model_config=self.config["model"],
                    output_dir=str(self.output_dir)
                )
                analyzer.analyze(self.model_name)

            # 调度任务
            scheduled_tasks = self.scheduler.schedule([processed_task])

            # 使用混合推理处理任务
            result = self.hybrid_inference.generate(
                task["input"],
                task.get("max_length", self.config["model"].get("max_length", 512))
            )

            # 收集性能指标
            metrics = {
                "latency": 0.0,
                "energy": 0.0,
                "throughput": 0.0
            }

            if self.profiler:
                try:
                    metrics = self.profiler.get_metrics()
                except Exception as e:
                    logger.warning(f"获取性能指标失败: {str(e)}")

            # 运行基准测试
            benchmark_results = self.benchmarking.run_benchmarks([processed_task])
            
            # 合并所有结果
            return {
                "result": result,
                "metrics": metrics,
                "thresholds": thresholds,
                "benchmark_results": benchmark_results,
                "scheduled_info": scheduled_tasks[0] if scheduled_tasks else None
            }

        except Exception as e:
            logger.error(f"处理任务时出错: {str(e)}")
            raise RuntimeError(f"任务处理失败: {str(e)}")

    def cleanup(self):
        """清理资源。"""
        if hasattr(self, 'profiler') and self.profiler is not None:
            self.profiler.cleanup()
        
        # 遍历所有组件并清理
        for component_name, component in self.components.items():
            if component is not None and hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except Exception as e:
                    logger.warning(f"清理组件 {component_name} 时出错: {str(e)}")
        
        # 重置状态
        self.initialized = False 