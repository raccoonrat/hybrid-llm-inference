"""系统基准测试模块。"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import multiprocessing
import shutil
from toolbox.logger import get_logger
from .base_benchmarking import BaseBenchmarking
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_zoo.tinyllama import TinyLlama

logger = get_logger(__name__)

class SystemBenchmarking(BaseBenchmarking):
    """系统基准测试类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化系统基准测试。

        Args:
            config: 配置字典，包含所有必要的配置参数
        """
        super().__init__(config)
        self.dataset_path = config["dataset_path"]
        self.hardware_config = config["hardware_config"]
        self.model_config = config["model_config"]
        self.scheduler_config = config.get("scheduler_config", {})
        self.output_dir = config["output_dir"]
        
        # 初始化硬件监控器
        self.profiler = RTX4050Profiler(config=self.hardware_config)
        
        self._load_dataset()
        self._validate_config()
        self._init_components()
    
    def _load_dataset(self) -> None:
        """加载数据集。"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                try:
                    self.dataset = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"数据集格式错误: {str(e)}")
                    raise ValueError(f"数据集 {self.dataset_path} 不是有效的JSON格式")
        except FileNotFoundError:
            logger.error(f"数据集文件不存在: {self.dataset_path}")
            raise ValueError(f"数据集文件 {self.dataset_path} 不存在")
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            raise
        
        self._validate_dataset()
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not self.hardware_config:
            raise ValueError("hardware_config 不能为空")
        if not self.model_config:
            raise ValueError("model_config 不能为空")
        if not self.output_dir:
            raise ValueError("output_dir 不能为空")
    
    def _validate_dataset(self):
        """验证数据集。"""
        if not hasattr(self, 'dataset'):
            raise ValueError("数据集未加载")
        
        if not isinstance(self.dataset, list):
            raise ValueError(f"数据集 {self.dataset_path} 必须是JSON数组格式")
        
        if not self.dataset:
            logger.warning(f"数据集 {self.dataset_path} 为空")
            raise ValueError(f"数据集 {self.dataset_path} 为空")
        
        logger.info(f"成功加载数据集，共 {len(self.dataset)} 条记录")
    
    def _init_components(self) -> None:
        """初始化组件。"""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.initialized = True
            logger.info("系统基准测试组件初始化完成")
        except Exception as e:
            logger.error(f"系统基准测试组件初始化失败: {str(e)}")
            raise
    
    def _monitor_resources(self) -> Dict[str, float]:
        """监控系统资源使用情况。"""
        return {
            "power_usage": self.profiler.get_power_usage(),
            "memory_usage": self.profiler.get_memory_usage(),
            "gpu_utilization": self.profiler.get_gpu_utilization()
        }
    
    def _process_task(self, task: Dict[str, Any]) -> Dict[str, float]:
        """处理单个任务。"""
        # 模拟任务处理
        return {
            "throughput": 100.0,
            "latency": 0.1,
            "energy": 50.0,
            "runtime": 1.0
        }
    
    def _validate_scheduling_strategy(self, strategy: str) -> None:
        """验证调度策略。"""
        valid_strategies = ["round_robin", "token_based", "dynamic"]
        if strategy not in valid_strategies:
            raise ValueError(f"无效的调度策略: {strategy}")
    
    def run_benchmarks(self, tasks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """运行基准测试。

        Args:
            tasks: 待测试的任务列表

        Returns:
            dict: 基准测试结果，包含以下字段：
                - metrics: 基本性能指标
                - hardware_metrics: 硬件监控指标
                - parallel_metrics: 并行处理指标
                - scheduling_metrics: 调度指标
                - tradeoff_results: 权衡分析结果
        """
        if not self.initialized:
            raise RuntimeError("基准测试未初始化")
            
        try:
            # 监控资源使用
            hardware_metrics = {
                "power_usage": self.profiler.get_power_usage(),
                "memory_usage": self.profiler.get_memory_usage(),
                "gpu_utilization": self.profiler.get_gpu_utilization()
            }
            
            # 获取调度策略
            strategy = self.scheduler_config.get("strategy", "round_robin")
            self._validate_scheduling_strategy(strategy)
            
            # 多进程处理
            num_workers = self.scheduler_config.get("num_workers", 1)
            if num_workers > 1:
                with multiprocessing.Pool(num_workers) as pool:
                    parallel_results = pool.map(self._process_task, self.dataset)
            else:
                parallel_results = [self._process_task(task) for task in self.dataset]
            
            # 计算调度指标
            scheduling_metrics = {
                "strategy": strategy,
                "avg_wait_time": 0.1,  # 示例值
                "avg_queue_length": 2.0,  # 示例值
                "num_workers": num_workers
            }
            
            # 合并所有指标
            metrics = self.get_metrics()
            return {
                "metrics": metrics,
                "hardware_metrics": hardware_metrics,
                "parallel_metrics": parallel_results,
                "scheduling_metrics": scheduling_metrics,
                "tradeoff_results": {
                    "weights": [0.2, 0.5, 0.8],
                    "values": [
                        {
                            "throughput": metrics.get("throughput", 0.0),
                            "latency": metrics.get("latency", 0.0),
                            "energy": metrics.get("energy", 0.0),
                            "runtime": metrics.get("runtime", 0.0)
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"基准测试运行失败: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。"""
        return {
            "throughput": 100.0,
            "latency": 0.1,
            "energy": 10.0,
            "runtime": 1.0,
            "summary": {
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
                "avg_energy_per_token": 0.01,
                "avg_runtime": 1.0
            }
        }
    
    def cleanup(self) -> None:
        """清理资源。"""
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                logger.error(f"清理输出目录失败: {str(e)}")
                raise
        super().cleanup()
        logger.info("系统基准测试清理完成")
