"""系统基准测试模块。"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import multiprocessing
import shutil
from .base_benchmarking import BaseBenchmarking
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_zoo.tinyllama import TinyLlama
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from src.scheduling.task_based_scheduler import TaskBasedScheduler
import time
import random
import torch
from .report_generator import ReportGenerator
from ..hardware_profiling.base_profiler import HardwareProfiler
from ..toolbox.logger import get_logger

logger = logging.getLogger(__name__)

class SystemBenchmarking(BaseBenchmarking):
    """系统基准测试类，用于测试系统整体性能。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化系统基准测试。

        Args:
            config: 配置字典，包含以下字段：
                - dataset_path: 数据集路径
                - model_config: 模型配置
                - output_dir: 输出目录
        """
        super().__init__(config)
        self.dataset_path = config["dataset_path"]
        self.model_config = config["model_config"]
        self.output_dir = config["output_dir"]
        self.dataset = None  # 初始化 dataset 属性
        self.logger = get_logger(__name__)
        self.initialized = False
        self.results = {}
        
        # 验证配置
        self._validate_config()
        
        # 加载数据集
        self._load_dataset()
        
        # 初始化组件
        self._init_components()
        
        # 设置初始化标志
        self.initialized = True
        
        logger.info("系统基准测试初始化完成")
    
    def _validate_config(self) -> None:
        """验证配置。"""
        # 验证硬件配置
        if not self.hardware_config.get("device"):
            raise ValueError("硬件配置中必须指定设备")
        
        # 验证模型配置
        if not self.model_config.get("model_path"):
            raise ValueError("模型配置中必须指定模型路径")
        
        # 验证调度器配置
        if not self.scheduler_config.get("scheduler_type"):
            raise ValueError("调度器配置中必须指定调度器类型")
        
        # 验证数据集路径
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集路径不存在: {self.dataset_path}")
    
    def _init_components(self) -> None:
        """初始化组件。

        初始化系统基准测试所需的所有组件，包括：
        - 模型
        - 调度器
        - 性能分析器
        - 报告生成器
        """
        # 初始化模型
        self._init_model()
        
        # 初始化调度器
        self._init_scheduler()
        
        # 初始化性能分析器
        self._init_profiler()
        
        # 初始化报告生成器
        self.report_generator = ReportGenerator(self.output_dir)
        
        logger.info("系统基准测试组件初始化完成")
    
    def _init_profiler(self) -> None:
        """初始化性能分析器。"""
        try:
            # 获取设备类型和ID
            device = self.hardware_config.get("device", "cpu")
            device_id = self.hardware_config.get("device_id", 0)
            
            # 根据设备类型初始化分析器
            if device.lower() == "cuda" and device_id == 0:
                self.profiler = RTX4050Profiler()
                logger.info("RTX4050性能分析器初始化成功")
            else:
                self.profiler = None
                logger.info(f"设备 {device}:{device_id} 不支持性能分析，跳过分析器初始化")
        except Exception as e:
            logger.error(f"性能分析器初始化失败: {str(e)}")
            self.profiler = None
    
    def _init_model(self) -> None:
        """初始化模型。"""
        try:
            # 获取模型配置
            model_config = self.config_manager.get_model_config()
            model_path = self.config_manager.get_model_path()
            
            # 创建模型实例
            model_type = model_config.get("model_type", "test_model")
            if model_type == "test_model":
                self.model = torch.nn.Linear(10, 10)
            elif model_type == "mock":
                from src.model_zoo.mock_model import MockModel
                self.model = MockModel(model_path=model_path)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 在非测试模式下加载模型状态
            if os.getenv('TEST_MODE') != '1':
                state_dict = torch.load(model_path)
                self.model.load_state_dict(state_dict)
            
            self.model.to(self.hardware_config["device"])
            self.model.eval()
            
            self.logger.info("模型初始化成功")
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
            raise
    
    def _init_scheduler(self) -> None:
        """初始化调度器。"""
        try:
            # 获取调度器配置
            scheduler_type = self.scheduler_config["scheduler_type"]
            
            # 根据类型初始化调度器
            if scheduler_type == "token_based":
                self.scheduler = TokenBasedScheduler(self.scheduler_config)
            elif scheduler_type == "task_based":
                self.scheduler = TaskBasedScheduler(self.scheduler_config)
            else:
                raise ValueError(f"不支持的调度器类型: {scheduler_type}")
            
            logger.info(f"调度器初始化完成: {scheduler_type}")
        except Exception as e:
            logger.error(f"调度器初始化失败: {str(e)}")
            raise
    
    def run_benchmarks(self, tasks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """运行基准测试。

        Args:
            tasks: 待测试的任务列表，如果为 None 则使用数据集中的任务

        Returns:
            基准测试结果
        """
        if not self.initialized:
            raise RuntimeError("基准测试未初始化")
        
        try:
            results = {}
            
            # 如果没有提供任务列表，则使用数据集中的任务
            if tasks is None:
                tasks = self.dataset
            
            # 确保任务列表是列表类型
            if isinstance(tasks, str):
                # 如果是字符串，假设是 JSON 文件路径
                with open(tasks, 'r', encoding='utf-8') as f:
                    tasks = json.load(f)
            elif not isinstance(tasks, list):
                raise ValueError("任务必须是列表类型或 JSON 文件路径")
            
            # 运行每个任务
            for task in tasks:
                if not isinstance(task, dict):
                    raise ValueError("每个任务必须是字典类型")
                
                task_id = task.get("task_id", str(len(results)))
                self.logger.info(f"开始运行任务 {task_id}")
                
                # 运行任务并记录结果
                task_result = self._run_task(task)
                results[task_id] = {
                    "task": task,
                    "result": task_result["result"],
                    "execution_time": task_result["execution_time"]
                }
            
            self.results = results
            return results
        except Exception as e:
            self.logger.error(f"基准测试运行失败: {str(e)}")
            raise
    
    def _run_task(self, task: Dict[str, Any]) -> Any:
        """运行单个任务。

        Args:
            task: 任务配置

        Returns:
            任务结果
        """
        try:
            # 获取任务输入
            input_data = task.get("input_data") or task.get("input")
            if not input_data:
                raise ValueError("任务必须包含输入数据 (input_data 或 input 字段)")
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行任务
            self.logger.info(f"开始处理任务: {input_data}")
            result = self.model.generate(input_data)
            
            # 记录结束时间
            end_time = time.time()
            execution_time = end_time - start_time
            
            return {
                "result": result,
                "execution_time": execution_time
            }
        except Exception as e:
            self.logger.error(f"任务运行失败: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。

        Returns:
            性能指标字典
        """
        if not self.results:
            raise RuntimeError("没有可用的基准测试结果")
        
        try:
            metrics = {}
            
            # 计算平均执行时间
            execution_times = [result["execution_time"] for result in self.results.values()]
            metrics["average_execution_time"] = sum(execution_times) / len(execution_times)
            
            # 计算最大执行时间
            metrics["max_execution_time"] = max(execution_times)
            
            # 计算最小执行时间
            metrics["min_execution_time"] = min(execution_times)
            
            # 计算吞吐量（任务/秒）
            total_time = sum(execution_times)
            metrics["throughput"] = len(self.results) / total_time if total_time > 0 else 0
            
            return metrics
        except Exception as e:
            logger.error(f"获取性能指标失败: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """清理资源。"""
        try:
            # 清理模型
            if hasattr(self, 'model'):
                del self.model
            
            # 清理调度器
            if hasattr(self, 'scheduler'):
                del self.scheduler
            
            # 清理性能分析器
            if hasattr(self, 'profiler'):
                del self.profiler
            
            # 调用父类的清理方法
            super().cleanup()
            
            logger.info("系统基准测试清理完成")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
            raise

    def run_benchmark(self):
        """运行基准测试的别名方法。"""
        return self.run_benchmarks(self.dataset)

    def _load_dataset(self) -> None:
        """加载数据集。"""
        try:
            # 加载数据集
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            
            # 验证数据集
            if self.dataset is None:
                raise ValueError("数据集不能为 None")
            
            if not isinstance(self.dataset, list):
                raise ValueError("数据集必须是列表类型")
            
            if not self.dataset:
                raise ValueError("数据集不能为空")
            
            for item in self.dataset:
                if not isinstance(item, dict):
                    raise ValueError("数据集中的每个项目必须是字典类型")
                if "input" not in item:
                    raise ValueError("数据集中的每个项目必须包含 input 字段")
                
            logger.info(f"成功加载数据集，共 {len(self.dataset)} 条数据")
        except json.JSONDecodeError:
            raise ValueError(f"数据集文件格式错误: {self.dataset_path}")
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            raise
