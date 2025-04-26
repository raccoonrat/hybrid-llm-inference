"""系统基准测试模块。"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import multiprocessing
import shutil
from toolbox.logger import get_logger
from toolbox.config_manager import ConfigManager
from .base_benchmarking import BaseBenchmarking
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_zoo.tinyllama import TinyLlama
import time
import random
import torch

logger = get_logger(__name__)

class SystemBenchmarking(BaseBenchmarking):
    """系统基准测试类，用于测试系统整体性能。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化系统基准测试。

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.initialized = False
        self.results = {}
        
        # 初始化组件
        self._init_components()
        
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
    
    def _init_components(self) -> None:
        """初始化组件。"""
        # 初始化模型
        self._init_model()
        
        # 初始化调度器
        self._init_scheduler()
        
        self.initialized = True
    
    def _init_model(self) -> None:
        """初始化模型。"""
        try:
            # 获取模型配置
            model_config = self.config_manager.get_model_config()
            model_path = self.config_manager.get_model_path()
            
            # 加载模型
            self.model = torch.load(model_path)
            self.model.to(self.hardware_config["device"])
            self.model.eval()
            
            logger.info(f"模型加载完成: {model_path}")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise
    
    def _init_scheduler(self) -> None:
        """初始化调度器。"""
        try:
            # 获取调度器配置
            scheduler_type = self.scheduler_config["scheduler_type"]
            
            # 根据类型初始化调度器
            if scheduler_type == "token_based":
                from src.scheduler.token_based_scheduler import TokenBasedScheduler
                self.scheduler = TokenBasedScheduler(self.scheduler_config)
            elif scheduler_type == "task_based":
                from src.scheduler.task_based_scheduler import TaskBasedScheduler
                self.scheduler = TaskBasedScheduler(self.scheduler_config)
            else:
                raise ValueError(f"不支持的调度器类型: {scheduler_type}")
            
            logger.info(f"调度器初始化完成: {scheduler_type}")
        except Exception as e:
            logger.error(f"调度器初始化失败: {str(e)}")
            raise
    
    def run_benchmarks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行基准测试。

        Args:
            tasks: 待测试的任务列表

        Returns:
            基准测试结果
        """
        if not self.initialized:
            raise RuntimeError("基准测试未初始化")
        
        try:
            results = {}
            
            # 运行每个任务
            for task in tasks:
                task_id = task.get("task_id", str(len(results)))
                logger.info(f"开始运行任务 {task_id}")
                
                # 记录开始时间
                start_time = time.time()
                
                # 运行任务
                task_result = self._run_task(task)
                
                # 记录结束时间
                end_time = time.time()
                
                # 计算执行时间
                execution_time = end_time - start_time
                
                # 记录结果
                results[task_id] = {
                    "task": task,
                    "result": task_result,
                    "execution_time": execution_time
                }
                
                logger.info(f"任务 {task_id} 完成，执行时间: {execution_time:.2f}秒")
            
            self.results = results
            return results
        except Exception as e:
            logger.error(f"基准测试运行失败: {str(e)}")
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
            input_data = task.get("input_data")
            if not input_data:
                raise ValueError("任务必须包含输入数据")
            
            # 使用调度器分配任务
            scheduled_task = self.scheduler.schedule_task(task)
            
            # 运行模型推理
            with torch.no_grad():
                output = self.model(input_data)
            
            return output
        except Exception as e:
            logger.error(f"任务运行失败: {str(e)}")
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
            
            # 调用父类的清理方法
            super().cleanup()
            
            logger.info("系统基准测试清理完成")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
            raise

    def run_benchmark(self):
        """运行基准测试的别名方法。"""
        return self.run_benchmarks()
