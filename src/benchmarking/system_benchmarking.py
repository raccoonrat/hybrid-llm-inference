"""系统基准测试模块。"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import multiprocessing
import shutil
import torch
import torch.nn as nn
from .base_benchmarking import BaseBenchmarking
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.model_zoo.tinyllama import TinyLlama
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from src.scheduling.task_based_scheduler import TaskBasedScheduler
import time
import random
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
                - model_name: 模型名称
                - batch_size: 批处理大小
                - dataset_path: 数据集路径
                - model_config: 模型配置
                - output_dir: 输出目录
                - scheduler_config: 调度器配置（可选，默认为token_based）
        """
        # 添加默认的调度器配置
        if "scheduler_config" not in config:
            config["scheduler_config"] = {
                "scheduler_type": "token_based",
                "max_batch_size": 32,
                "max_queue_size": 100
            }
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
        """验证配置。

        验证系统基准测试的配置是否有效。

        Raises:
            ValueError: 当配置无效时抛出
        """
        # 验证基本配置
        required_fields = ["model_name", "batch_size", "dataset_path"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"配置缺少必需字段: {field}")

        # 验证调度器配置
        if "scheduler_config" not in self.config:
            raise ValueError("配置缺少调度器配置")
        
        scheduler_config = self.config["scheduler_config"]
        if not isinstance(scheduler_config, dict):
            raise ValueError("调度器配置必须是字典类型")
        
        if "scheduler" not in scheduler_config:
            raise ValueError("调度器配置缺少scheduler字段")
        
        scheduler = scheduler_config["scheduler"]
        required_scheduler_fields = ["scheduler_type", "max_batch_size", "max_queue_size"]
        for field in required_scheduler_fields:
            if field not in scheduler:
                raise ValueError(f"调度器配置缺少必需字段: {field}")

        # 验证输出目录
        if "output_dir" not in self.config:
            raise ValueError("配置缺少输出目录")
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])

        # 验证数据集路径
        if not os.path.exists(self.config["dataset_path"]):
            raise ValueError(f"数据集文件不存在: {self.config['dataset_path']}")
    
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
            model_path = self.config.get("model_path")
            if not model_path:
                raise ValueError("模型路径未指定")
            
            # 检查测试模式
            if os.getenv("TEST_MODE") == "1":
                logger.info("测试模式：跳过模型加载")
                return
            
            # 验证模型路径
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            if not os.access(model_path, os.R_OK):
                raise PermissionError(f"没有读取模型路径的权限: {model_path}")
            
            # 检查是否存在safetensors文件
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                from safetensors.torch import load_file
                try:
                    state_dict = load_file(safetensors_path)
                    logger.info(f"成功加载模型状态(safetensors): {safetensors_path}")
                except Exception as e:
                    logger.error(f"加载safetensors文件失败: {e}")
                    raise
            else:
                # 如果没有safetensors文件，尝试加载PyTorch模型
                try:
                    state_dict = torch.load(model_path, weights_only=False)
                    logger.info(f"成功加载模型状态(PyTorch): {model_path}")
                except Exception as e:
                    logger.error(f"加载PyTorch模型失败: {e}")
                    raise
            
            # 初始化模型
            self.model = self._create_model(state_dict)
            logger.info("模型初始化完成")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def _init_scheduler(self) -> None:
        """初始化调度器。"""
        try:
            # 获取调度器配置
            scheduler_config = self.scheduler_config["scheduler"]
            scheduler_type = scheduler_config["scheduler_type"]
            
            # 根据调度器类型初始化对应的调度器
            if scheduler_type == "token_based":
                self.scheduler = TokenBasedScheduler(scheduler_config)
            else:
                raise ValueError(f"不支持的调度器类型: {scheduler_type}")
            
            logger.info(f"成功初始化{scheduler_type}调度器")
            
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
            
            # 检查测试模式
            if os.getenv("TEST_MODE") == "1":
                # 在测试模式下，返回模拟结果
                result = {
                    "output": "测试输出",
                    "tokens": len(input_data.split()),
                    "execution_time": 0.1
                }
            else:
                # 正常模式下使用模型生成结果
                result = self.model.generate(input_data)
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
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
            # 停止测量
            if self.profiler is not None:
                self.profiler.is_measuring = False
            
            # 清理性能分析器
            if self.profiler is not None:
                self.profiler.cleanup()
                self.profiler = None
            
            # 清理模型
            if self.model is not None:
                self.model.cleanup()
                self.model = None
            
            # 清理调度器
            if self.scheduler is not None:
                self.scheduler.cleanup()
                self.scheduler = None
            
            # 清理数据集
            if self.dataset is not None:
                self.dataset = None
            
            # 清理报告生成器
            if self.report_generator is not None:
                self.report_generator.cleanup()
                self.report_generator = None
            
            # 调用父类的清理方法
            super().cleanup()
            
            # 重置初始化状态
            self.initialized = False
            
            logger.info("系统基准测试清理完成")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
            # 不抛出异常，确保资源被清理
    
    def run_benchmark(self):
        """运行基准测试的别名方法。"""
        return self.run_benchmarks(self.dataset)

    def _load_dataset(self) -> None:
        """加载数据集。"""
        try:
            # 检查测试模式
            if os.getenv("TEST_MODE") == "1":
                logger.info("测试模式：使用模拟数据集")
                self.dataset = [{"input": "test", "output": "test"}]
                return
                
            # 检查数据集路径是否是目录
            if os.path.isdir(self.dataset_path):
                # 如果是目录，尝试找到 test.json 文件
                test_file = os.path.join(self.dataset_path, "test.json")
                if os.path.exists(test_file):
                    self.dataset_path = test_file
                else:
                    raise FileNotFoundError(f"在目录中未找到 test.json 文件: {self.dataset_path}")
            
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

    def _check_pytorch_version(self) -> None:
        """检查 PyTorch 版本是否满足要求。"""
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        
        if major < 2 or (major == 2 and minor < 0):
            raise RuntimeError(f"PyTorch 版本 {version} 过低，需要 2.0.0 或更高版本")
            
        # 对于 2.6.0 及以上版本，使用 weights_only 参数
        self._use_weights_only = major > 2 or (major == 2 and minor >= 6)

    def _run_benchmark(self, input_data: Union[str, List[str]], batch_size: int = 1) -> Dict[str, Any]:
        """运行单个基准测试。

        Args:
            input_data: 输入数据
            batch_size: 批处理大小

        Returns:
            Dict[str, Any]: 基准测试结果
        """
        try:
            # 准备输入数据
            if isinstance(input_data, str):
                input_data = [input_data]
            
            # 创建任务
            task = {
                "input": input_data[0],
                "max_tokens": 100  # 默认值
            }
            
            # 执行推理
            result = self.model.infer(task)
            
            return {
                "output": result["output"],
                "metrics": result["metrics"]
            }
        except Exception as e:
            logger.error(f"基准测试执行失败: {str(e)}")
            raise

class Linear(nn.Linear):
    """线性模型类。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """初始化线性模型。

        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            bias: 是否使用偏置
        """
        super().__init__(in_features, out_features, bias)

    def generate(self, input_data: Union[str, List[str]]) -> Union[str, List[str]]:
        """生成输出。

        Args:
            input_data: 输入数据，可以是字符串或字符串列表

        Returns:
            生成的输出，与输入格式相同
        """
        if isinstance(input_data, str):
            # 对于单个字符串输入，返回其长度的两倍
            return str(len(input_data) * 2)
        elif isinstance(input_data, list):
            # 对于字符串列表，返回每个字符串长度的两倍
            return [str(len(x) * 2) for x in input_data]
        else:
            raise ValueError("输入数据必须是字符串或字符串列表")
