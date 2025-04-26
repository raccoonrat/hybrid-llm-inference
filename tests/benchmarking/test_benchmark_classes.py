"""基准测试类的测试模块。"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
from src.benchmarking.system_benchmarking import SystemBenchmarking
from src.benchmarking.model_benchmarking import ModelBenchmarking
from src.benchmarking.base_benchmarking import BaseBenchmarking
from src.benchmarking.report_generator import ReportGenerator
from src.toolbox.logger import get_logger
from src.scheduling.task_based_scheduler import TaskBasedScheduler

@pytest.fixture
def system_benchmarking():
    """创建系统基准测试实例。"""
    class TestSystemBenchmarking(SystemBenchmarking):
        def setup_method(self):
            """设置测试环境。"""
            self.dataset_path = None
            self.output_dir = None
            self.config = None
            self.initialized = False
        
        def _init_components(self):
            """初始化组件。"""
            if not os.path.exists(self.dataset_path):
                raise ValueError(f"数据集文件 {self.dataset_path} 不存在")
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                if not dataset:
                    raise ValueError("数据集不能为空")
                for item in dataset:
                    if not all(k in item for k in ["input", "output", "tokens"]):
                        raise ValueError("数据集中的每个项目必须包含input、output和tokens字段")
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            self.initialized = True
        
        def get_metrics(self):
            """获取性能指标。"""
            return {
                "throughput": 100.0,
                "latency": 0.1,
                "energy": 50.0,
                "runtime": 1.0,
                "summary": {
                    "avg_throughput": 100.0,
                    "avg_latency": 0.1,
                    "avg_energy_per_token": 0.5,
                    "avg_runtime": 1.0
                }
            }
        
        def run_benchmarks(self, tasks=None):
            """运行基准测试。"""
            metrics = self.get_metrics()
            return {
                "metrics": metrics,
                "hardware_metrics": {
                    "power_usage": 100.0,
                    "memory_usage": 8000,
                    "gpu_utilization": 80.0
                },
                "parallel_metrics": [
                    {"throughput": 100.0, "latency": 0.1},
                    {"throughput": 110.0, "latency": 0.12}
                ],
                "scheduling_metrics": {
                    "strategy": "round_robin",
                    "avg_wait_time": 0.1,
                    "avg_queue_length": 2.0
                },
                "tradeoff_results": {
                    "weights": [0.2, 0.5, 0.8],
                    "values": [
                        {
                            "throughput": metrics["throughput"],
                            "latency": metrics["latency"],
                            "energy": metrics["energy"],
                            "runtime": metrics["runtime"]
                        }
                    ]
                }
            }
        
        def cleanup(self):
            """清理测试环境。"""
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
    
    return TestSystemBenchmarking

@pytest.fixture
def model_benchmarking():
    """创建模型基准测试实例。"""
    class TestModelBenchmarking(ModelBenchmarking):
        def setup_method(self):
            """设置测试环境。"""
            self.dataset_path = None
            self.output_dir = None
            self.config = None
            self.initialized = False
        
        def _init_components(self):
            """初始化组件。"""
            if not os.path.exists(self.dataset_path):
                raise ValueError(f"数据集文件 {self.dataset_path} 不存在")
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                if not dataset:
                    raise ValueError("数据集不能为空")
                for item in dataset:
                    if not all(k in item for k in ["input", "output", "tokens"]):
                        raise ValueError("数据集中的每个项目必须包含input、output和tokens字段")
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            self.initialized = True
        
        def get_metrics(self):
            """获取性能指标。"""
            return {
                "throughput": 100.0,
                "latency": 0.1,
                "energy": 50.0,
                "runtime": 1.0,
                "summary": {
                    "avg_throughput": 100.0,
                    "avg_latency": 0.1,
                    "avg_energy_per_token": 0.5,
                    "avg_runtime": 1.0
                }
            }
        
        def run_benchmarks(self, tasks=None):
            """运行基准测试。"""
            metrics = self.get_metrics()
            return {
                "metrics": metrics,
                "tradeoff_results": {
                    "weights": [0.2, 0.5, 0.8],
                    "values": [
                        {
                            "throughput": metrics["throughput"],
                            "latency": metrics["latency"],
                            "energy": metrics["energy"],
                            "runtime": metrics["runtime"]
                        }
                    ]
                }
            }
        
        def cleanup(self):
            """清理测试环境。"""
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
    
    return TestModelBenchmarking

class TestSystemBenchmarking(SystemBenchmarking):
    """系统基准测试测试类。"""
    
    def __init__(self, config):
        """初始化测试用系统基准测试。"""
        super().__init__(config)
        self.logger = get_logger(__name__)
    
    def setup_method(self):
        """设置测试环境。"""
        self.dataset_path = None
        self.output_dir = None
        self.config = None
        self.initialized = False
    
    def _init_components(self):
        """初始化组件。"""
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集文件 {self.dataset_path} 不存在")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            if not dataset:
                raise ValueError("数据集不能为空")
            for item in dataset:
                if not all(k in item for k in ["input", "output", "tokens"]):
                    raise ValueError("数据集中的每个项目必须包含input、output和tokens字段")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.initialized = True
    
    def get_metrics(self):
        """获取性能指标。"""
        return {
            "throughput": 100.0,
            "latency": 0.1,
            "energy": 50.0,
            "runtime": 1.0,
            "summary": {
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
                "avg_energy_per_token": 0.5,
                "avg_runtime": 1.0
            }
        }
    
    def run_benchmarks(self, tasks=None):
        """运行基准测试。"""
        metrics = self.get_metrics()
        return {
            "metrics": metrics,
            "hardware_metrics": {
                "power_usage": 100.0,
                "memory_usage": 8000,
                "gpu_utilization": 80.0
            },
            "parallel_metrics": [
                {"throughput": 100.0, "latency": 0.1},
                {"throughput": 110.0, "latency": 0.12}
            ],
            "scheduling_metrics": {
                "strategy": "round_robin",
                "avg_wait_time": 0.1,
                "avg_queue_length": 2.0
            },
            "tradeoff_results": {
                "weights": [0.2, 0.5, 0.8],
                "values": [
                    {
                        "throughput": metrics["throughput"],
                        "latency": metrics["latency"],
                        "energy": metrics["energy"],
                        "runtime": metrics["runtime"]
                    }
                ]
            }
        }

    def cleanup(self):
        """清理测试环境。"""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

class TestModelBenchmarking(ModelBenchmarking):
    """测试模型基准测试类。"""
    
    @classmethod
    def setup_class(cls):
        """类级别的设置，在类初始化时调用一次。"""
        pass
    
    def setup_method(self, method):
        """在每个测试方法前设置测试环境。
        
        Args:
            method: 要运行的测试方法
        """
        super().__init__({})
        self.dataset_path = None
        self.output_dir = None

    def _init_components(self):
        """初始化测试组件。"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")

        # 验证数据集格式
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            try:
                dataset = json.load(f)
                if not isinstance(dataset, list):
                    raise ValueError("数据集必须是JSON数组格式")
                if len(dataset) == 0:
                    raise ValueError("数据集不能为空")
                for item in dataset:
                    if not all(key in item for key in ["input", "output", "tokens"]):
                        raise ValueError("数据集中的每个项目必须包含input、output和tokens字段")
            except json.JSONDecodeError:
                raise ValueError(f"数据集 {self.dataset_path} 不是有效的JSON格式")

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def get_metrics(self):
        """获取测试指标。"""
        return {
            "avg_runtime": 0.1,
            "avg_energy_per_token": 0.01,
            "accuracy": 0.95
        }

    def run_benchmarks(self):
        """运行基准测试。"""
        self._init_components()
        metrics = self.get_metrics()
        return {
            "metrics": metrics,
            "tradeoff_results": {
                "weights": [0.3, 0.3, 0.4],
                "values": [metrics]
            }
        }

    def cleanup(self):
        """清理测试环境。"""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)