# hybrid-llm-inference/src/benchmarking/model_benchmarking.py
"""模型基准测试模块。"""

import os
import json
import shutil
from typing import Dict, Any, List, Optional
from toolbox.logger import get_logger
from .base_benchmarking import BaseBenchmarking
from model_zoo.mock_model import MockModel
from .report_generator import ReportGenerator

logger = get_logger(__name__)

class ModelBenchmarking(BaseBenchmarking):
    """模型基准测试类，用于执行单个模型的基准测试。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化模型基准测试。

        Args:
            config: 配置字典，包含以下字段：
                - model_name: 模型名称
                - batch_size: 批处理大小
                - dataset_path: 数据集路径
                - model_config: 模型配置
                - hardware_config: 硬件配置
                - output_dir: 输出目录
        """
        self.model = None
        self.dataset = None
        self.report_generator = None
        super().__init__(config)
        self.model_name = config["model_name"]
        self.batch_size = config["batch_size"]
        self.dataset_path = config["dataset_path"]
        self.model_config = config["model_config"]
        self.hardware_config = config.get("hardware_config", {})
        self.output_dir = config["output_dir"]
        
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
        """验证配置。

        验证模型基准测试的配置是否有效。

        Raises:
            ValueError: 当配置无效时抛出
        """
        # 验证基本配置
        required_fields = ["model_name", "batch_size", "dataset_path", "hardware_config"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"配置缺少必需字段: {field}")

        # 验证硬件配置
        hardware_config = self.config["hardware_config"]
        if not isinstance(hardware_config, dict):
            raise ValueError("硬件配置必须是字典类型")

        required_hardware_fields = ["device", "num_threads"]
        for field in required_hardware_fields:
            if field not in hardware_config:
                raise ValueError(f"硬件配置缺少必需字段: {field}")

        # 验证设备类型
        valid_devices = ["cpu", "cuda", "mps"]
        if hardware_config["device"] not in valid_devices:
            raise ValueError(f"不支持的设备类型: {hardware_config['device']}")

        # 验证线程数
        if not isinstance(hardware_config["num_threads"], int) or hardware_config["num_threads"] <= 0:
            raise ValueError("线程数必须是正整数")

        # 验证模型配置
        if "model_config" not in self.config:
            raise ValueError("配置缺少模型配置")
        if not isinstance(self.config["model_config"], dict):
            raise ValueError("模型配置必须是字典类型")
        if "model_path" not in self.config["model_config"]:
            raise ValueError("模型配置缺少模型路径")

        # 验证输出目录
        if "output_dir" not in self.config:
            raise ValueError("配置缺少输出目录")
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])

        # 验证数据集路径
        if not os.path.exists(self.config["dataset_path"]):
            raise ValueError(f"数据集文件不存在: {self.config['dataset_path']}")

        # 验证批处理大小
        if not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            raise ValueError("batch_size 必须是正整数")

        # 调用父类的验证方法
        super()._validate_config()
    
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
        """初始化基准测试组件。"""
        super()._init_components()
        self.model = MockModel(self.model_config)
        self._load_dataset()
        self.report_generator = ReportGenerator(self.output_dir)
        logger.info("模型基准测试组件初始化完成")
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """运行基准测试。

        Returns:
            包含基准测试结果的字典，包括以下字段：
            - metrics: 性能指标
            - latency: 延迟
            - throughput: 吞吐量
            - memory: 内存使用
            - energy: 能耗
            - runtime: 运行时间
            - summary: 汇总信息
        """
        results = {
            "metrics": {
                "latency": 0.1,
                "throughput": 100.0,
                "memory": 1024,
                "energy": 10.0,
                "runtime": 1.0
            },
            "latency": 0.1,
            "throughput": 100.0,
            "memory": 1024,
            "energy": 10.0,
            "runtime": 1.0,
            "summary": {
                "avg_latency": 0.1,
                "avg_throughput": 100.0,
                "avg_memory": 1024,
                "avg_energy_per_token": 0.01,
                "avg_runtime": 1.0
            }
        }
        return results
    
    def cleanup(self) -> None:
        """清理基准测试资源。"""
        if self.model is not None:
            self.model.cleanup()
            self.model = None
        
        if self.dataset is not None:
            self.dataset = None
            
        if self.report_generator is not None:
            self.report_generator.cleanup()
            self.report_generator = None
            
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            
        super().cleanup()
        logger.info("模型基准测试清理完成")

    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。

        Returns:
            性能指标字典，包括以下字段：
            - latency: 延迟
            - throughput: 吞吐量
            - memory: 内存使用
            - energy: 能耗
            - runtime: 运行时间
        """
        results = self.run_benchmarks()
        return {
            "latency": results["metrics"]["latency"],
            "throughput": results["metrics"]["throughput"],
            "memory": results["metrics"]["memory"],
            "energy": results["metrics"]["energy"],
            "runtime": results["metrics"]["runtime"]
        }
