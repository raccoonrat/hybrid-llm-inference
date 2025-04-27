"""基准测试基类模块。"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import os
import shutil
from toolbox.logger import get_logger
from toolbox.config_manager import ConfigManager
import json

logger = get_logger(__name__)

class BaseBenchmarking(ABC):
    """基准测试基类，定义所有基准测试必须实现的接口。"""
    
    REQUIRED_CONFIG_FIELDS = ["model_name", "batch_size", "model_config", "hardware_config"]
    REQUIRED_METRICS = ["latency", "energy"]
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化基准测试。

        Args:
            config: 配置字典
        """
        self.validate_config(config)
        self.config = config
        self.config_manager = ConfigManager(config)
        self.logger = get_logger(__name__)
        
        # 获取基础配置
        self.dataset_path = self.config_manager.get_dataset_path()
        self.output_dir = self.config_manager.get_output_dir()
        self.hardware_config = self.config_manager.get_hardware_config()
        self.model_config = self.config_manager.get_model_config()
        self.scheduler_config = self.config_manager.get_scheduler_config()
        self.initialized = False
        self.resources = []  # 用于跟踪需要清理的资源
        
        # 验证基础配置
        self._validate_base_config()
        
        # 验证特定配置
        self._validate_config()
        
        # 初始化组件
        self._init_components()
        
        logger.info("基准测试初始化完成")
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """验证基准测试配置。
        
        Args:
            config: 基准测试配置
            
        Raises:
            ValueError: 当配置无效时
        """
        # 检查必需字段
        for field in self.REQUIRED_CONFIG_FIELDS:
            if field not in config:
                raise ValueError(f"配置缺少必需字段: {field}")
                
        # 验证 batch_size
        if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
            raise ValueError("batch_size 必须大于 0")
            
        # 验证 model_config
        if not isinstance(config["model_config"], dict):
            raise ValueError("model_config 必须是字典类型")
            
        # 验证 hardware_config
        if not isinstance(config["hardware_config"], dict):
            raise ValueError("hardware_config 必须是字典类型")
            
        # 验证数据集
        if "dataset" not in config and "dataset_path" not in config:
            raise ValueError("配置必须包含 dataset 或 dataset_path")
            
        if "dataset" in config:
            if not isinstance(config["dataset"], list):
                raise ValueError("dataset 必须是列表类型")
            if not config["dataset"]:
                raise ValueError("数据集不能为空")
            for item in config["dataset"]:
                if not isinstance(item, dict):
                    raise ValueError("数据集中的每个项目必须是字典类型")
        
        if "dataset_path" in config:
            if not os.path.exists(config["dataset_path"]):
                raise ValueError(f"数据集路径不存在: {config['dataset_path']}")
        
        # 设置默认值
        config.setdefault("num_threads", 1)
        config.setdefault("device", "cpu")
        config.setdefault("metrics", ["latency", "energy", "throughput"])
        
    def _validate_base_config(self) -> None:
        """验证基础配置。"""
        # 如果配置中同时存在dataset和dataset_path，优先使用dataset
        if "dataset" not in self.config and not self.config_manager.get_dataset_path():
            raise ValueError("配置必须包含 dataset 或 dataset_path")
            
        if "dataset" in self.config:
            if not isinstance(self.config["dataset"], list):
                raise ValueError("dataset 必须是列表类型")
            if not self.config["dataset"]:
                raise ValueError("dataset 不能为空")
        elif self.config_manager.get_dataset_path():
            dataset_path = self.config_manager.get_dataset_path()
            if not os.path.exists(dataset_path) and not os.environ.get('TEST_MODE'):
                raise ValueError(f"数据集路径不存在: {dataset_path}")
                
        if not self.config_manager.get_output_dir():
            raise ValueError("output_dir 不能为空")
        if not self.hardware_config:
            raise ValueError("hardware_config 不能为空")
        if not self.model_config:
            raise ValueError("model_config 不能为空")
        
        # 验证模型配置
        if "model_path" in self.model_config:
            model_path = self.model_config["model_path"]
            if not isinstance(model_path, str):
                raise ValueError("model_path 必须是字符串类型")
            # 在测试模式下,允许模型路径不存在
            if not os.path.exists(model_path) and not os.environ.get('TEST_MODE'):
                raise ValueError(f"模型路径不存在: {model_path}")
    
    def register_resource(self, resource_path: str) -> None:
        """注册需要清理的资源。

        Args:
            resource_path: 资源路径
        """
        self.resources.append(resource_path)
    
    def cleanup(self) -> None:
        """清理资源。"""
        # 清理注册的资源
        for resource in self.resources:
            try:
                if os.path.isfile(resource):
                    os.remove(resource)
                elif os.path.isdir(resource):
                    shutil.rmtree(resource)
            except Exception as e:
                logger.error(f"清理资源失败: {str(e)}")
        
        # 清理输出目录
        if hasattr(self, 'output_dir') and os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                logger.error(f"清理输出目录失败: {str(e)}")
        
        # 清理临时文件和目录
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                self.register_resource(self.temp_file)
            except Exception as e:
                logger.error(f"清理临时文件失败: {str(e)}")
                
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.register_resource(self.temp_dir)
            except Exception as e:
                logger.error(f"清理临时目录失败: {str(e)}")
        
        self.resources.clear()
        self.initialized = False
        logger.info("基准测试清理完成")
    
    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置。

        由子类实现，用于验证特定基准测试的配置。
        
        Raises:
            NotImplementedError: 当子类没有实现此方法时
        """
        raise NotImplementedError("子类必须实现 _validate_config 方法")
    
    @abstractmethod
    def _init_components(self) -> None:
        """初始化组件。

        由子类实现，用于初始化特定基准测试的组件。
        
        Raises:
            NotImplementedError: 当子类没有实现此方法时
        """
        raise NotImplementedError("子类必须实现 _init_components 方法")
    
    @abstractmethod
    def run_benchmarks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行基准测试。

        Args:
            tasks: 待测试的任务列表

        Returns:
            基准测试结果
            
        Raises:
            NotImplementedError: 当子类没有实现此方法时
        """
        raise NotImplementedError("子类必须实现 run_benchmarks 方法")
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。

        Returns:
            性能指标字典
            
        Raises:
            NotImplementedError: 当子类没有实现此方法时
        """
        raise NotImplementedError("子类必须实现 get_metrics 方法")

    def validate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证指标数据
        
        Args:
            metrics: 指标数据
            
        Returns:
            验证后的指标数据
            
        Raises:
            ValueError: 当指标数据无效时
        """
        # 检查必需的指标
        for metric in self.REQUIRED_METRICS:
            if metric not in metrics:
                raise ValueError(f"缺少必需的指标: {metric}")
                
        # 验证延迟指标
        if not isinstance(metrics["latency"], (list, tuple)):
            raise ValueError("latency 指标必须是列表或元组类型")
            
        # 验证能耗指标
        if not isinstance(metrics["energy"], (list, tuple)):
            raise ValueError("energy 指标必须是列表或元组类型")
            
        return metrics
        
    def run(self) -> None:
        """运行基准测试。
        
        Raises:
            NotImplementedError: 当子类没有实现此方法时
        """
        raise NotImplementedError("子类必须实现 run 方法")
        
    def collect_metrics(self) -> Dict[str, Any]:
        """收集性能指标。
        
        Returns:
            性能指标字典
            
        Raises:
            NotImplementedError: 当子类没有实现此方法时
        """
        raise NotImplementedError("子类必须实现 collect_metrics 方法") 