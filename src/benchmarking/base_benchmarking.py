"""基准测试基类模块。"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
import shutil
from toolbox.logger import get_logger
from toolbox.config_manager import ConfigManager

logger = get_logger(__name__)

class BaseBenchmarking(ABC):
    """基准测试基类，定义所有基准测试必须实现的接口。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化基准测试。

        Args:
            config: 配置字典
        """
        # 初始化配置管理器
        self.config_manager = ConfigManager(config)
        self.config = self.config_manager.get_config()
        
        # 获取基础配置
        self.dataset_path = self.config.get("dataset_path")
        self.output_dir = self.config.get("output_dir")
        self.hardware_config = self.config.get("hardware_config", {})
        self.model_config = self.config.get("model_config", {})
        self.scheduler_config = self.config.get("scheduler_config", {})
        self.initialized = False
        self.resources = []  # 用于跟踪需要清理的资源
        
        # 验证基础配置
        self._validate_base_config()
        
        self._init_components()
        
        logger.info("基准测试初始化完成")
    
    def _validate_base_config(self) -> None:
        """验证基础配置。"""
        if not self.config_manager.get_dataset_path():
            raise ValueError("dataset_path 不能为空")
        if not self.config_manager.get_output_dir():
            raise ValueError("output_dir 不能为空")
        if not self.hardware_config:
            raise ValueError("hardware_config 不能为空")
        if not self.model_config:
            raise ValueError("model_config 不能为空")
        
        # 验证数据集路径
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集路径不存在: {self.dataset_path}")
        
        # 验证模型配置
        if "model_path" in self.model_config:
            model_path = self.model_config["model_path"]
            if not isinstance(model_path, str):
                raise ValueError("model_path 必须是字符串类型")
            if not os.path.exists(model_path):
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
        """
        pass
    
    @abstractmethod
    def _init_components(self) -> None:
        """初始化组件。

        由子类实现，用于初始化特定基准测试的组件。
        """
        pass
    
    @abstractmethod
    def run_benchmarks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行基准测试。

        Args:
            tasks: 待测试的任务列表

        Returns:
            基准测试结果
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。

        Returns:
            性能指标字典
        """
        pass 