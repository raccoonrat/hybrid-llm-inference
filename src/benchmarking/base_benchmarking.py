"""基准测试基类模块。"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from toolbox.logger import get_logger

logger = get_logger(__name__)

class BaseBenchmarking(ABC):
    """基准测试基类，定义所有基准测试必须实现的接口。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化基准测试。

        Args:
            config: 基准测试配置
        """
        self.config = config
        self.initialized = False
        
        # 验证基础配置
        self._validate_base_config()
        
        logger.info("基准测试初始化完成")
    
    def _validate_base_config(self) -> None:
        """验证基础配置。"""
        if not isinstance(self.config, dict):
            raise ValueError("配置必须是字典类型")
    
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
    
    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False
        logger.info("基准测试清理完成") 