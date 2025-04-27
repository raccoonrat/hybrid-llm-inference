"""任务分配器基类模块。"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from toolbox.logger import get_logger

logger = get_logger(__name__)

class BaseAllocator(ABC):
    """任务分配器基类，定义所有任务分配器必须实现的接口。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化任务分配器。

        Args:
            config: 任务分配器配置
        """
        self.config = config
        self.initialized = False
    
    def initialize(self) -> None:
        """初始化任务分配器。"""
        if self.initialized:
            return
            
        self._init_allocator()
        self.initialized = True
        logger.info("任务分配器初始化完成")
    
    @abstractmethod
    def _init_allocator(self) -> None:
        """初始化任务分配器的具体实现。"""
        pass
    
    @abstractmethod
    def allocate(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分配任务。

        Args:
            tasks: 待分配的任务列表

        Returns:
            分配后的任务列表，每个任务包含分配的硬件和模型信息
        """
        pass
    
    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False
        logger.info("任务分配器清理完成") 