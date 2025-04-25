"""调度器基类模块。"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from toolbox.logger import get_logger

logger = get_logger(__name__)

class BaseScheduler(ABC):
    """调度器基类，定义所有调度器必须实现的接口。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化调度器。

        Args:
            config: 调度器配置
        """
        self.config = config
        self.initialized = False
    
    def initialize(self) -> None:
        """初始化调度器。"""
        if self.initialized:
            return
            
        self._init_scheduler()
        self.initialized = True
        logger.info("调度器初始化完成")
    
    @abstractmethod
    def _init_scheduler(self) -> None:
        """初始化调度器的具体实现。"""
        pass
    
    @abstractmethod
    def schedule(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """调度任务。

        Args:
            tasks: 待调度的任务列表

        Returns:
            调度后的任务列表
        """
        pass
    
    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False
        logger.info("调度器清理完成") 