"""推理器基类模块。"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from toolbox.logger import get_logger

logger = get_logger(__name__)

class BaseInference(ABC):
    """推理器基类，定义所有推理器必须实现的接口。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化推理器。

        Args:
            config: 推理器配置
        """
        self.config = config
        self.initialized = False
    
    def initialize(self) -> None:
        """初始化推理器。"""
        if self.initialized:
            return
            
        self._init_inference()
        self.initialized = True
        logger.info("推理器初始化完成")
    
    @abstractmethod
    def _init_inference(self) -> None:
        """初始化推理器的具体实现。"""
        pass
    
    @abstractmethod
    def infer(self, input_text: str, max_tokens: Optional[int] = None) -> str:
        """执行推理。

        Args:
            input_text: 输入文本
            max_tokens: 最大生成token数

        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """获取推理性能指标。

        Returns:
            性能指标字典
        """
        pass
    
    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False
        logger.info("推理器清理完成") 