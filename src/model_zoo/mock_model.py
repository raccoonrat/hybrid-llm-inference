"""
用于测试的模拟模型类。
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from toolbox.logger import get_logger

logger = get_logger(__name__)

class MockModel:
    """
    模拟模型类，用于测试目的。
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化模拟模型。

        Args:
            config: 配置字典
        """
        self.config = config
        self.hidden_size = config.get("hidden_size", 256)
        self.intermediate_size = config.get("intermediate_size", 2048)
        self.device = config.get("device", "cuda")
        self.dtype = config.get("dtype", torch.float32)
        self.batch_size = config.get("batch_size", 1)
        self.max_length = config.get("max_length", 2048)
        
        # 创建一个简单的线性层作为模拟
        self.linear = nn.Linear(self.hidden_size, self.intermediate_size)
        
        # 创建模拟的tokenizer
        self.tokenizer = MockTokenizer()
        
    def to(self, device: str) -> 'MockModel':
        """
        将模型移动到指定设备。

        Args:
            device: 目标设备

        Returns:
            移动后的模型
        """
        self.device = device
        return self
        
    def eval(self) -> 'MockModel':
        """
        将模型设置为评估模式。

        Returns:
            模型实例
        """
        return self
        
    def generate(self, input_text: str, **kwargs) -> str:
        """
        生成响应文本。

        Args:
            input_text: 输入文本
            **kwargs: 其他参数

        Returns:
            生成的响应文本
        """
        return "这是一个测试模式的响应。"

class MockTokenizer:
    """
    模拟的分词器类。
    """
    def __init__(self):
        self.pad_token_id = 0
        
    def encode(self, text: str, **kwargs) -> List[int]:
        """模拟编码过程"""
        return [1, 2, 3]  # 返回固定的token序列
        
    def decode(self, tokens: List[int], **kwargs) -> str:
        """模拟解码过程"""
        return "这是测试模式的解码结果。" 