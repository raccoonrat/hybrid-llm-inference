"""
用于测试的模拟模型类。
"""

from typing import Dict, Any, List
from .base_model import BaseModel

class MockModel(BaseModel):
    """
    模拟模型类，用于测试目的。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟模型。

        Args:
            config: 模型配置
        """
        super().__init__(config)
        self.response_text = "这是一个模拟的响应。"
        self.token_multiplier = 1.5  # 用于模拟token计数

    def _do_inference(self, text: str) -> str:
        """
        执行模拟推理。

        Args:
            text: 输入文本

        Returns:
            str: 模拟的响应文本
        """
        return self.response_text

    def infer(self, text: str) -> str:
        """
        执行推理。

        Args:
            text: 输入文本

        Returns:
            str: 推理结果
        """
        return self._do_inference(text)

    def get_token_count(self, text: str) -> int:
        """
        获取文本的模拟token数量。

        Args:
            text: 输入文本

        Returns:
            int: 模拟的token数量
        """
        return int(len(text) * self.token_multiplier)  # 简单地将文本长度乘以一个系数 