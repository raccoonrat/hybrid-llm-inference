"""
用于测试的模拟模型类。
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from toolbox.logger import get_logger

logger = get_logger(__name__)

class MockModel(nn.Module):
    """
    模拟模型类，用于测试目的。
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化模拟模型。

        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 2048)
        self.intermediate_size = config.get("intermediate_size", 5632)
        self.device = config.get("device", "cuda")
        self.dtype = config.get("dtype", torch.float32)
        self.batch_size = config.get("batch_size", 1)
        self.max_length = config.get("max_length", 2048)
        
        # 创建一个简单的线性层作为模拟
        self.linear = nn.Linear(256, self.hidden_size)
        
        # 创建模拟的tokenizer
        self.tokenizer = MockTokenizer()
        
        logger.info("MockModel 初始化完成")
        
    def to(self, device: str) -> 'MockModel':
        """
        将模型移动到指定设备。

        Args:
            device: 目标设备

        Returns:
            移动后的模型
        """
        self.device = device
        self.linear = self.linear.to(device)
        return self
        
    def eval(self) -> 'MockModel':
        """
        将模型设置为评估模式。

        Returns:
            模型实例
        """
        self.linear.eval()
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
        return f"测试模式响应: {input_text}"
        
    def get_token_count(self, text: str) -> int:
        """
        获取文本的token数量。

        Args:
            text: 输入文本

        Returns:
            token数量
        """
        return len(text.split())
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播。

        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            **kwargs: 其他参数

        Returns:
            模型输出
        """
        batch_size = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1
        seq_len = input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else 3
        return torch.ones((batch_size, seq_len, 256), device=self.device)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        获取模型状态字典。

        Returns:
            状态字典
        """
        return {
            'linear.weight': torch.ones((256, self.hidden_size)),
            'linear.bias': torch.zeros(self.hidden_size)
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        加载模型状态字典。在测试模式下，我们不需要实际加载权重。

        Args:
            state_dict: 模型状态字典
        """
        logger.info("测试模式：跳过加载状态字典")
        return

class MockTokenizer:
    """
    模拟的分词器类。
    """
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        
    def encode(self, text: str, **kwargs) -> List[int]:
        """模拟编码过程"""
        return [1, 2, 3]  # 返回固定的token序列
        
    def decode(self, tokens: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """模拟解码过程"""
        return "这是测试模式的解码结果。"
        
    def __call__(self, text: str, return_tensors: str = "pt", **kwargs) -> Dict[str, torch.Tensor]:
        """
        处理分词器调用。

        Args:
            text: 输入文本
            return_tensors: 返回张量的类型
            **kwargs: 其他参数

        Returns:
            分词结果字典
        """
        if isinstance(text, str):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]])
            }
        else:
            batch_size = len(text)
            return {
                "input_ids": torch.tensor([[1, 2, 3]] * batch_size),
                "attention_mask": torch.tensor([[1, 1, 1]] * batch_size)
            } 