"""
用于测试的模拟模型类。
"""

from typing import Dict, Any, List, Optional
from .base_model import BaseModel
from toolbox.logger import get_logger
import logging
import torch
import os
import torch.nn as nn

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
        self.hidden_size = config.get("hidden_size", 2048)
        self.intermediate_size = config.get("intermediate_size", 5632)
        self.device = config.get("device", "cuda")
        self.dtype = config.get("dtype", torch.float32)
        self.batch_size = config.get("batch_size", 1)
        self.max_length = config.get("max_length", 2048)
        
        # 创建一个简单的线性层，使用配置中的维度
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 初始化权重
        with torch.no_grad():
            self.linear.weight.fill_(0.1)
            if self.linear.bias is not None:
                self.linear.bias.fill_(0.0)
        
        self.logger = logging.getLogger(__name__)
        self.model_path = config.get("model_path", "")
        
        # 设置默认响应文本
        self.response_text = "这是一个测试模式的响应。"
        
        # 创建模拟tokenizer
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0  # 设置pad_token_id为0
                self.eos_token_id = 1
                self.model_max_length = 2048
                self.vocab_size = 32000
                self.all_special_tokens = ["<s>", "</s>", "<pad>"]
                self.unk_token = "<unk>"
                self.pad_token = "<pad>"
                self.eos_token = "</s>"
                self.bos_token = "<s>"
            
            def __call__(self, text, return_tensors="pt", **kwargs):
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
            
            def decode(self, token_ids, skip_special_tokens=True, **kwargs):
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                return f"测试模式响应: {token_ids}"
            
            def encode(self, text, **kwargs):
                return [1, 2, 3]
            
            def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
                if isinstance(sequences, torch.Tensor):
                    sequences = sequences.tolist()
                return [f"测试模式响应: {seq}" for seq in sequences]
        
        self.tokenizer = MockTokenizer()
        self.logger.info("测试模式：模拟模型初始化完成")

    def generate(self, input_ids=None, attention_mask=None, max_length=None, temperature=0.7, do_sample=True, **kwargs):
        """生成文本。"""
        if isinstance(input_ids, str):
            return f"测试模式响应: {input_ids}"
            
        # 如果input_ids是字典（来自tokenizer的输出）
        if isinstance(input_ids, dict):
            input_ids = input_ids.get("input_ids")
            
        # 返回一个固定的张量作为生成结果
        batch_size = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1
        return torch.tensor([[1, 2, 3, 4, 5]] * batch_size)

    def to(self, device):
        """将模型移动到指定设备。"""
        self.device = device
        self.linear = self.linear.to(device)
        return self
        
    def eval(self):
        """设置为评估模式。"""
        self.linear.eval()
        return self
    
    def parameters(self):
        """返回模型参数。"""
        return self.linear.parameters()
    
    def state_dict(self):
        """返回模型状态字典。"""
        return {"linear.weight": self.linear.weight, "linear.bias": self.linear.bias}
        
    def get_token_count(self, text: str) -> int:
        """获取文本的token数量。"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        """处理模型调用。"""
        if input_ids is None:
            return None
        batch_size = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1
        seq_len = input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else 3
        return torch.ones((batch_size, seq_len, self.hidden_size), device=self.device)

    def _load_model(self) -> None:
        """加载模型。在测试模式下，这个方法只是一个空实现。"""
        if os.getenv('TEST_MODE') == '1':
            self.logger.info("测试模式：跳过模型加载")
            self.model = None
            self.tokenizer = None
            return

        try:
            # 在测试模式下不加载实际模型
            self.model = None
            self.tokenizer = None
            self.logger.info("模拟模型加载完成")
        except Exception as e:
            self.logger.error(f"模型加载失败：{str(e)}")
            raise

    def _validate_base_config(self) -> None:
        """验证基础配置。"""
        logger.debug("验证基础配置")
        pass

    def _validate_config(self) -> None:
        """验证配置。"""
        logger.debug("验证配置")
        pass

    def _init_model(self) -> None:
        """初始化模型。"""
        logger.debug("初始化模型")
        pass

    def inference(self, input_text: str, max_tokens: Optional[int] = None) -> str:
        """执行推理。

        Args:
            input_text: 输入文本
            max_tokens: 最大生成令牌数

        Returns:
            生成的文本
        """
        logger.debug(f"执行推理，输入文本: {input_text}, 最大令牌数: {max_tokens}")
        return self.response_text

    def cleanup(self) -> None:
        """清理资源。"""
        logger.debug("清理资源")
        pass

    def _do_inference(self, text: str) -> str:
        """
        执行模拟推理。

        Args:
            text: 输入文本

        Returns:
            str: 模拟的响应文本
        """
        logger.debug(f"执行模拟推理，输入文本长度: {len(text)}")
        return self.response_text

    def infer(self, text: str) -> str:
        """
        执行推理。

        Args:
            text: 输入文本

        Returns:
            str: 推理结果
        """
        if not text:
            logger.warning("输入文本为空")
            return ""
        return self._do_inference(text)

    def generate(self, input_text: str, max_tokens: Optional[int] = None) -> str:
        """生成文本。

        Args:
            input_text: 输入文本
            max_tokens: 最大生成令牌数

        Returns:
            str: 生成的文本
        """
        return f"Mock response for: {input_text}"

    def save(self, path: str) -> None:
        """保存模型。

        Args:
            path: 保存路径
        """
        # 创建一个空的状态字典
        state_dict = {}
        torch.save(state_dict, path)

    @classmethod
    def load(cls, path: str) -> "MockModel":
        """加载模型。

        Args:
            path: 模型路径

        Returns:
            加载的模型
        """
        model = cls(path)
        return model

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载模型状态。

        Args:
            state_dict: 模型状态字典
        """
        # 在测试模式下，我们不实际加载权重
        pass

    def eval(self):
        """设置模型为评估模式。"""
        pass

    def save_pretrained(self, save_directory: str) -> None:
        """保存模型。
        
        Args:
            save_directory: 保存目录
        """
        os.makedirs(save_directory, exist_ok=True)
        self.logger.info(f"模型已保存到 {save_directory}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x) 