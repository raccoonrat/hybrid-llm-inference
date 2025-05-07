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

class MockModel(BaseModel):
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
        
        # 创建一个简单的线性层，使用配置中的维度
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 初始化权重
        with torch.no_grad():
            self.linear.weight.fill_(0.1)
            if self.linear.bias is not None:
                self.linear.bias.fill_(0.0)
        
        self.logger = logging.getLogger(__name__)
        self.model_path = None
        self.device = "cuda"
        self.response_text = "这是一个模拟的响应。"
        self.token_multiplier = 1.5  # 用于模拟token计数
        
        super().__init__(config)

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

    def get_token_count(self, text: str) -> int:
        """
        获取文本的模拟token数量。

        Args:
            text: 输入文本

        Returns:
            int: 模拟的token数量
        """
        if not text:
            logger.warning("输入文本为空")
            return 0
        token_count = int(len(text) * self.token_multiplier)
        logger.debug(f"计算token数量: {token_count}")
        return token_count

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

    def state_dict(self) -> Dict[str, Any]:
        """获取模型状态。

        Returns:
            Dict[str, Any]: 模型状态字典
        """
        return {}

    def to(self, device: str) -> 'MockModel':
        """将模型移动到指定设备。

        Args:
            device: 目标设备

        Returns:
            移动后的模型实例
        """
        self.device = device
        return self

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