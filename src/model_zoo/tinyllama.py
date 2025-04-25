"""TinyLlama 模型实现。"""

import os
from typing import Dict, Any, Optional
from toolbox.logger import get_logger
from .base_model import BaseModel

logger = get_logger(__name__)

class TinyLlama(BaseModel):
    """TinyLlama 模型实现。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 TinyLlama 模型。

        Args:
            config: 配置字典，包含：
                - model_path: 模型路径
                - device: 设备类型
                - dtype: 数据类型
        """
        super().__init__(config)
        self.model_path = config.get("model_path")
        self.device = config.get("device", "cuda")
        self.dtype = config.get("dtype", "float16")
        self.initialized = False
        
        # 验证配置
        self._validate_config()
        
        logger.info("TinyLlama 模型初始化完成")
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not self.config:
            raise ValueError("配置不能为空")
            
        if not self.model_path:
            raise ValueError("模型路径不能为空")
            
        if not os.path.exists(self.model_path):
            raise ValueError(f"模型路径不存在: {self.model_path}")
            
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("设备类型必须是 'cuda' 或 'cpu'")
            
        if self.dtype not in ["float16", "float32"]:
            raise ValueError("数据类型必须是 'float16' 或 'float32'")
    
    def initialize(self) -> None:
        """初始化模型。"""
        if self.initialized:
            return
            
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.dtype == "float16" else torch.float32,
                device_map=self.device
            )
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            self.initialized = True
            logger.info("TinyLlama 模型初始化完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _do_inference(self, input_text: str) -> str:
        """执行推理。

        Args:
            input_text: 输入文本

        Returns:
            输出文本
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化")
            
        try:
            # 编码输入
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # 生成输出
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 解码输出
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return output_text
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """获取文本的令牌数。

        Args:
            text: 输入文本

        Returns:
            令牌数
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化")
            
        try:
            # 编码文本
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # 获取令牌数
            token_count = inputs.input_ids.shape[1]
            
            return token_count
        except Exception as e:
            logger.error(f"获取令牌数失败: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。

        Returns:
            性能指标字典
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化")
            
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype
        }
    
    def cleanup(self) -> None:
        """清理资源。"""
        if hasattr(self, "model"):
            del self.model
            
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            
        self.initialized = False
        logger.info("TinyLlama 模型清理完成") 