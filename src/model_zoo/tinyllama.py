"""TinyLlama 模型模块。"""

import os
import logging
from typing import Dict, Any, Optional

from .base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from toolbox.logger import get_logger

logger = logging.getLogger(__name__)

class LocalTinyLlama(BaseModel):
    """本地 TinyLlama 模型类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化模型。
        
        Args:
            config: 配置字典，必须包含：
                - model_name: 模型名称
                - model_path: 模型路径
                - mode: 运行模式（必须是 "local"）
                - batch_size: 批处理大小
                - max_length: 最大长度
        """
        super().__init__(config)
        
        # 验证模式
        if config["mode"] != "local":
            raise ValueError("TinyLlama 只支持本地模式")
        
        # 设置模型路径
        self.model_path = config["model_path"]
        if not os.path.exists(self.model_path):
            raise ValueError(f"模型路径不存在: {self.model_path}")
        
        # 初始化模型和分词器
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self) -> None:
        """加载模型和分词器。"""
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto"
            )
            
            logger.info("TinyLlama 模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def _do_inference(self, input_text: str) -> str:
        """执行推理。
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 生成的文本
        """
        try:
            # 编码输入
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.config["max_length"],
                truncation=True
            )
            
            # 生成输出
            outputs = self.model.generate(
                **inputs,
                max_length=self.config["max_length"],
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"推理失败: {str(e)}")
            raise
    
    def infer(self, input_text: str) -> str:
        """执行推理。
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 生成的文本
        """
        if not input_text:
            raise ValueError("输入文本不能为空")
            
        return self._do_inference(input_text)
    
    def get_token_count(self, text: str) -> int:
        """获取文本的 token 数量。
        
        Args:
            text: 输入文本
            
        Returns:
            int: token 数量
        """
        if not text:
            return 0
            
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"获取 token 数量失败: {str(e)}")
            return 0
    
    def cleanup(self) -> None:
        """清理资源。"""
        if self.model:
            del self.model
            self.model = None
            
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None 