"""Token处理模块。"""

from typing import List, Union, Dict, Optional
import logging
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

logging.basicConfig(level=logging.INFO)

class MockTokenizer:
    """测试模式下使用的模拟分词器。"""
    
    def __init__(self, max_length: Optional[int] = None):
        self.vocab_size = 1000
        self.max_length = max_length
        
    def encode(self, text: str, **kwargs) -> List[int]:
        """模拟编码过程"""
        if text is None:
            raise ValueError("输入文本不能为 None")
        tokens = [ord(c) for c in text]
        if self.max_length:
            tokens = tokens[:self.max_length]
        return tokens
        
    def decode(self, tokens: List[int], **kwargs) -> str:
        """模拟解码过程"""
        return ''.join(chr(t) for t in tokens)

class TokenProcessor:
    """Token处理器类。
    
    用于处理文本的tokenization。
    """
    
    DEFAULT_MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0"
    DEFAULT_MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0"
    
    def __init__(self, model_path: str, validate_path: bool = False, max_length: Optional[int] = None):
        """初始化TokenProcessor。
        
        Args:
            model_path: 模型路径
            validate_path: 是否验证模型路径存在
            max_length: 最大token长度
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.max_length = max_length
        
        if not model_path or not isinstance(model_path, str):
            raise ValueError("模型路径必须是非空字符串")
            
        if validate_path and not os.path.exists(model_path):
            raise ValueError("模型路径不存在")
            
        if os.getenv('TEST_MODE') == '1':
            self.tokenizer = MockTokenizer(max_length=max_length)
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                if max_length:
                    self.tokenizer.model_max_length = max_length
            except Exception as e:
                raise RuntimeError(f"加载模型失败：{str(e)}")
                
    def process(self, text: str) -> List[int]:
        """处理单个文本。
        
        Args:
            text: 输入文本
            
        Returns:
            List[int]: token列表
            
        Raises:
            ValueError: 当输入为 None 时
        """
        if text is None:
            raise ValueError("输入文本不能为 None")
        return self.tokenizer.encode(text)
                
    def process_text(self, text: str) -> Dict[str, Union[List[int], str]]:
        """处理单个文本。
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Union[List[int], str]]: 包含token和解码文本的字典
        """
        try:
            if text is None:
                raise ValueError("输入文本不能为 None")
            tokens = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(tokens)
            return {
                "input_tokens": tokens,
                "decoded_text": decoded
            }
        except Exception as e:
            self.logger.error(f"处理文本失败: {str(e)}")
            return {
                "input_tokens": [],
                "decoded_text": ""
            }
            
    def encode(self, text: str) -> List[int]:
        """编码文本。
        
        Args:
            text: 输入文本
            
        Returns:
            List[int]: token列表
            
        Raises:
            ValueError: 当输入为 None 时
        """
        if text is None:
            raise ValueError("输入文本不能为 None")
        return self.tokenizer.encode(text)
        
    def decode(self, tokens: List[int]) -> str:
        """解码token。
        
        Args:
            tokens: token列表
            
        Returns:
            str: 解码后的文本
        """
        return self.tokenizer.decode(tokens)
        
    def batch_process(self, texts: List[str]) -> List[List[int]]:
        """批量处理多个文本。
        
        Args:
            texts: 输入文本列表
            
        Returns:
            List[List[int]]: 每个文本的token列表
            
        Raises:
            ValueError: 当输入列表为 None 时
        """
        if texts is None:
            raise ValueError("输入文本列表不能为 None")
            
        try:
            return [self.process(text) for text in texts]
        except Exception as e:
            self.logger.error(f"批量处理失败: {str(e)}")
            raise
            
    def cleanup(self):
        """清理tokenizer资源。"""
        try:
            if not isinstance(self.tokenizer, MockTokenizer):
                del self.tokenizer
            self.logger.info("已清理tokenizer资源")
        except Exception as e:
            self.logger.error(f"清理tokenizer失败: {str(e)}") 