"""Token处理模块。"""

from typing import List, Union, Dict
import logging
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

logging.basicConfig(level=logging.INFO)

class MockTokenizer:
    """测试模式下使用的模拟分词器。"""
    
    def __init__(self):
        self.vocab_size = 1000
        
    def encode(self, text: str, **kwargs) -> List[int]:
        """模拟编码过程"""
        return [ord(c) for c in text]
        
    def decode(self, tokens: List[int], **kwargs) -> str:
        """模拟解码过程"""
        return ''.join(chr(t) for t in tokens)

class TokenProcessor:
    """Token处理器类。
    
    用于处理文本的tokenization。
    """
    
    DEFAULT_MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0"
    DEFAULT_MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0"
    
    def __init__(self, model_path: str):
        """初始化TokenProcessor。
        
        Args:
            model_path: 模型路径
        """
        self.logger = logging.getLogger(__name__)
        
        if not model_path or not isinstance(model_path, str):
            raise ValueError("模型路径必须是非空字符串")
            
        if os.getenv('TEST_MODE') == '1':
            self.tokenizer = MockTokenizer()
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                raise RuntimeError(f"加载模型失败：{str(e)}")
                
    def process_text(self, text: str) -> Dict[str, Union[List[int], str]]:
        """处理单个文本。
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Union[List[int], str]]: 包含token和解码文本的字典
        """
        try:
            tokens = self.tokenizer.encode(text, return_tensors='pt')[0].tolist()
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
        
    def batch_process(self, texts: List[str]) -> pd.DataFrame:
        """批量处理多个文本。
        
        Args:
            texts: 输入文本列表
            
        Returns:
            pd.DataFrame: 每个文本的处理结果
        """
        try:
            results = [self.process_text(text) for text in texts]
            return pd.DataFrame(results)
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