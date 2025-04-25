"""Token处理模块。"""

from typing import List, Union
import logging
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

class MockTokenizer:
    """测试模式下使用的模拟分词器。"""
    
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        
    def tokenize(self, text: str) -> List[str]:
        """简单的字符级分词。"""
        return list(text)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本转换为token ID。"""
        return [self.vocab.get(char, 1) for char in text]
        
    def decode(self, token_ids: List[int]) -> str:
        """将token ID转换回文本。"""
        return "".join([char for char in "".join([str(id) for id in token_ids])])
        
    def __len__(self) -> int:
        """返回词汇表大小。"""
        return len(self.vocab)

class TokenProcessor:
    """Token处理器类。
    
    用于处理文本的tokenization。
    """
    
    DEFAULT_MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0"
    DEFAULT_MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0"
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model_path: str = DEFAULT_MODEL_PATH):
        """初始化TokenProcessor。
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
        """
        self.logger = logging.getLogger(__name__)
        
        if not model_name or not isinstance(model_name, str):
            raise ValueError("模型名称必须是非空字符串")
        if not model_path or not isinstance(model_path, str):
            raise ValueError("模型路径必须是非空字符串")
            
        self.model_name = model_name
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在：{model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            self.logger.error(f"加载模型失败：{str(e)}")
            raise RuntimeError(f"加载模型失败：{str(e)}")
            
    def process(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """处理文本。
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            处理后的token ID列表
            
        Raises:
            ValueError: 输入无效
            RuntimeError: 处理失败
        """
        if text is None:
            raise ValueError("输入不能为None")
            
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("输入必须是字符串或字符串列表")
            
        if not all(isinstance(t, str) for t in text):
            raise ValueError("列表中所有元素必须是字符串")
            
        if not text:
            raise ValueError("输入不能为空")

        try:
            tokens = self.tokenizer(text, return_tensors="pt", padding=True)
            return tokens["input_ids"].tolist()
        except Exception as e:
            self.logger.error(f"Token处理失败: {str(e)}")
            raise RuntimeError(f"Token处理失败: {str(e)}")
            
    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID。
        
        Args:
            text: 输入文本
            
        Returns:
            List[int]: token ID列表
        """
        try:
            # 编码文本
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            return encoded
        except Exception as e:
            self.logger.error(f"文本编码失败: {str(e)}")
            raise
            
    def decode(self, token_ids: List[int]) -> str:
        """将token ID解码为文本。
        
        Args:
            token_ids: token ID列表
            
        Returns:
            str: 解码后的文本
        """
        try:
            # 解码token ID
            decoded = self.tokenizer.decode(token_ids)
            return decoded
        except Exception as e:
            self.logger.error(f"Token解码失败: {str(e)}")
            raise
            
    def get_vocab_size(self) -> int:
        """获取词汇表大小。
        
        Returns:
            int: 词汇表大小
        """
        return len(self.tokenizer)
        
    def batch_process(self, texts: List[str]) -> List[List[str]]:
        """批量处理多个文本。
        
        Args:
            texts: 输入文本列表
            
        Returns:
            List[List[str]]: 每个文本的token列表
        """
        try:
            return [self.process(text) for text in texts]
        except Exception as e:
            self.logger.error(f"批量处理失败: {str(e)}")
            raise
            
    def cleanup(self):
        """清理tokenizer资源。"""
        try:
            del self.tokenizer
            self.logger.info("已清理tokenizer资源")
        except Exception as e:
            self.logger.error(f"清理tokenizer失败: {str(e)}") 