"""Token处理模块。"""

from typing import List
import logging
import os
from pathlib import Path

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
    
    def __init__(self, model_name: str = None, model_path: str = None):
        """初始化Token处理器。
        
        Args:
            model_name: 使用的模型名称，默认为TinyLlama
            model_path: 模型路径，默认为models/TinyLlama目录
        """
        self.logger = logging.getLogger(__name__)
        
        # 设置默认值
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        
        try:
            # 检查是否在测试模式
            if os.environ.get("TEST_MODE", "").lower() == "true":
                self.logger.info("在测试模式下运行，使用模拟tokenizer")
                self.tokenizer = MockTokenizer()
            else:
                from transformers import AutoTokenizer
                # 检查本地模型路径
                model_dir = Path(self.model_path)
                if model_dir.exists():
                    self.logger.info(f"从本地加载tokenizer: {model_dir}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_dir,
                        trust_remote_code=True
                    )
                else:
                    self.logger.warning(f"本地模型不存在: {model_dir}，尝试从Hugging Face下载")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    
                self.logger.info(f"已加载tokenizer: {self.model_name}")
                
        except Exception as e:
            self.logger.error(f"加载tokenizer失败: {str(e)}")
            raise
            
    def process(self, text: str) -> List[str]:
        """处理文本并返回token列表。
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: token列表
        """
        try:
            # 对文本进行tokenization
            tokens = self.tokenizer.tokenize(text)
            return tokens
        except Exception as e:
            self.logger.error(f"Token处理失败: {str(e)}")
            raise
            
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