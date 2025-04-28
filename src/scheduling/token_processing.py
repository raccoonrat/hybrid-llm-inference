"""令牌处理模块。"""

from typing import List, Dict, Any, Optional
from toolbox.logger import get_logger

logger = get_logger(__name__)

class TokenProcessing:
    """令牌处理类，用于处理模型推理过程中的令牌。"""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """初始化令牌处理器。

        Args:
            model_name: 模型名称
            model_config: 模型配置
        """
        self.model_name = model_name
        self.model_config = model_config
        self.initialized = False
        self.tokenizer = None
        self.max_length = model_config.get('max_length', 2048)
    
    def initialize(self) -> None:
        """初始化令牌处理器。"""
        if self.initialized:
            return
            
        # 初始化模型特定的令牌处理逻辑
        self._init_model_specific_processing()
        self.initialized = True
        logger.info(f"已初始化模型 {self.model_name} 的令牌处理器")
    
    def _init_model_specific_processing(self) -> None:
        """初始化模型特定的令牌处理逻辑。"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.model_config.get('trust_remote_code', True),
                cache_dir=self.model_config.get('cache_dir', None)
            )
        except Exception as e:
            logger.error(f"初始化tokenizer失败: {str(e)}")
            raise RuntimeError(f"初始化tokenizer失败: {str(e)}")
    
    def process_tokens(self, tokens: List[int], max_new_tokens: Optional[int] = None) -> List[int]:
        """处理令牌序列。

        Args:
            tokens: 输入令牌序列
            max_new_tokens: 最大新令牌数量

        Returns:
            处理后的令牌序列
        """
        if not self.initialized:
            self.initialize()
            
        # 应用模型特定的令牌处理逻辑
        processed_tokens = self._apply_model_specific_processing(tokens)
        
        # 如果指定了最大新令牌数量，则截断令牌序列
        if max_new_tokens is not None:
            max_context_length = self.max_length - max_new_tokens
            if len(processed_tokens) > max_context_length:
                processed_tokens = processed_tokens[-max_context_length:]
                logger.warning(f"令牌序列已截断至 {max_context_length} 个令牌")
        
        return processed_tokens
    
    def _apply_model_specific_processing(self, tokens: List[int]) -> List[int]:
        """应用模型特定的令牌处理逻辑。

        Args:
            tokens: 输入令牌序列

        Returns:
            处理后的令牌序列
        """
        if not self.initialized:
            self.initialize()
            
        # 检查特殊令牌
        if self.tokenizer is not None:
            # 确保序列以正确的起始令牌开始
            if len(tokens) > 0 and tokens[0] != self.tokenizer.bos_token_id:
                tokens = [self.tokenizer.bos_token_id] + tokens
                
            # 确保序列以正确的结束令牌结束
            if len(tokens) > 0 and tokens[-1] != self.tokenizer.eos_token_id:
                tokens = tokens + [self.tokenizer.eos_token_id]
        
        return tokens
    
    def cleanup(self) -> None:
        """清理资源。"""
        try:
            # 清理tokenizer
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            self.initialized = False
            logger.info(f"已清理模型 {self.model_name} 的令牌处理器")
        except Exception as e:
            logger.warning(f"清理资源时出错: {str(e)}") 