# hybrid-llm-inference/src/model_zoo/llama3.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from model_zoo.base_model import BaseModel
from toolbox.logger import get_logger
from toolbox.accelerate_wrapper import AccelerateWrapper
from typing import Dict, Any

logger = get_logger(__name__)

class LocalLlama3(BaseModel):
    def __init__(self, model_name="meta-llama/Llama-3-8B", config=None):
        super().__init__(model_name, config or {})
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.accelerate = AccelerateWrapper(self.model, config or {})
            self.model = self.accelerate.get_model()
            logger.info(f"Loaded local Llama-3 model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load local Llama-3 model: {e}")
            raise

    def infer(self, input_text):
        if not input_text:
            logger.warning("Empty input text provided")
            return ""
        
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=self.config.get("max_length", 512))
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def get_token_count(self, text):
        if not text:
            logger.warning("Empty text provided for token counting")
            return 0
        
        try:
            tokens = self.tokenizer.encode(text, return_tensors="pt")
            return len(tokens[0])
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            raise

class APILlama3(BaseModel):
    def __init__(self, model_name="meta-llama/Llama-3-8B", config=None):
        super().__init__(model_name, config or {})
        self.api_key = config.get("api_key")
        if not self.api_key:
            logger.error("API key not provided")
            raise ValueError("API key is required for API mode")
        
        self.api_url = config.get("api_url", "https://api-inference.huggingface.co/models/" + model_name)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        logger.info(f"Initialized API Llama-3 model: {model_name}")

    def infer(self, input_text):
        if not input_text:
            logger.warning("Empty input text provided")
            return ""
        
        payload = {"inputs": input_text, "parameters": {"max_length": self.config.get("max_length", 512)}}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()[0]["generated_text"]
        except requests.RequestException as e:
            logger.error(f"API inference failed: {e}")
            raise

    def get_token_count(self, text):
        if not text:
            logger.warning("Empty text provided for token counting")
            return 0
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokens = tokenizer.encode(text, return_tensors="pt")
            return len(tokens[0])
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            raise

class Llama3Model(BaseModel):
    """Llama3 模型类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Llama3 模型。

        Args:
            config: 模型配置
        """
        super().__init__(config)
        
    def _do_inference(self, input_text: str) -> str:
        """
        执行实际的推理操作。

        Args:
            input_text: 输入文本

        Returns:
            str: 生成的文本
        """
        if not input_text:
            raise ValueError("输入文本不能为空")
            
        try:
            # 在测试模式下返回模拟响应
            if self.is_test_mode:
                return "这是一个模拟的 Llama3 响应。"
                
            # TODO: 实现实际的推理逻辑
            return "这是一个 Llama3 响应。"
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """
        获取文本的token数量。

        Args:
            text: 输入文本

        Returns:
            int: token数量
        """
        if not text:
            return 0
            
        try:
            # 在测试模式下返回固定值
            if self.is_test_mode:
                return len(text.split())
                
            # TODO: 实现实际的token计数逻辑
            return len(text.split())
        except Exception as e:
            logger.error(f"计算token数量失败: {e}")
            return 0
    
    def cleanup(self) -> None:
        """
        清理资源。
        """
        pass
