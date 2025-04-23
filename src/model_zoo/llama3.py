# hybrid-llm-inference/src/model_zoo/llama3.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from model_zoo.base_model import BaseModel
from toolbox.logger import get_logger
from toolbox.accelerate_wrapper import AccelerateWrapper

class LocalLlama3(BaseModel):
    def __init__(self, model_name="meta-llama/Llama-3-8B", config=None):
        super().__init__(model_name, config or {})
        self.logger = get_logger(__name__)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.accelerate = AccelerateWrapper(self.model, config or {})
            self.model = self.accelerate.get_model()
            self.logger.info(f"Loaded local Llama-3 model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load local Llama-3 model: {e}")
            raise

    def infer(self, input_text):
        if not input_text:
            self.logger.warning("Empty input text provided")
            return ""
        
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=self.config.get("max_length", 512))
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise

    def get_token_count(self, text):
        if not text:
            self.logger.warning("Empty text provided for token counting")
            return 0
        
        try:
            tokens = self.tokenizer.encode(text, return_tensors="pt")
            return len(tokens[0])
        except Exception as e:
            self.logger.error(f"Token counting failed: {e}")
            raise

class APILlama3(BaseModel):
    def __init__(self, model_name="meta-llama/Llama-3-8B", config=None):
        super().__init__(model_name, config or {})
        self.logger = get_logger(__name__)
        self.api_key = config.get("api_key")
        if not self.api_key:
            self.logger.error("API key not provided")
            raise ValueError("API key is required for API mode")
        
        self.api_url = config.get("api_url", "https://api-inference.huggingface.co/models/" + model_name)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.logger.info(f"Initialized API Llama-3 model: {model_name}")

    def infer(self, input_text):
        if not input_text:
            self.logger.warning("Empty input text provided")
            return ""
        
        payload = {"inputs": input_text, "parameters": {"max_length": self.config.get("max_length", 512)}}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()[0]["generated_text"]
        except requests.RequestException as e:
            self.logger.error(f"API inference failed: {e}")
            raise

    def get_token_count(self, text):
        if not text:
            self.logger.warning("Empty text provided for token counting")
            return 0
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokens = tokenizer.encode(text, return_tensors="pt")
            return len(tokens[0])
        except Exception as e:
            self.logger.error(f"Token counting failed: {e}")
            raise
