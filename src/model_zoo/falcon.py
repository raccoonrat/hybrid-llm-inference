# hybrid-llm-inference/src/model_zoo/falcon.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from model_zoo.base_model import BaseModel
from toolbox.logger import get_logger
from toolbox.accelerate_wrapper import AccelerateWrapper

class LocalFalcon(BaseModel):
    def __init__(self, model_name="tiiuae/falcon-7b", config=None):
        super().__init__(model_name, config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.accelerate = AccelerateWrapper(self.model, config)
        self.model = self.accelerate.get_model()
        self.logger.info(f"Loaded local Falcon model: {model_name}")
    
    def infer(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=self.config.get("max_length", 512))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_token_count(self, text):
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        return len(tokens[0])

class APIFalcon(BaseModel):
    def __init__(self, model_name="tiiuae/falcon-7b", config=None):
        super().__init__(model_name, config)
        self.api_key = config.get("api_key")
        self.api_url = config.get("api_url", "https://api-inference.huggingface.co/models/" + model_name)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.logger.info(f"Initialized API Falcon model: {model_name}")
    
    def infer(self, input_text):
        payload = {"inputs": input_text, "parameters": {"max_length": self.config.get("max_length", 512)}}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    
    def get_token_count(self, text):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokens = tokenizer.encode(text, return_tensors="pt")
        return len(tokens[0])

