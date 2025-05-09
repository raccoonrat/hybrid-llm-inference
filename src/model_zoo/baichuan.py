class BaichuanModel(BaseModel):
    def infer(self, input_text: str, **kwargs) -> str:
        if not input_text:
            raise ValueError("输入文本不能为空")
        return self._do_inference(input_text)

class LocalBaichuan(BaseModel):
    def __init__(self, model_name="baichuan-inc/Baichuan-13B-Chat", config=None):
        super().__init__(model_name, config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.accelerate = AccelerateWrapper(self.model, config)
        self.model = self.accelerate.get_model()
        self.logger.info(f"Loaded local Baichuan model: {model_name}")
    
    def infer(self, input_text: str, **kwargs) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=self.config.get("max_length", 512))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class APIBaichuan(BaseModel):
    def __init__(self, model_name="baichuan-inc/Baichuan-13B-Chat", config=None):
        super().__init__(model_name, config)
        self.api_key = config.get("api_key")
        self.api_url = config.get("api_url", "https://api-inference.huggingface.co/models/" + model_name)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.logger.info(f"Initialized API Baichuan model: {model_name}")
    
    def infer(self, input_text: str, **kwargs) -> str:
        payload = {"inputs": input_text, "parameters": {"max_length": self.config.get("max_length", 512)}}
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"] 