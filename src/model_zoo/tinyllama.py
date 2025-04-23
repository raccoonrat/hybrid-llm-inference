from .base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from toolbox.logger import get_logger

class LocalTinyLlama(BaseModel):
    """本地 TinyLlama 模型实现"""
    
    def __init__(self, model_name, config):
        """初始化 TinyLlama 模型
        
        Args:
            model_name (str): 模型路径
            config (dict): 模型配置
        """
        super().__init__(model_name, config)
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.config = config
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=config.get("local_files_only", True)
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda",
                torch_dtype=torch.float16,
                local_files_only=config.get("local_files_only", True)
            )
        except Exception as e:
            self.logger.error(f"加载本地 TinyLlama 模型失败: {str(e)}")
            raise
            
    def infer(self, prompt):
        """执行推理
        
        Args:
            prompt (str): 输入提示
            
        Returns:
            str: 生成的文本
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_length", 512),
                do_sample=True,
                temperature=0.7
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"TinyLlama 推理失败: {str(e)}")
            raise
            
    def get_token_count(self, text):
        """获取文本的 token 数量
        
        Args:
            text (str): 输入文本
            
        Returns:
            int: token 数量
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.error(f"获取 token 数量失败: {str(e)}")
            raise 