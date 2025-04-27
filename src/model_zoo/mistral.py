# hybrid-llm-inference/src/model_zoo/mistral.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from model_zoo.base_model import BaseModel
from toolbox.logger import get_logger
from toolbox.accelerate_wrapper import AccelerateWrapper
import os
import torch
from typing import Dict, Any, Optional, List, Generator
import time
import json

logger = get_logger(__name__)

class Mistral(BaseModel):
    """Mistral 模型实现。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 Mistral 模型。

        Args:
            config: 配置字典，包含：
                - model_path: 模型路径
                - device: 设备类型
                - dtype: 数据类型
        """
        super().__init__(config)
        self.model_path = config.get("model_path")
        self.device = config.get("device", "cuda")
        self.dtype = config.get("dtype", "float16")
        self.initialized = False
        
        # 验证配置
        self._validate_config()
        
        logger.info("Mistral 模型初始化完成")
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not self.config:
            raise ValueError("配置不能为空")
            
        if not self.model_path:
            raise ValueError("模型路径不能为空")
            
        if not os.path.exists(self.model_path):
            raise ValueError(f"模型路径不存在: {self.model_path}")
            
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("设备类型必须是 'cuda' 或 'cpu'")
            
        if self.dtype not in ["float16", "float32"]:
            raise ValueError("数据类型必须是 'float16' 或 'float32'")
    
    def initialize(self) -> None:
        """初始化模型。"""
        if self.initialized:
            return
            
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.dtype == "float16" else torch.float32,
                device_map=self.device
            )
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            self.initialized = True
            logger.info("Mistral 模型初始化完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _do_inference(self, input_text: str) -> str:
        """执行推理。

        Args:
            input_text: 输入文本

        Returns:
            输出文本
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化")
            
        try:
            # 编码输入
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # 生成输出
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 解码输出
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return output_text
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """获取文本的令牌数。

        Args:
            text: 输入文本

        Returns:
            令牌数
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化")
            
        try:
            # 编码文本
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # 获取令牌数
            token_count = inputs.input_ids.shape[1]
            
            return token_count
        except Exception as e:
            logger.error(f"获取令牌数失败: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。

        Returns:
            性能指标字典
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化")
            
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype
        }
    
    def cleanup(self) -> None:
        """清理资源。"""
        if hasattr(self, "model"):
            del self.model
            
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            
        self.initialized = False
        logger.info("Mistral 模型清理完成")

class LocalMistral(BaseModel):
    """Mistral模型的本地实现。
    
    属性:
        model: Mistral模型实例
        tokenizer: Mistral分词器实例
        device: 运行设备 ('cuda' 或 'cpu')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Mistral模型。
        
        Args:
            config: 模型配置，必须包含:
                - model_name: 模型名称 (默认: 'mistralai/Mistral-7B-v0.1')
                - model_path: 模型路径
                - device: 运行设备 ('cuda' 或 'cpu')
                - batch_size: 批处理大小
                - max_length: 最大序列长度
        """
        super().__init__(config)
        
        # 设置设备
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        if not self.is_test_mode:
            try:
                model_name = config.get("model_name", "mistralai/Mistral-7B-v0.1")
                self.logger.info(f"加载模型: {model_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                    
                self.logger.info("模型加载完成")
            except Exception as e:
                self.logger.error(f"模型加载失败: {str(e)}")
                raise RuntimeError(f"模型加载失败: {str(e)}")
                
    def _do_inference(self, input_text: str) -> str:
        """执行Mistral模型的推理。
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 生成的文本
        """
        if self.is_test_mode:
            return f"测试输出: {input_text}"
            
        try:
            # 准备输入
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)
            
            # 生成输出
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_length"],
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
            # 解码输出
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return output_text
            
        except Exception as e:
            self.logger.error(f"推理失败: {str(e)}")
            raise RuntimeError(f"推理失败: {str(e)}")
            
    def get_token_count(self, text: str) -> int:
        """获取文本的token数量。
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
        """
        if self.is_test_mode:
            return len(text.split())
            
        try:
            tokens = self.tokenizer.encode(text, return_tensors="pt")
            return tokens.shape[1]
        except Exception as e:
            self.logger.error(f"计算token数量失败: {str(e)}")
            return len(text.split())
            
    def cleanup(self) -> None:
        """清理模型资源。"""
        if not self.is_test_mode and hasattr(self, "model"):
            try:
                del self.model
                del self.tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("模型资源已清理")
            except Exception as e:
                self.logger.error(f"清理资源失败: {str(e)}")

class APIMistral(BaseModel):
    """Mistral API 模型类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 Mistral API 模型。

        Args:
            config: 配置字典，包含：
                - api_key: API 密钥
                - model_name: 模型名称
                - max_length: 最大生成长度
                - max_retries: 最大重试次数
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "mistralai/Mistral-7B-v0.1")
        self.max_length = config.get("max_length", 512)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.base_url = "https://api.mistral.ai/v1"
        
        # 验证配置
        self._validate_config()
        
        # 初始化模型
        self._init_model()
        
        logger.info("Mistral API 模型初始化完成")
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not self.config:
            raise ValueError("配置不能为空")
            
        if not self.api_key:
            raise ValueError("API 密钥不能为空，请提供有效的 Mistral API 密钥")
            
        if not isinstance(self.max_length, int) or self.max_length <= 0:
            raise ValueError("最大生成长度必须是正整数")
            
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("最大重试次数必须是非负整数")
            
        if not isinstance(self.retry_delay, (int, float)) or self.retry_delay <= 0:
            raise ValueError("重试延迟必须是正数")
    
    def _init_model(self) -> None:
        """初始化模型。"""
        try:
            # 验证 API 密钥
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            if response.status_code == 401:
                raise ValueError("API 密钥无效，请检查您的 Mistral API 密钥")
            elif response.status_code == 403:
                raise ValueError("API 密钥权限不足，请检查您的 API 密钥权限")
            elif response.status_code != 200:
                raise RuntimeError(f"API 密钥验证失败: {response.status_code}")
                
            logger.info("Mistral API 模型初始化完成")
        except requests.exceptions.RequestException as e:
            logger.error(f"API 请求失败: {e}")
            raise RuntimeError("无法连接到 Mistral API 服务器，请检查网络连接")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def inference(self, input_text: str, max_tokens: Optional[int] = None) -> str:
        """执行推理。

        Args:
            input_text: 输入文本
            max_tokens: 最大生成令牌数

        Returns:
            生成的文本
        """
        if not self.initialized:
            raise RuntimeError("模型未初始化")
            
        try:
            # 设置最大令牌数
            max_tokens = max_tokens or self.max_length
            
            # 准备请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": input_text}],
                "max_tokens": max_tokens
            }
            
            # 发送请求
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    elif response.status_code == 429:  # 速率限制
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                            continue
                        raise RuntimeError("达到 API 速率限制，请稍后再试")
                    elif response.status_code == 401:
                        raise ValueError("API 密钥无效，请检查您的 Mistral API 密钥")
                    elif response.status_code == 403:
                        raise ValueError("API 密钥权限不足，请检查您的 API 密钥权限")
                    else:
                        response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    raise RuntimeError(f"API 请求失败: {e}")
                    
            raise RuntimeError("达到最大重试次数")
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """获取文本的令牌数。

        Args:
            text: 输入文本

        Returns:
            令牌数
        """
        try:
            # 准备请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "text": text
            }
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/tokenize",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return len(result["tokens"])
            else:
                response.raise_for_status()
        except Exception as e:
            logger.error(f"获取令牌数失败: {e}")
            # 如果 API 调用失败，使用简单的分词方法作为后备
            return len(text.split())
    
    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False
        logger.info("Mistral API 模型清理完成")

