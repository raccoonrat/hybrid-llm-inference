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
    """Mistral模型的API实现。
    
    属性:
        api_key: API密钥
        api_url: API端点URL
        headers: API请求头
        tokenizer: 用于token计数的分词器
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化API Mistral模型。
        
        Args:
            config: 模型配置，必须包含:
                - api_key: API密钥
                - api_url: API端点URL (可选)
                - model_name: 模型名称 (默认: 'mistralai/Mistral-7B-v0.1')
                - max_length: 最大序列长度
                - max_retries: 最大重试次数 (默认: 3)
                - retry_delay: 重试延迟（秒）(默认: 1)
        """
        super().__init__(config)
        
        # 验证API配置
        self.api_key = config.get("api_key")
        if not self.api_key and not self.is_test_mode:
            raise ValueError("API密钥是必需的")
            
        # 设置API端点
        model_name = config.get("model_name", "mistralai/Mistral-7B-v0.1")
        self.api_url = config.get("api_url", f"https://api-inference.huggingface.co/models/{model_name}")
        
        # 设置请求头
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        # 设置重试参数
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)
        
        # 初始化分词器用于token计数
        if not self.is_test_mode:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                self.logger.info("分词器加载完成")
            except Exception as e:
                self.logger.error(f"分词器加载失败: {str(e)}")
                raise RuntimeError(f"分词器加载失败: {str(e)}")
                
    def _make_api_request(self, payload: Dict[str, Any], stream: bool = False) -> Any:
        """发送API请求并处理重试逻辑。
        
        Args:
            payload: 请求数据
            stream: 是否使用流式输出
            
        Returns:
            Any: API响应
            
        Raises:
            RuntimeError: 当所有重试都失败时
        """
        for attempt in range(self.max_retries):
            try:
                if stream:
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        stream=True,
                        timeout=30
                    )
                else:
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=30
                    )
                    
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"API请求失败，已重试{self.max_retries}次: {str(e)}")
                self.logger.warning(f"API请求失败，第{attempt + 1}次重试: {str(e)}")
                time.sleep(self.retry_delay)
                
    def _do_inference(self, input_text: str) -> str:
        """通过API执行Mistral模型的推理。
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 生成的文本
            
        Raises:
            RuntimeError: 当API调用失败时
        """
        if self.is_test_mode:
            return f"测试输出: {input_text}"
            
        try:
            # 准备请求数据
            payload = {
                "inputs": input_text,
                "parameters": {
                    "max_new_tokens": self.config["max_length"],
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            # 发送请求
            response = self._make_api_request(payload)
            result = response.json()
            
            # 提取生成的文本
            if isinstance(result, list) and len(result) > 0:
                return result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                raise RuntimeError("API响应格式无效")
                
        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            raise RuntimeError(f"API调用失败: {str(e)}")
            
    def batch_inference(self, input_texts: List[str]) -> List[str]:
        """批量执行推理。
        
        Args:
            input_texts: 输入文本列表
            
        Returns:
            List[str]: 生成的文本列表
            
        Raises:
            ValueError: 当输入列表为空时
            RuntimeError: 当API调用失败时
        """
        if not input_texts:
            raise ValueError("输入文本列表不能为空")
            
        if self.is_test_mode:
            return [f"测试输出: {text}" for text in input_texts]
            
        try:
            # 准备批量请求数据
            payload = {
                "inputs": input_texts,
                "parameters": {
                    "max_new_tokens": self.config["max_length"],
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            # 发送请求
            response = self._make_api_request(payload)
            result = response.json()
            
            # 提取生成的文本
            if isinstance(result, list):
                return [item["generated_text"] for item in result]
            else:
                raise RuntimeError("API响应格式无效")
                
        except Exception as e:
            self.logger.error(f"批量推理失败: {str(e)}")
            raise RuntimeError(f"批量推理失败: {str(e)}")
            
    def stream_inference(self, input_text: str) -> Generator[str, None, None]:
        """流式执行推理。
        
        Args:
            input_text: 输入文本
            
        Yields:
            str: 生成的文本片段
            
        Raises:
            RuntimeError: 当API调用失败时
        """
        if self.is_test_mode:
            yield f"测试输出: {input_text}"
            return
            
        try:
            # 准备请求数据
            payload = {
                "inputs": input_text,
                "parameters": {
                    "max_new_tokens": self.config["max_length"],
                    "temperature": 0.7,
                    "do_sample": True,
                    "stream": True
                }
            }
            
            # 发送流式请求
            response = self._make_api_request(payload, stream=True)
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    try:
                        result = json.loads(line)
                        if "generated_text" in result:
                            yield result["generated_text"]
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            self.logger.error(f"流式推理失败: {str(e)}")
            raise RuntimeError(f"流式推理失败: {str(e)}")
            
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
        """清理资源。"""
        if not self.is_test_mode and hasattr(self, "tokenizer"):
            try:
                del self.tokenizer
                self.logger.info("API资源已清理")
            except Exception as e:
                self.logger.error(f"清理资源失败: {str(e)}")

