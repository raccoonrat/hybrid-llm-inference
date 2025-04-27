"""TinyLlama 模型实现。"""

import os
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from toolbox.logger import get_logger
from toolbox.config_manager import ConfigManager
from .base_model import BaseModel

logger = get_logger(__name__)

class TinyLlama(BaseModel):
    """TinyLlama 模型类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 TinyLlama 模型。

        Args:
            config: 配置字典，包含：
                - model_path: 模型路径
                - device: 运行设备 (cpu/cuda)
                - dtype: 数据类型 (float32/float16/bfloat16)
                - max_memory: 最大显存使用量
        """
        # 调用父类构造函数
        super().__init__(config)
        
        self.dtype = config.get("dtype", "float32")
        self.max_memory = config.get("max_memory", None)
        
        # 验证配置
        self._validate_base_config()
        self._validate_config()
        
        # 初始化模型
        if os.getenv('TEST_MODE') != '1':
            self._init_model()
    
    def _validate_base_config(self) -> None:
        """验证基础配置。"""
        # 验证模型路径
        if not self.config.get("model_path"):
            raise ValueError("模型路径不能为空")
        
        # 验证设备
        if self.config.get("device") not in ["cpu", "cuda"]:
            raise ValueError("设备必须是 'cpu' 或 'cuda'")
            
        # 验证数据类型
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError("数据类型必须是 'float32'、'float16' 或 'bfloat16'")
    
    def _validate_config(self) -> None:
        """验证配置。"""
        # 验证数据类型
        valid_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtype not in valid_dtypes:
            raise ValueError(f"不支持的数据类型: {self.dtype}，支持的类型: {valid_dtypes}")
        
        # 验证设备
        if self.config.get("device") not in ["cpu", "cuda"]:
            raise ValueError("设备必须是 'cpu' 或 'cuda'")
    
    def _init_model(self) -> None:
        """初始化模型。"""
        try:
            # 设置数据类型
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16
            }
            torch_dtype = dtype_map[self.dtype]
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_path"],
                device_map=self.config.get("device", "cuda"),
                torch_dtype=torch_dtype,
                max_memory=self.max_memory
            )
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"])
            
            logger.info(f"模型加载完成，设备: {self.config.get('device', 'cuda')}, 数据类型: {self.dtype}")
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
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未初始化")
            
        try:
            # 编码输入
            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = inputs.to(self.config.get("device", "cuda"))
            
            # 生成输出
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or 512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # 解码输出
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return output_text
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def cleanup(self) -> None:
        """清理资源。"""
        try:
            if self.model:
                self.model.cpu()
                del self.model
                self.model = None
                
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("TinyLlama 模型清理完成")
        except Exception as e:
            logger.error(f"清理失败: {e}")
            raise 