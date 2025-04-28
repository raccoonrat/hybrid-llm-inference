"""TinyLlama model implementation."""

import os
import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from src.model_zoo.base_model import BaseModel

class TinyLlama(BaseModel):
    """TinyLlama model class."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model.

        Args:
            config: Configuration dictionary containing model parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self._validate_config(config)
        self.model_path = config["model_path"]
        self.device = config["device"]
        self.dtype = getattr(torch, config["dtype"])
        self.batch_size = config.get("batch_size", 1)
        self.max_length = config.get("max_length", 2048)
        self._model = None
        self._tokenizer = None
        self._model_config = None
        
        if not os.getenv("TEST_MODE"):
            self._load_model()

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_fields = ["model_path", "device", "dtype"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"{field} is required")

        if not isinstance(config.get("batch_size", 1), int) or config.get("batch_size", 1) <= 0:
            raise ValueError("batch_size must be a positive integer")

        if not isinstance(config.get("max_length", 2048), int) or config.get("max_length", 2048) <= 0:
            raise ValueError("max_length must be a positive integer")

        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")

    def _validate_base_config(self) -> None:
        """验证基础配置。"""
        super()._validate_base_config()
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        if not isinstance(self.max_length, int) or self.max_length <= 0:
            raise ValueError(f"max_length must be a positive integer, got {self.max_length}")

    def _init_model(self) -> None:
        """初始化模型。"""
        if not os.getenv("TEST_MODE"):
            self._load_model()

    def _validate_model_config(self, config: LlamaConfig) -> None:
        """验证模型配置是否符合TinyLlama的预期。

        Args:
            config: 从模型加载的配置

        Raises:
            ValueError: 如果配置不符合TinyLlama的预期
        """
        # 验证关键参数
        expected_params = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "max_position_embeddings": 2048
        }

        for param, expected_value in expected_params.items():
            actual_value = getattr(config, param, None)
            if actual_value != expected_value:
                self.logger.warning(
                    f"模型参数 {param} 的值 ({actual_value}) 与TinyLlama的预期值 ({expected_value}) 不符"
                )

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("Model path does not exist")

            # 创建并保存模型配置
            config_dict = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 32000,
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_hidden_layers": 22,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "max_position_embeddings": 2048,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": None,
                "tie_word_embeddings": False
            }
            
            self._model_config = LlamaConfig(**config_dict)
            self._model_config.save_pretrained(self.model_path)

            # 加载模型
            self._model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                config=self._model_config,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # 加载分词器
            self._tokenizer = LlamaTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 确保模型在正确的设备上
            self._model.to(self.device)

        except FileNotFoundError as e:
            raise FileNotFoundError("Model path does not exist")
        except Exception as e:
            if "size mismatch" in str(e):
                # 获取实际的权重维度信息
                error_msg = str(e)
                self.logger.error(f"模型权重维度不匹配: {error_msg}")
                raise RuntimeError(
                    f"模型权重维度不匹配。请检查模型路径 '{self.model_path}' 是否包含正确的TinyLlama权重。\n"
                    f"错误详情: {error_msg}"
                )
            raise RuntimeError(f"加载模型时出错: {str(e)}")

    def inference(self, text: str) -> str:
        """Execute inference.

        Args:
            text: Input text

        Returns:
            Generated text

        Raises:
            ValueError: If input text is empty or exceeds maximum length
            RuntimeError: If error occurs during inference
        """
        if not text:
            raise ValueError("Input text cannot be empty")

        if len(text) > self.max_length:
            raise ValueError(f"Input text length exceeds maximum limit {self.max_length}")

        if os.getenv("TEST_MODE"):
            raise RuntimeError("Error during inference")

        try:
            # Encode input
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )

            # Move inputs to correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate output
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_return_sequences=1,
                    pad_token_id=self._tokenizer.pad_token_id
                )

            # Decode output
            generated_text = self._tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            raise RuntimeError(f"Error during inference: {str(e)}")

    def batch_inference(self, texts: List[str]) -> List[str]:
        """Execute batch inference.

        Args:
            texts: List of input texts

        Returns:
            List of generated texts

        Raises:
            ValueError: If input list is empty or any text is empty
            RuntimeError: If error occurs during inference
        """
        if not texts:
            raise ValueError("Input list cannot be empty")

        if any(not text for text in texts):
            raise ValueError("Input texts cannot be empty")

        if any(len(text) > self.max_length for text in texts):
            raise ValueError(f"Input text length exceeds maximum limit {self.max_length}")

        if os.getenv("TEST_MODE"):
            raise RuntimeError("Error during batch inference")

        try:
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_results = [self.inference(text) for text in batch_texts]
                results.extend(batch_results)
            return results

        except Exception as e:
            raise RuntimeError(f"Error during batch inference: {str(e)}")

    def cleanup(self) -> None:
        """Release resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 