"""混合LLM推理模块。"""

import os
import logging
from typing import Dict, Any, Optional
from .hardware_profiling import get_profiler
from .model_zoo.mock_model import MockModel
from .model_zoo.base_model import BaseModel
from .task_allocation import TaskAllocator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridInference:
    """混合LLM推理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化混合推理系统。

        Args:
            config: 配置字典，包含模型和硬件配置
        """
        self._validate_config(config)
        self.config = config
        self.test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
        
        # 在测试模式下使用MockModel
        if self.test_mode:
            self.model = MockModel(config["model_config"])
            logger.info("使用MockModel进行测试")
        else:
            raise NotImplementedError("非测试模式下需要指定具体的模型类")
            
        self.profiler = get_profiler(skip_nvml=self.test_mode)
        self.allocator = TaskAllocator(self.model, self.profiler)
        
    def infer(self, text: str) -> str:
        """
        执行推理。

        Args:
            text: 输入文本

        Returns:
            str: 推理结果
        """
        return self.inference(text)
        
    def cleanup(self):
        """清理资源"""
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        if hasattr(self.profiler, 'cleanup'):
            self.profiler.cleanup()
    
    def _validate_config(self, config: Dict[str, Any]):
        """
        验证配置参数
        
        Args:
            config: 配置字典，包含硬件配置和模型配置
            
        Raises:
            ValueError: 当配置参数无效时
        """
        if "hardware_config" not in config or "model_config" not in config:
            raise ValueError("配置必须包含hardware_config和model_config")
            
        hardware_config = config["hardware_config"]
        model_config = config["model_config"]
        
        # 验证硬件配置
        required_hw_keys = ['device_id']
        for key in required_hw_keys:
            if key not in hardware_config:
                raise ValueError(f"硬件配置缺少必要参数: {key}")
                
        # 验证模型配置
        required_model_keys = ['model_name', 'model_path', 'model_type', 'mode', 'batch_size', 'max_length']
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"模型配置缺少必要参数: {key}")
                
        # 验证mode的值
        if model_config["mode"] not in ["local", "remote"]:
            raise ValueError("mode必须是'local'或'remote'")
            
        # 验证batch_size和max_length的值
        if not isinstance(model_config["batch_size"], int) or model_config["batch_size"] <= 0:
            raise ValueError("batch_size必须是正整数")
            
        if not isinstance(model_config["max_length"], int) or model_config["max_length"] <= 0:
            raise ValueError("max_length必须是正整数")
        
    def inference(self, input_text: str) -> str:
        """
        执行推理
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 输出文本
        
        Raises:
            ValueError: 当输入文本为None或空字符串时
        """
        if input_text is None or not input_text.strip():
            raise ValueError("输入文本不能为空")
            
        # 测量性能
        power = self.profiler.measure_power()
        memory = self.profiler.get_memory_info()
        temperature = self.profiler.get_temperature()
        
        # 记录性能数据
        logger.info("Power: %.2fW, Memory: %d/%d bytes, Temperature: %.1f°C",
                        power, memory['used'], memory['total'], temperature)
        
        # 执行推理
        output = self.model.infer(input_text)
        
        return output
    
    def measure_performance(self, input_text: str) -> Dict[str, float]:
        """
        测量性能指标
        
        Args:
            input_text: 输入文本
            
        Returns:
            Dict[str, float]: 性能指标
            
        Raises:
            ValueError: 当输入文本为None时
        """
        if input_text is None:
            raise ValueError("输入文本不能为None")
            
        # 获取输入token数量
        input_tokens = self.model.get_token_count(input_text)
        
        # 执行推理并获取输出
        output_text = self.model.inference(input_text)
        output_tokens = self.model.get_token_count(output_text)
        
        # 获取性能指标
        def task():
            return output_text
            
        metrics = self.profiler.measure(task, input_tokens, output_tokens)
        
        # 添加额外的指标
        metrics["power"] = self.profiler.measure_power()
        metrics["temperature"] = self.profiler.get_temperature()
        memory_info = self.profiler.get_memory_info()
        metrics["memory_used"] = memory_info["used"]
        metrics["total_tokens"] = input_tokens + output_tokens
        
        return metrics 