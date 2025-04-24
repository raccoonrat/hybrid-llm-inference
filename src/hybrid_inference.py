"""混合LLM推理模块。"""

import os
import logging
from typing import Dict, Any, Optional
from .hardware_profiling import get_profiler
from .model_zoo.base_model import BaseModel

class HybridInference:
    """混合LLM推理类"""
    
    def __init__(self, hardware_config: Dict[str, Any], model_config: Dict[str, Any]):
        """
        初始化混合推理
        
        Args:
            hardware_config: 硬件配置
            model_config: 模型配置
            
        Raises:
            ValueError: 当配置参数无效时
        """
        # 验证配置
        self._validate_config(hardware_config, model_config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized for %s", __name__)
        
        # 检查是否在测试模式下
        self.is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
        
        # 初始化性能分析器
        self.profiler = get_profiler(
            device_id=hardware_config.get('device_id', 0),
            skip_nvml=self.is_test_mode or hardware_config.get('skip_nvml', False),
            config=hardware_config
        )
        
        # 初始化模型
        self.model = BaseModel(model_config)
        
    def _validate_config(self, hardware_config: Dict[str, Any], model_config: Dict[str, Any]):
        """
        验证配置参数
        
        Args:
            hardware_config: 硬件配置
            model_config: 模型配置
            
        Raises:
            ValueError: 当配置参数无效时
        """
        # 验证硬件配置
        required_hw_keys = ['device_id']
        for key in required_hw_keys:
            if key not in hardware_config:
                raise ValueError(f"硬件配置缺少必要参数: {key}")
                
        # 验证模型配置
        required_model_keys = ['model_name', 'model_path', 'model_type']
        for key in required_model_keys:
            if key not in model_config:
                raise ValueError(f"模型配置缺少必要参数: {key}")
        
    def inference(self, input_text: str) -> str:
        """
        执行推理
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 输出文本
        
        Raises:
            ValueError: 当输入文本为None时
        """
        if input_text is None:
            raise ValueError("输入文本不能为None")
            
        # 测量性能
        power = self.profiler.measure_power()
        memory = self.profiler.get_memory_info()
        temperature = self.profiler.get_temperature()
        
        # 记录性能数据
        self.logger.info("Power: %.2fW, Memory: %d/%d bytes, Temperature: %.1f°C",
                        power, memory['used'], memory['total'], temperature)
        
        # 执行推理
        output = self.model.inference(input_text)
        
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
            
        def task():
            return self.model.inference(input_text)
            
        input_tokens = len(input_text.split())
        output_tokens = len(task().split())
        
        return self.profiler.measure(task, input_tokens, output_tokens)
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'profiler') and self.profiler is not None:
            self.profiler.cleanup()
            self.profiler = None
        if hasattr(self, 'model'):
            self.model = None 