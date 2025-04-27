"""混合推理模块。"""

import logging
from typing import Dict, Any, Optional
from src.model_zoo.base_model import BaseModel
from src.hardware_profiling import get_profiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridInference:
    """混合推理类。"""
    
    def __init__(self, model: BaseModel, hardware_config: Dict[str, Any], skip_nvml: bool = False):
        """
        初始化混合推理。

        Args:
            model: 基础模型
            hardware_config: 硬件配置，必须包含:
                - device_type: 设备类型 ('rtx4050' 或 'a100')
                - idle_power: 空闲功率（瓦特）
                - sample_interval: 采样间隔（毫秒）
            skip_nvml: 是否跳过 NVML 初始化（用于测试）
        """
        self.model = model
        self.hardware_config = hardware_config
        self.logger = logging.getLogger(__name__)
        self.profiler = get_profiler(config=hardware_config, skip_nvml=skip_nvml)
        
    def infer(self, text: str) -> str:
        """
        执行推理。

        Args:
            text: 输入文本

        Returns:
            str: 生成的文本
        """
        if not text:
            raise ValueError("输入文本不能为空")
            
        try:
            # 开始性能测量
            self.profiler.start_measurement()
            
            # 执行推理
            output = self.model.infer(text)
            
            # 结束性能测量
            self.profiler.end_measurement()
            
            return output
            
        except Exception as e:
            self.logger.error(f"推理失败: {str(e)}")
            raise
            
    def cleanup(self) -> None:
        """清理资源。"""
        if hasattr(self, "model"):
            self.model.cleanup()
        if hasattr(self, "profiler"):
            self.profiler.cleanup()
            
    def measure_performance(self, text: str) -> Dict[str, float]:
        """
        测量性能指标。
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, float]: 性能指标，包含:
                - energy: 能耗（焦耳）
                - runtime: 运行时间（秒）
                - throughput: 吞吐量（token/秒）
        """
        if not text:
            raise ValueError("输入文本不能为空")
            
        try:
            # 获取输入token数量
            input_tokens = self.model.get_token_count(text)
            
            # 执行推理并测量性能
            def inference_task():
                return self.model.infer(text)
                
            metrics = self.profiler.measure(
                inference_task,
                input_tokens,
                self.model.get_token_count(inference_task())
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"性能测量失败: {str(e)}")
            raise 