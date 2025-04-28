"""混合推理模块。"""

import logging
from typing import Dict, Any, Optional, Union
from src.model_zoo.base_model import BaseModel
from src.hardware_profiling import get_profiler
import time

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
        
    def infer(self, task: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行推理。

        Args:
            task: 输入任务，可以是字符串或字典
                如果是字典，必须包含:
                - input: 输入文本
                - max_tokens: 最大生成token数

        Returns:
            Dict[str, Any]: 包含输出文本和性能指标的字典
        """
        # 处理输入
        if isinstance(task, str):
            text = task
            max_tokens = None
        elif isinstance(task, dict):
            if "input" not in task:
                raise ValueError("任务字典必须包含 'input' 字段")
            if "max_tokens" not in task:
                raise ValueError("任务字典必须包含 'max_tokens' 字段")
            text = task["input"]
            max_tokens = task["max_tokens"]
        else:
            raise TypeError("任务必须是字符串或字典")
            
        if not text and not isinstance(text, str):
            raise ValueError("输入文本不能为空且必须是字符串")
            
        try:
            # 获取输入token数量
            input_tokens = self.model.get_token_count(text)
            
            # 定义推理任务
            def inference_task():
                return self.model.infer(text, max_tokens=max_tokens) if max_tokens else self.model.infer(text)
            
            # 执行推理并测量性能
            output = None
            def measure_task():
                nonlocal output
                output = inference_task()
                
            # 使用 profiler.measure 测量性能
            metrics = self.profiler.measure(
                measure_task,
                input_tokens=input_tokens,
                output_tokens=self.model.get_token_count(output) if output else 0
            )
            
            # 确保所有必需的指标都存在
            if "runtime" not in metrics:
                metrics["runtime"] = time.time() - start_time
            if "throughput" not in metrics:
                total_tokens = input_tokens + (self.model.get_token_count(output) if output else 0)
                metrics["throughput"] = total_tokens / metrics["runtime"] if metrics["runtime"] > 0 else 0
            
            return {
                "output": output,
                "metrics": metrics
            }
            
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
            output = None
            def inference_task():
                nonlocal output
                output = self.model.infer(text)
                
            metrics = self.profiler.measure(
                inference_task,
                input_tokens=input_tokens,
                output_tokens=self.model.get_token_count(output) if output else 0
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"性能测量失败: {str(e)}")
            raise 