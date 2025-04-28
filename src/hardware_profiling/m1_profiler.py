"""Apple M1 处理器性能分析器。"""

import logging
import time
from typing import Dict, Any, Optional, Callable
import psutil

from src.hardware_profiling.base_profiler import HardwareProfiler
from src.toolbox.logger import get_logger

logger = get_logger(__name__)

class M1Profiler(HardwareProfiler):
    """Apple M1 处理器性能分析器。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 M1 性能分析器。

        Args:
            config: 配置字典，必须包含：
                - device_id: 设备 ID
                - idle_power: 空闲功耗（瓦特）
                - sample_interval: 采样间隔（毫秒）
        """
        super().__init__(config)
        self.device_id = config.get("device_id", 0)
        self.idle_power = config.get("idle_power", 15.0)  # 默认空闲功耗
        self.sample_interval = config.get("sample_interval", 200)  # 默认采样间隔
        self._validate_config()
        
    def _validate_config(self) -> None:
        """验证配置参数。"""
        if not self.config:
            raise ValueError("配置不能为空")
            
        if "idle_power" in self.config and not isinstance(self.config["idle_power"], (int, float)):
            raise ValueError("空闲功耗必须是数字")
            
        if "sample_interval" in self.config and not isinstance(self.config["sample_interval"], (int, float)):
            raise ValueError("采样间隔必须是数字")
    
    def measure_power(self) -> float:
        """测量当前功耗。

        Returns:
            当前功耗（瓦特）
        """
        # 由于无法直接测量 M1 功耗，返回预设值
        return self.idle_power
    
    def measure(self, task: Callable, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """测量任务的性能指标。

        Args:
            task: 要执行的任务
            input_tokens: 输入token数量
            output_tokens: 输出token数量

        Returns:
            Dict[str, float]: 性能指标字典
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 执行任务
            task()
            
            # 记录结束时间
            end_time = time.time()
            
            # 计算指标
            runtime = end_time - start_time
            total_tokens = input_tokens + output_tokens
            throughput = total_tokens / runtime if runtime > 0 else 0
            energy = self.idle_power * runtime  # 简化的能耗计算
            energy_per_token = energy / total_tokens if total_tokens > 0 else 0
            
            return {
                "energy": energy,
                "runtime": runtime,
                "throughput": throughput,
                "energy_per_token": energy_per_token
            }
            
        except Exception as e:
            self.logger.error(f"测量失败: {e}")
            raise
    
    def cleanup(self) -> None:
        """清理资源。"""
        pass

    def profile_cpu(self) -> Dict[str, float]:
        """测量CPU使用率。

        Returns:
            Dict[str, float]: CPU使用率指标
        """
        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
        }

    def profile_gpu(self) -> Dict[str, float]:
        """测量GPU使用率。

        Returns:
            Dict[str, float]: GPU使用率指标
        """
        # M1芯片的GPU使用率无法直接测量，返回默认值
        return {
            "gpu_usage": 0.0,
            "memory_usage": 0.0
        }

    def profile_memory(self) -> Dict[str, float]:
        """测量内存使用情况。

        Returns:
            Dict[str, float]: 内存使用指标
        """
        memory = psutil.virtual_memory()
        return {
            "total_memory": memory.total,
            "used_memory": memory.used,
            "free_memory": memory.free
        } 