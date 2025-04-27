import logging
from typing import Dict, Any, Optional

from src.hardware_profiling.base_profiler import HardwareProfiler

logger = logging.getLogger(__name__)

class M1Profiler(HardwareProfiler):
    """Apple M1 处理器性能分析器。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 M1 性能分析器。

        Args:
            config: 配置字典，必须包含：
                - device_type: 设备类型，必须为 "m1"
                - idle_power: 空闲功耗（瓦特）
                - sample_interval: 采样间隔（毫秒）
        """
        super().__init__(config)
        self._validate_config()
        
        self.device_type = config["device_type"]
        self.idle_power = config.get("idle_power", 15.0)  # 默认空闲功耗
        self.sample_interval = config.get("sample_interval", 200)  # 默认采样间隔
        
    def _validate_config(self) -> None:
        """验证配置参数。"""
        if not self.config:
            raise ValueError("配置不能为空")
            
        if self.config.get("device_type") != "m1":
            raise ValueError("设备类型必须为 m1")
            
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
    
    def measure(self, task: Dict[str, Any]) -> Dict[str, float]:
        """测量任务执行的性能指标。

        Args:
            task: 任务信息字典

        Returns:
            性能指标字典，包含：
            - energy: 能耗（焦耳）
            - runtime: 运行时间（秒）
            - throughput: 吞吐量（token/秒）
            - energy_per_token: 每 token 能耗（焦耳/token）
        """
        # 模拟测量结果
        runtime = 1.0  # 假设运行时间为 1 秒
        energy = self.idle_power * runtime
        
        input_tokens = task.get("input_tokens", 0)
        output_tokens = task.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        
        return self._compute_metrics(energy, runtime, total_tokens)
    
    def cleanup(self) -> None:
        """清理资源。"""
        pass 