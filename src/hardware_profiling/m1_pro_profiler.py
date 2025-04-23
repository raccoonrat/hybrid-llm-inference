# hybrid-llm-inference/src/hardware_profiling/m1_pro_profiler.py
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
import psutil

class M1ProProfiler(HardwareProfiler):
    """M1 Pro的硬件分析器"""
    
    def __init__(self, config):
        """
        初始化M1 Pro分析器
        
        Args:
            config (dict): 硬件配置
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.idle_power = config.get("idle_power", 10.0)  # 默认空闲功耗10W
        self.sample_interval = config.get("sample_interval", 200)  # 毫秒
        
    def measure_power(self):
        """
        测量M1 Pro的当前功率（简化实现）
        
        Returns:
            float: 当前功率（瓦特）
        """
        try:
            # 使用CPU利用率作为功率估算的依据
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # 假设最大功耗为60W，根据CPU利用率线性估算
            power = (cpu_percent / 100.0) * 60.0
            return max(0.0, power - self.idle_power)
        except Exception as e:
            self.logger.error(f"功率测量失败: {str(e)}")
            return 0.0 