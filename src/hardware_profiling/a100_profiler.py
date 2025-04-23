# hybrid-llm-inference/src/hardware_profiling/a100_profiler.py
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
import pynvml
import logging
import os

logger = logging.getLogger(__name__)

class A100Profiler(HardwareProfiler):
    """NVIDIA A100 GPU的硬件分析器"""
    
    def __init__(self, config):
        """
        初始化A100分析器
        
        Args:
            config (dict): 硬件配置
        """
        super().__init__(config)
        self.device_id = config.get("device_id", 0)
        self.idle_power = config.get("idle_power", 40.0)  # 默认空闲功耗40W
        self.sample_interval = config.get("sample_interval", 200)  # 毫秒
        self.is_test_mode = os.environ.get("TEST_MODE", "false").lower() == "true"
        self.handle = None
        
        if not self.is_test_mode:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                device_name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode()
                if "A100" not in device_name:
                    logger.warning(f"当前设备不是A100: {device_name}")
                    raise ValueError(f"当前设备不是A100: {device_name}")
            except Exception as e:
                logger.error(f"初始化NVML失败: {str(e)}")
                raise
        else:
            logger.info("在测试模式下运行A100分析器")
            
    def measure_power(self):
        """
        测量A100的当前功率
        
        Returns:
            float: 当前功率（瓦特）
        """
        if self.is_test_mode:
            # 测试模式下返回模拟功率
            return 100.0  # 模拟100W功率
            
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 转换为瓦特
            return max(0.0, power - self.idle_power)  # 减去空闲功耗
        except Exception as e:
            logger.error(f"功率测量失败: {str(e)}")
            return 0.0
            
    def __del__(self):
        """清理NVML资源"""
        try:
            if not self.is_test_mode and self.handle is not None:
                pynvml.nvmlShutdown()
        except:
            pass 