# hybrid-llm-inference/src/hardware_profiling/a800_profiling.py
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
import pynvml
import time
import psutil
from typing import Dict, Any
import logging

try:
    import pyjoules
except ImportError:
    pyjoules = None

class A800Profiler(HardwareProfiler):
    """A800 GPU性能分析器。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化A800性能分析器。
        
        Args:
            config: 配置字典
        """
        # 调用父类初始化
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        self.device_id = config.get("device_id", 0)
        self.idle_power = config.get("idle_power", 50.0)  # A800的默认空闲功率更高
        self.sample_interval = config.get("sample_interval", 200)  # 毫秒
        self.init_nvml()

    def init_nvml(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            device_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
            if "A800" not in device_name:
                self.logger.warning(f"Current device is not A800: {device_name}")
            self.logger.info(f"Initialized A800 profiler for device {self.device_id}")
        except pynvml.NVMLError as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            raise

    def measure_power(self):
        """
        测量A800的当前功率
        
        Returns:
            float: 当前功率（瓦特）
        """
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 转换为瓦特
            return max(0.0, power - self.idle_power)  # 减去空闲功耗
        except pynvml.NVMLError as e:
            self.logger.error(f"功率测量失败: {e}")
            return 0.0

    def __del__(self):
        """清理NVML资源"""
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def measure(self, task, input_tokens, output_tokens):
        """测量任务的能耗和运行时间。
        
        Args:
            task: 要执行的任务（如模型推理）
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            
        Returns:
            dict: 指标 {"energy": float, "runtime": float, "throughput": float, "energy_per_token": float}
        """
        try:
            if pyjoules is None:
                self.logger.warning("pyjoules未安装，使用模拟数据")
                # 使用模拟数据
                energy = self.idle_power * 0.5  # 假设使用了50%的空闲功率
                runtime = (input_tokens + output_tokens) * 0.01  # 每个token 10ms
                return {
                    "energy": energy,
                    "runtime": runtime,
                    "throughput": (input_tokens + output_tokens) / runtime,
                    "energy_per_token": energy / (input_tokens + output_tokens)
                }
                
            # 使用pyjoules测量
            energy_monitor = pyjoules.EnergyMonitor()
            energy_monitor.start()
            start_time = time.time()
            
            # 执行任务
            task()
            
            end_time = time.time()
            energy_monitor.stop()
            
            # 计算指标
            runtime = end_time - start_time
            energy = energy_monitor.get_energy()
            total_tokens = input_tokens + output_tokens
            
            return {
                "energy": energy,
                "runtime": runtime,
                "throughput": total_tokens / runtime,
                "energy_per_token": energy / total_tokens
            }
        except Exception as e:
            self.logger.error(f"测量失败: {e}")
            raise

    def profile_cpu(self) -> Dict[str, float]:
        """分析CPU使用情况。
        
        Returns:
            Dict[str, float]: CPU使用率和频率信息
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        return {
            "cpu_usage": cpu_percent,
            "cpu_freq": cpu_freq.current if cpu_freq else 0.0
        }
        
    def profile_gpu(self) -> Dict[str, float]:
        """分析GPU使用情况。
        
        Returns:
            Dict[str, float]: GPU使用率、温度和功耗信息
        """
        handle = self.get_device_handle()
        utilization = nvmlDeviceGetUtilizationRates(handle)
        temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
        
        return {
            "gpu_usage": utilization.gpu,
            "gpu_temp": float(temperature),
            "gpu_power": power
        }
        
    def profile_memory(self) -> Dict[str, float]:
        """分析内存使用情况。
        
        Returns:
            Dict[str, float]: 内存使用信息
        """
        handle = self.get_device_handle()
        memory = nvmlDeviceGetMemoryInfo(handle)
        
        return {
            "total_memory": memory.total / (1024**2),  # MB
            "used_memory": memory.used / (1024**2),
            "free_memory": memory.free / (1024**2)
        }
