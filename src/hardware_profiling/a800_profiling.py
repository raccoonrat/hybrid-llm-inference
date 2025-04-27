# hybrid-llm-inference/src/hardware_profiling/a800_profiling.py
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
import pynvml
import time
import psutil
from typing import Dict

class A800Profiler(HardwareProfiler):
    def __init__(self, config):
        """
        Initialize profiler for NVIDIA A800 GPU.
        
        Args:
            config (dict): Hardware configuration (e.g., idle_power, sample_interval).
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        self.device_id = config.get("device_id", 0)
        self.sample_interval = config.get("sample_interval", 200)  # 保持毫秒单位
        self.idle_power = 50.0  # A800 的默认空闲功率，忽略配置中的值
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
        """
        Measure energy and runtime for a task on A800.
        
        Args:
            task (callable): Task to execute (e.g., model inference).
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
        
        Returns:
            dict: Metrics {"energy": float, "runtime": float, "throughput": float, "energy_per_token": float}.
        """
        try:
            # 记录开始时间
            start_time = time.time()
            start_power = self.measure_power()

            # 执行任务
            task()

            # 记录结束时间和功率
            end_time = time.time()
            end_power = self.measure_power()

            # 计算指标
            runtime = end_time - start_time
            avg_power = (start_power + end_power) / 2
            energy = avg_power * runtime
            total_tokens = input_tokens + output_tokens
            throughput = total_tokens / runtime if runtime > 0 else 0
            energy_per_token = energy / total_tokens if total_tokens > 0 else 0

            return {
                "energy": energy,
                "runtime": runtime,
                "throughput": throughput,
                "energy_per_token": energy_per_token,
                "avg_power": avg_power
            }

        except Exception as e:
            self.logger.error(f"测量失败: {e}")
            raise

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
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return {
                "gpu_usage": utilization.gpu,
                "memory_usage": utilization.memory
            }
        except pynvml.NVMLError as e:
            self.logger.error(f"GPU使用率测量失败: {e}")
            return {"gpu_usage": 0.0, "memory_usage": 0.0}

    def profile_memory(self) -> Dict[str, float]:
        """测量内存使用情况。

        Returns:
            Dict[str, float]: 内存使用指标
        """
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "total_memory": memory_info.total,
                "used_memory": memory_info.used,
                "free_memory": memory_info.free
            }
        except pynvml.NVMLError as e:
            self.logger.error(f"内存使用测量失败: {e}")
            return {"total_memory": 0.0, "used_memory": 0.0, "free_memory": 0.0}
