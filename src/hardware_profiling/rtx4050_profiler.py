"""RTX 4050 显卡性能分析器。"""

import os
import sys
import time
from typing import Dict, Any, Optional, Callable
from toolbox.logger import get_logger
from src.hardware_profiling.base_profiler import HardwareProfiler

logger = get_logger(__name__)

def _get_nvml_library_path() -> str:
    """获取 NVML 库的路径。
    
    Returns:
        str: NVML 库的完整路径
        
    Raises:
        RuntimeError: 当找不到 NVML 库时抛出
    """
    if sys.platform == "win32":
        # Windows 平台，固定路径为 c:\windows\system32\nvml.dll
        nvml_path = "c:\\windows\\system32\\nvml.dll"
    else:
        # Linux 平台，在 /usr/local/cuda-12.8 下查找
        cuda_base = "/usr/local/cuda-12.8"
        cuda_lib_paths = [
            os.path.join(cuda_base, "lib64"),
            os.path.join(cuda_base, "lib"),
            os.path.join(cuda_base, "targets/x86_64-linux/lib/stus"),
            "/usr/lib64",
            "/usr/lib"
        ]
        
        nvml_path = None
        for path in cuda_lib_paths:
            test_path = os.path.join(path, "libnvidia-ml.so")
            if os.path.exists(test_path):
                nvml_path = test_path
                break
        
        if not nvml_path:
            raise RuntimeError(f"无法在 {cuda_base} 及系统库目录下找到 NVML 库，请确保已安装 NVIDIA 驱动和 CUDA")
    
    if not os.path.exists(nvml_path):
        raise RuntimeError(f"NVML 库不存在: {nvml_path}")
    
    return nvml_path

# 测试模式下模拟 PyTorch 和 NVML
if os.getenv('TEST_MODE') == '1':
    class MockTorch:
        def __init__(self):
            self.cuda = MockCuda()
            self.device = self._device

        def _device(self, device_str):
            return MockDevice(device_str)

    class MockCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 1

    class MockDevice:
        def __init__(self, device_str):
            self.device_str = device_str

    class MockNVML:
        NVML_TEMPERATURE_GPU = 0  # 添加温度常量
        
        def nvmlInit(self):
            pass

        def nvmlShutdown(self):
            pass

        def nvmlDeviceGetHandleByIndex(self, index):
            return MockNVMLHandle()

        def nvmlDeviceGetName(self, handle):
            return b"RTX 4050"

        def nvmlDeviceGetPowerUsage(self, handle):
            return 50000  # 50W in mW

        def nvmlDeviceGetMemoryInfo(self, handle):
            class MockMemoryInfo:
                def __init__(self):
                    self.total = 2 * 1024 * 1024 * 1024  # 2GB
                    self.used = 1024 * 1024 * 1024  # 1GB
                    self.free = 1024 * 1024 * 1024  # 1GB
            return MockMemoryInfo()

        def nvmlDeviceGetUtilizationRates(self, handle):
            class MockUtilization:
                def __init__(self):
                    self.gpu = 50  # 50% utilization
                    self.memory = 50
            return MockUtilization()

        def nvmlDeviceGetTemperature(self, handle, temp_type):
            return 65  # 65°C

    class MockNVMLHandle:
        pass

    torch = MockTorch()
    pynvml = MockNVML()
else:
    import torch
    import pynvml

class RTX4050Profiler(HardwareProfiler):
    """RTX 4050 显卡性能分析器类。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化 RTX 4050 性能分析器。
        
        Args:
            config: 配置字典，包含以下字段：
                - device_id: 设备 ID
                - device_type: 设备类型
                - idle_power: 空闲功率
                - sample_interval: 采样间隔
        """
        # 调用父类初始化
        super().__init__(config)
        
        self.device_id = config.get("device_id", 0)
        self.device_type = config.get("device_type", "gpu")
        self.idle_power = config.get("idle_power", 15.0)
        self.sample_interval = config.get("sample_interval", 200)
        self.initialized = False
        self.device = None
        self.handle = None
        self.nvml_initialized = False
        self.is_measuring = False
        self.is_test_mode = os.getenv('TEST_MODE') == '1'
        
        # 验证配置
        self._validate_config()
        
        # 初始化分析器
        self._init_profiler()

    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.device_id, int):
            raise ValueError("device_id 必须是整数")
        if not isinstance(self.device_type, str):
            raise ValueError("device_type 必须是字符串")
        if not isinstance(self.idle_power, (int, float)) or self.idle_power <= 0:
            raise ValueError("idle_power 必须是正数")
        if not isinstance(self.sample_interval, int) or self.sample_interval <= 0:
            raise ValueError("sample_interval 必须是正整数")

    def _init_profiler(self) -> None:
        """初始化性能分析器。"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA 不可用，使用 CPU 模式")
                self.device = torch.device("cpu")
                self.initialized = True
                return
                
            if self.device_id >= torch.cuda.device_count():
                logger.warning(f"设备 ID {self.device_id} 无效，使用 CPU 模式")
                self.device = torch.device("cpu")
                self.initialized = True
                return
                
            # 初始化 NVML
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                
                # 验证设备类型
                device_name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode()
                if "RTX 4050" not in device_name and not self.is_test_mode:
                    logger.warning(f"当前设备不是 RTX 4050: {device_name}")
            except (pynvml.NVMLError_LibraryNotFound, FileNotFoundError) as e:
                logger.warning(f"NVML 库不可用: {str(e)}，使用 CPU 模式")
                self.nvml_initialized = False
                
            self.device = torch.device(f"cuda:{self.device_id}")
            self.initialized = True
            logger.info(f"RTX 4050 性能分析器初始化完成，设备 ID: {self.device_id}")
        except Exception as e:
            logger.error(f"RTX 4050 性能分析器初始化失败: {str(e)}")
            self.cleanup()
            raise

    def measure_power(self) -> float:
        """测量 GPU 的当前功率。
        
        Returns:
            float: 当前功率（瓦特）
        """
        if not self.initialized or not self.nvml_initialized or self.handle is None:
            return self.idle_power
            
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 转换为瓦特
            return max(0.0, power - self.idle_power)  # 减去空闲功耗
        except pynvml.NVMLError as e:
            logger.error(f"功率测量失败: {e}")
            return self.idle_power

    def measure(self, task: Callable, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """测量任务性能。

        Args:
            task: 要测量的任务
            input_tokens: 输入令牌数
            output_tokens: 输出令牌数

        Returns:
            包含以下字段的字典：
                - energy: 能耗 (J)
                - runtime: 运行时间 (s)
                - throughput: 吞吐量 (tokens/s)
                - energy_per_token: 每令牌能耗 (J/token)
        """
        if not self.initialized:
            raise RuntimeError("性能分析器未初始化")

        try:
            # 获取初始功率
            start_power = self.measure_power()
            
            # 执行任务
            start_time = time.time()
            task()
            end_time = time.time()
            
            # 获取结束功率
            end_power = self.measure_power()
            
            # 计算指标
            runtime = end_time - start_time
            avg_power = (start_power + end_power) / 2.0
            energy = avg_power * runtime
            total_tokens = input_tokens + output_tokens
            throughput = total_tokens / runtime if runtime > 0 else 0
            energy_per_token = energy / total_tokens if total_tokens > 0 else 0

            return {
                "energy": energy,
                "runtime": runtime,
                "throughput": throughput,
                "energy_per_token": energy_per_token
            }
        except Exception as e:
            logger.error(f"性能测量失败: {str(e)}")
            raise

    def cleanup(self) -> None:
        """清理资源。"""
        try:
            if self.nvml_initialized:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
            self.initialized = False
            self.handle = None
            self.device = None
            logger.info("RTX 4050 性能分析器清理完成")
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
            # 不抛出异常，确保资源被清理

    def __del__(self):
        """析构函数。"""
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构函数中的错误

    def get_power_usage(self) -> float:
        """获取当前功率使用情况。

        Returns:
            float: 当前功率使用值（瓦特）
        """
        return self.measure_power()

    def get_memory_usage(self) -> int:
        """获取当前显存使用情况。

        Returns:
            int: 当前显存使用量（字节）
        """
        if not self.initialized or not self.nvml_initialized or self.handle is None:
            return 0
            
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return info.used
        except pynvml.NVMLError as e:
            logger.error(f"显存使用量获取失败: {e}")
            return 0

    def get_gpu_utilization(self) -> float:
        """获取 GPU 利用率。

        Returns:
            float: GPU 利用率（百分比）
        """
        if not self.initialized or not self.nvml_initialized or self.handle is None:
            return 0.0
            
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return float(utilization.gpu)
        except pynvml.NVMLError as e:
            logger.error(f"GPU 利用率获取失败: {e}")
            return 0.0

    def start_monitoring(self) -> None:
        """开始性能监控。"""
        if not self.initialized:
            raise RuntimeError("性能分析器未初始化")
            
        self.is_measuring = True
        self.monitoring_start_time = time.time()
        self.monitoring_start_power = self.measure_power()
        self.monitoring_start_memory = self.get_memory_usage()
        self.monitoring_start_utilization = self.get_gpu_utilization()
        
    def stop_monitoring(self) -> Dict[str, float]:
        """停止性能监控。

        Returns:
            Dict[str, float]: 包含以下字段的性能指标：
                - energy_consumption: 能耗（J）
                - runtime: 运行时间（s）
                - avg_power: 平均功率（W）
                - avg_memory: 平均内存使用（MB）
                - avg_utilization: 平均 GPU 利用率（%）
        """
        if not hasattr(self, 'monitoring_start_time'):
            raise RuntimeError("未开始性能监控")
            
        self.is_measuring = False
        end_time = time.time()
        end_power = self.measure_power()
        end_memory = self.get_memory_usage()
        end_utilization = self.get_gpu_utilization()
        
        runtime = end_time - self.monitoring_start_time
        avg_power = (self.monitoring_start_power + end_power) / 2.0
        avg_memory = (self.monitoring_start_memory + end_memory) / 2.0
        avg_utilization = (self.monitoring_start_utilization + end_utilization) / 2.0
        energy_consumption = avg_power * runtime
        
        return {
            "energy_consumption": energy_consumption,
            "runtime": runtime,
            "avg_power": avg_power,
            "avg_memory": avg_memory,
            "avg_utilization": avg_utilization
        }

    def get_memory_info(self) -> Dict[str, float]:
        """获取 GPU 内存信息。

        Returns:
            Dict[str, float]: 包含以下字段的内存信息：
                - total: 总内存（字节）
                - used: 已使用内存（字节）
                - free: 可用内存（字节）
        """
        if not self.initialized or not self.nvml_initialized or self.handle is None:
            return {
                'total': 0.0,
                'used': 0.0,
                'free': 0.0
            }
            
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                'total': float(info.total),
                'used': float(info.used),
                'free': float(info.free)
            }
        except pynvml.NVMLError as e:
            logger.error(f"获取内存信息失败: {e}")
            return {
                'total': 0.0,
                'used': 0.0,
                'free': 0.0
            }

    def measure_performance(self, task: Dict[str, Any]) -> Dict[str, float]:
        """测量任务性能。

        Args:
            task: 任务字典，包含以下字段：
                - input: 输入文本
                - max_tokens: 最大输出令牌数

        Returns:
            Dict[str, float]: 包含以下字段的性能指标：
                - energy: 能耗（J）
                - runtime: 运行时间（s）
                - throughput: 吞吐量（tokens/s）
                - energy_per_token: 每令牌能耗（J/token）
        """
        if not self.initialized:
            raise RuntimeError("性能分析器未初始化")

        try:
            # 开始测量
            self.start_monitoring()
            
            # 执行任务
            def execute_task():
                # 这里应该执行实际的模型推理
                time.sleep(0.1)  # 模拟任务执行
            
            execute_task()
            
            # 停止测量并获取指标
            metrics = self.stop_monitoring()
            
            # 计算令牌相关的指标
            input_tokens = len(task["input"].split())  # 简单估算
            output_tokens = min(task["max_tokens"], 100)  # 假设平均输出100个令牌
            
            total_tokens = input_tokens + output_tokens
            metrics["throughput"] = total_tokens / metrics["runtime"] if metrics["runtime"] > 0 else 0
            metrics["energy_per_token"] = metrics["energy_consumption"] / total_tokens if total_tokens > 0 else 0
            
            return {
                "energy": metrics["energy_consumption"],
                "runtime": metrics["runtime"],
                "throughput": metrics["throughput"],
                "energy_per_token": metrics["energy_per_token"]
            }
        except Exception as e:
            logger.error(f"性能测量失败: {str(e)}")
            raise

    def profile_memory(self) -> Dict[str, float]:
        """分析内存使用情况。
        
        Returns:
            Dict[str, float]: 内存使用指标
        """
        if not self.initialized:
            return {
                "total": 0.0,
                "used": 0.0,
                "free": 0.0,
                "utilization": 0.0
            }
            
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            total = float(info.total)
            used = float(info.used)
            free = float(info.free)
            utilization = (used / total) * 100 if total > 0 else 0
            
            return {
                "total": total,
                "used": used,
                "free": free,
                "utilization": utilization
            }
        except pynvml.NVMLError as e:
            logger.error(f"获取内存信息失败: {e}")
            return {
                "total": 0.0,
                "used": 0.0,
                "free": 0.0,
                "utilization": 0.0
            }

    def profile_cpu(self) -> Dict[str, float]:
        """分析 CPU 使用情况。
        
        Returns:
            Dict[str, float]: CPU 使用指标
        """
        import psutil
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            
            return {
                "utilization": float(cpu_percent),
                "frequency": float(cpu_freq.current) if cpu_freq else 0.0,
                "core_count": float(cpu_count)
            }
        except Exception as e:
            logger.error(f"获取 CPU 信息失败: {e}")
            return {
                "utilization": 0.0,
                "frequency": 0.0,
                "core_count": 0.0
            }

    def profile_gpu(self) -> Dict[str, float]:
        """分析 GPU 使用情况。
        
        Returns:
            Dict[str, float]: GPU 使用指标
        """
        if not self.initialized or not self.nvml_initialized or self.handle is None:
            return {
                "power_usage": 0.0,
                "temperature": 0.0,
                "utilization": 0.0,
                "memory_utilization": 0.0
            }
            
        try:
            # 获取功率使用
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 转换为瓦特
            
            # 获取温度
            temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # 获取利用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = float(utilization.gpu)
            memory_util = float(utilization.memory)
            
            return {
                "power_usage": power,
                "temperature": float(temperature),
                "utilization": gpu_util,
                "memory_utilization": memory_util
            }
        except pynvml.NVMLError as e:
            logger.error(f"获取 GPU 信息失败: {e}")
            return {
                "power_usage": 0.0,
                "temperature": 0.0,
                "utilization": 0.0,
                "memory_utilization": 0.0
            } 