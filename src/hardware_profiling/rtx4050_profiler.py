"""RTX 4050 显卡性能分析器。"""

import os
import sys
import time
import torch
import pynvml
from typing import Dict, Any, Optional, Callable
from toolbox.logger import get_logger
from src.hardware_profiling.base_profiler import HardwareProfiler

logger = get_logger(__name__)

def _get_nvml_library_path() -> str:
    """获取 NVML 库的路径。
    
    Returns:
        str: NVML 库的完整路径
    """
    if sys.platform == "win32":
        # Windows 平台
        nvml_path = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "nvml.dll")
        if not os.path.exists(nvml_path):
            # 尝试 NVIDIA 驱动目录
            program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
            nvml_path = os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvml.dll")
    else:
        # Linux 平台
        cuda_paths = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib64",
            "/usr/lib"
        ]
        nvml_path = None
        for path in cuda_paths:
            test_path = os.path.join(path, "libnvidia-ml.so")
            if os.path.exists(test_path):
                nvml_path = test_path
                break
        
        if not nvml_path:
            raise RuntimeError("无法找到 NVML 库，请确保已安装 NVIDIA 驱动和 CUDA")
    
    if not os.path.exists(nvml_path):
        raise RuntimeError(f"NVML 库不存在: {nvml_path}")
    
    return nvml_path

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
        # 初始化基本属性
        self.device_id = config.get("device_id", 0)
        self.device_type = config.get("device_type", "gpu")
        self.idle_power = config.get("idle_power", 15.0)
        self.sample_interval = config.get("sample_interval", 200)
        self.initialized = False
        self.device = None
        self.handle = None
        self.nvml_initialized = False

        # 验证配置
        self._validate_config()
        
        # 初始化分析器
        self._init_profiler()
        
        # 调用父类初始化
        super().__init__(config)

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
                raise RuntimeError("CUDA 不可用")
            if self.device_id >= torch.cuda.device_count():
                raise ValueError(f"设备 ID {self.device_id} 无效")
                
            # 获取 NVML 库路径并初始化
            nvml_path = _get_nvml_library_path()
            pynvml.nvmlInit(nvml_path)
            self.nvml_initialized = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            # 验证设备类型
            device_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
            if "RTX 4050" not in device_name:
                logger.warning(f"当前设备不是 RTX 4050: {device_name}")
                
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