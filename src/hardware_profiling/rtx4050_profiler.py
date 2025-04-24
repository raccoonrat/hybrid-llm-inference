"""RTX4050性能分析器模块。"""

import os
import sys
import time
import ctypes
import logging
from ctypes import c_uint, c_ulong, c_ulonglong, c_int, c_char_p, byref, create_string_buffer, POINTER, c_void_p, Structure, CDLL
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
from typing import Dict, Any, Optional, Callable
import pynvml

logger = get_logger(__name__)

class NVMLMemoryInfo(Structure):
    """NVML 内存信息结构体。"""
    _fields_ = [
        ("total", c_ulonglong),
        ("free", c_ulonglong),
        ("used", c_ulonglong)
    ]

# 设置NVML DLL搜索路径
nvml_dll_path = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32', 'nvml.dll')
logger.info(f"Setting NVML DLL path to: {nvml_dll_path}")

# 检查是否在测试模式下
is_test_mode = os.environ.get('TEST_MODE', '0') == '1'
if is_test_mode:
    logger.info("Test mode detected, skipping NVML initialization")
    nvml_lib = None
else:
    try:
        nvml_lib = CDLL(nvml_dll_path)
        logger.info("Successfully loaded NVML library")
        
        # 设置函数参数和返回类型
        nvml_lib.nvmlInit.restype = c_int
        nvml_lib.nvmlErrorString.restype = c_char_p
        nvml_lib.nvmlDeviceGetHandleByIndex.argtypes = [c_uint, POINTER(c_void_p)]
        nvml_lib.nvmlDeviceGetHandleByIndex.restype = c_int
        nvml_lib.nvmlDeviceGetName.argtypes = [c_void_p, c_char_p, c_int]
        nvml_lib.nvmlDeviceGetName.restype = c_int
        nvml_lib.nvmlDeviceGetPowerUsage.argtypes = [c_void_p, POINTER(c_uint)]
        nvml_lib.nvmlDeviceGetPowerUsage.restype = c_int
        nvml_lib.nvmlDeviceGetTemperature.argtypes = [c_void_p, c_int, POINTER(c_uint)]
        nvml_lib.nvmlDeviceGetTemperature.restype = c_int
        nvml_lib.nvmlDeviceGetMemoryInfo.argtypes = [c_void_p, POINTER(nvmlMemory_t)]
        nvml_lib.nvmlDeviceGetMemoryInfo.restype = c_int
        nvml_lib.nvmlShutdown.restype = c_int
        
        logger.info("Successfully set up NVML function types")
    except Exception as e:
        logger.warning(f"Failed to load NVML library: {e}")
        nvml_lib = None

class RTX4050Profiler(HardwareProfiler):
    """RTX 4050 GPU 性能分析器。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 RTX 4050 性能分析器。

        Args:
            config: 配置字典，必须包含：
                - device_type: 设备类型，必须为 "rtx4050"
                - idle_power: 空闲功耗（瓦特）
                - sample_interval: 采样间隔（毫秒）
        """
        super().__init__(config)
        self._validate_config()
        
        self.device_type = config["device_type"]
        self.idle_power = config.get("idle_power", 15.0)  # 默认空闲功耗
        self.sample_interval = config.get("sample_interval", 200)  # 默认采样间隔
        
        # 初始化 NVML
        if os.getenv("TEST_MODE") == "true":
            self.handle = None
            self.nvml = None
            logger.info("在测试模式下运行，跳过 NVML 初始化")
        else:
            self._init_nvml()
        
    def _validate_config(self) -> None:
        """验证配置。

        Args:
            config: 要验证的配置

        Raises:
            ValueError: 当配置无效时
        """
        if not self.config:
            raise ValueError("配置不能为空")
            
        if self.config.get("device_type") != "rtx4050":
            raise ValueError("设备类型必须为 rtx4050")
            
        if "idle_power" in self.config and not isinstance(self.config["idle_power"], (int, float)):
            raise ValueError("空闲功耗必须是数字")
            
        if "sample_interval" in self.config and not isinstance(self.config["sample_interval"], (int, float)):
            raise ValueError("采样间隔必须是数字")
    
    def _init_nvml(self) -> None:
        """初始化 NVML。"""
        try:
            # 设置 NVML DLL 路径
            nvml_path = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "nvml.dll")
            logger.info(f"Setting NVML DLL path to: {nvml_path}")
            
            # 加载 NVML 库
            self.nvml = ctypes.CDLL(nvml_path)
            logger.info("Successfully loaded NVML library")
            
            # 设置函数类型
            self.nvml.nvmlInit_v2.restype = c_uint
            self.nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [c_uint, POINTER(c_ulonglong)]
            self.nvml.nvmlDeviceGetHandleByIndex_v2.restype = c_uint
            self.nvml.nvmlDeviceGetPowerUsage.argtypes = [c_ulonglong, POINTER(c_uint)]
            self.nvml.nvmlDeviceGetPowerUsage.restype = c_uint
            self.nvml.nvmlDeviceGetMemoryInfo.argtypes = [c_ulonglong, POINTER(NVMLMemoryInfo)]
            self.nvml.nvmlDeviceGetMemoryInfo.restype = c_uint
            self.nvml.nvmlDeviceGetTemperature.argtypes = [c_ulonglong, c_uint, POINTER(c_uint)]
            self.nvml.nvmlDeviceGetTemperature.restype = c_uint
            logger.info("Successfully set up NVML function types")
            
            # 初始化 NVML
            result = self.nvml.nvmlInit_v2()
            if result != 0:
                raise RuntimeError(f"NVML 初始化失败: {result}")
            
            # 获取设备句柄
            self.handle = c_ulonglong()
            result = self.nvml.nvmlDeviceGetHandleByIndex_v2(0, byref(self.handle))
            if result != 0:
                raise RuntimeError(f"获取设备句柄失败: {result}")
                
        except Exception as e:
            logger.error(f"NVML 初始化失败: {str(e)}")
            self.handle = None
            self.nvml = None
    
    def measure_power(self) -> float:
        """测量当前功耗。

        Returns:
            当前功耗（瓦特）
        """
        if self.handle is None or self.nvml is None:
            return self.idle_power
            
        try:
            power = c_uint()
            result = self.nvml.nvmlDeviceGetPowerUsage(self.handle, byref(power))
            if result != 0:
                raise RuntimeError(f"功率测量失败: {result}")
            return power.value / 1000.0  # 转换为瓦特
        except Exception as e:
            logger.error(f"功率测量失败: {str(e)}")
            return self.idle_power
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息。

        Returns:
            包含内存信息的字典
        """
        if self.handle is None or self.nvml is None:
            return {"total": 0, "used": 0, "free": 0}
            
        try:
            memory = NVMLMemoryInfo()
            result = self.nvml.nvmlDeviceGetMemoryInfo(self.handle, byref(memory))
            if result != 0:
                raise RuntimeError(f"获取内存信息失败: {result}")
            return {
                "total": memory.total,
                "used": memory.used,
                "free": memory.free
            }
        except Exception as e:
            logger.error(f"获取内存信息失败: {str(e)}")
            return {"total": 0, "used": 0, "free": 0}
    
    def get_temperature(self) -> float:
        """获取温度。

        Returns:
            温度值（摄氏度）
        """
        if self.handle is None or self.nvml is None:
            return 0.0
            
        try:
            temp = c_uint()
            result = self.nvml.nvmlDeviceGetTemperature(self.handle, 0, byref(temp))
            if result != 0:
                raise RuntimeError(f"获取温度失败: {result}")
            return float(temp.value)
        except Exception as e:
            logger.error(f"获取温度失败: {str(e)}")
            return 0.0
    
    def start_measurement(self) -> None:
        """开始测量。"""
        if self.handle is None or self.nvml is None:
            self.start_time = time.time()
            self.start_power = self.idle_power
            return
            
        self.start_time = time.time()
        self.start_power = self.measure_power()
    
    def stop_measurement(self) -> Dict[str, float]:
        """停止测量。

        Returns:
            包含测量结果的字典
        """
        if self.handle is None or self.nvml is None:
            end_time = time.time()
            runtime = end_time - self.start_time
            return {
                "runtime": runtime,
                "power": self.idle_power,
                "energy": self.idle_power * runtime
            }
            
        end_time = time.time()
        end_power = self.measure_power()
        
        runtime = end_time - self.start_time
        avg_power = (self.start_power + end_power) / 2
        energy = avg_power * runtime
        
        return {
            "runtime": runtime,
            "power": avg_power,
            "energy": energy
        }
    
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
        try:
            self.start_measurement()
            
            # 执行任务
            input_tokens = task.get("input_tokens", 0)
            output_tokens = task.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            
            # 停止测量
            results = self.stop_measurement()
            
            # 计算指标
            throughput = total_tokens / results["runtime"] if results["runtime"] > 0 else 0
            energy_per_token = results["energy"] / total_tokens if total_tokens > 0 else 0
            
            return {
                "energy": results["energy"],
                "runtime": results["runtime"],
                "throughput": throughput,
                "energy_per_token": energy_per_token
            }
            
        except Exception as e:
            logger.error(f"测量失败: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """清理资源。"""
        if self.handle is not None and self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
                self.handle = None
                self.nvml = None
            except Exception as e:
                logger.error(f"清理失败: {str(e)}")
                raise

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构过程中的任何错误

    def start_measurement(self) -> None:
        """开始测量。"""
        if self.handle is None:
            return
            
        try:
            import pynvml
            pynvml.nvmlDeviceSetPowerManagementMode(self.handle, pynvml.NVML_POWER_MANAGEMENT_MODE_AUTO)
            logger.info("开始测量")
        except Exception as e:
            logger.error(f"开始测量失败: {str(e)}")

    def measure(self, task: Callable, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """执行测量。

        Args:
            task: 要测量的任务
            input_tokens: 输入token数
            output_tokens: 输出token数

        Returns:
            包含测量结果的字典
        """
        if self.handle is None:
            return {
                "energy": 0.0,
                "runtime": 0.0,
                "throughput": 0.0,
                "energy_per_token": 0.0
            }
            
        try:
            import pynvml
            start_time = time.time()
            start_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            
            task()
            
            end_time = time.time()
            end_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            
            runtime = end_time - start_time
            energy = (start_power + end_power) * runtime / 2.0
            total_tokens = input_tokens + output_tokens
            
            return {
                "energy": energy,
                "runtime": runtime,
                "throughput": total_tokens / runtime,
                "energy_per_token": energy / total_tokens
            }
        except Exception as e:
            logger.error(f"测量失败: {str(e)}")
            return {
                "energy": 0.0,
                "runtime": 0.0,
                "throughput": 0.0,
                "energy_per_token": 0.0
            } 