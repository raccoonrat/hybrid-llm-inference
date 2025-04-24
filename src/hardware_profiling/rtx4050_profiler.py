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

class nvmlMemory_t(Structure):
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
    """RTX4050 性能分析器类。"""
    
    def __init__(self, config: Dict[str, Any], skip_nvml: bool = False):
        """
        初始化 RTX4050 性能分析器。
        
        Args:
            config: 硬件配置，必须包含:
                - device_type: 设备类型 ('rtx4050')
                - idle_power: 空闲功率（瓦特）
                - sample_interval: 采样间隔（毫秒）
            skip_nvml: 是否跳过 NVML 初始化（用于测试）
        """
        super().__init__(config)
        self.device_type = config.get("device_type", "rtx4050")
        self.idle_power = config.get("idle_power", 30.0)
        self.sample_interval = config.get("sample_interval", 200)
        self.is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
        self.skip_nvml = skip_nvml or self.is_test_mode
        self.nvml_initialized = False
        self.start_time = None
        self.start_power = None
        
        if not self.skip_nvml:
            try:
                # 设置 NVML DLL 路径
                if os.path.exists(nvml_dll_path):
                    os.environ['PATH'] = nvml_dll_path + os.pathsep + os.environ.get('PATH', '')
                
                # 初始化 NVML
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.info("NVML 初始化成功")
            except Exception as e:
                logger.warning(f"NVML 初始化失败: {str(e)}，将使用模拟数据")
                self.skip_nvml = True
                
    def start_measurement(self) -> None:
        """开始性能测量。"""
        self.start_time = time.time()
        self.start_power = self.measure_power()
        
    def end_measurement(self) -> None:
        """结束性能测量。"""
        self.start_time = None
        self.start_power = None
                
    def measure_power(self) -> float:
        """测量当前功率。"""
        if self.skip_nvml:
            return self.idle_power
            
        try:
            if self.nvml_initialized:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 转换为瓦特
                return max(power, self.idle_power)
        except Exception as e:
            logger.warning(f"获取功率失败: {str(e)}，使用空闲功率")
            
        return self.idle_power
            
    def get_memory_info(self) -> Dict[str, int]:
        """获取内存信息。"""
        if self.skip_nvml:
            return {"used": 1000, "total": 8000}
            
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "used": info.used,
                "total": info.total
            }
        except Exception as e:
            logger.warning(f"内存信息获取失败: {str(e)}，使用模拟数据")
            return {"used": 1000, "total": 8000}
            
    def get_temperature(self) -> float:
        """获取温度。"""
        if self.skip_nvml:
            return 50.0
            
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except Exception as e:
            logger.warning(f"温度获取失败: {str(e)}，使用模拟数据")
            return 50.0
            
    def measure(self, task: Callable, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """测量任务执行的性能指标。"""
        if self.skip_nvml:
            # 在测试模式下返回模拟数据
            return {
                "energy": 10.0,
                "runtime": 2.0,
                "throughput": (input_tokens + output_tokens) / 2.0,
                "energy_per_token": 0.1
            }
            
        # 记录开始时间和功率
        self.start_measurement()
        
        # 执行任务
        result = task()
        
        # 记录结束时间和功率
        end_time = time.time()
        end_power = self.measure_power()
        
        # 计算性能指标
        runtime = end_time - self.start_time
        avg_power = (self.start_power + end_power) / 2
        energy = avg_power * runtime
        total_tokens = input_tokens + output_tokens
        throughput = total_tokens / runtime if runtime > 0 else 0
        energy_per_token = energy / total_tokens if total_tokens > 0 else 0
        
        # 结束测量
        self.end_measurement()
        
        return {
            "energy": energy,
            "runtime": runtime,
            "throughput": throughput,
            "energy_per_token": energy_per_token
        }
        
    def cleanup(self) -> None:
        """清理资源。"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
                logger.info("NVML 清理完成")
            except Exception as e:
                logger.warning(f"NVML 关闭失败: {str(e)}")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构过程中的任何错误 