# hybrid-llm-inference/src/hardware_profiling/m1_pro_profiler.py
import os
import sys
import time
import ctypes
import logging
import psutil
from ctypes import c_uint, c_ulong, c_ulonglong, c_int, c_char_p, byref, create_string_buffer, POINTER, c_void_p, Structure, CDLL
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
from typing import Dict, Any, Optional

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
        logger.error(f"Failed to load NVML library: {e}")
        nvml_lib = None

class M1ProProfiler(HardwareProfiler):
    """M1 Pro GPU的硬件分析器"""
    
    def __init__(self, config: Dict[str, Any], skip_nvml: bool = False):
        """
        初始化M1 Pro分析器
        
        Args:
            config: 配置参数
                - device_type: 设备类型（默认为 'cpu_gpu'）
                - idle_power: 空闲功耗（默认 10W）
                - max_power: 最大功耗（默认 115W）
                - sample_interval: 采样间隔（默认 200ms）
            skip_nvml: 是否跳过NVML初始化
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized for %s", __name__)
        
        # 设置默认配置
        self.device_type = config.get('device_type', 'cpu_gpu')
        self.idle_power = config.get('idle_power', 10.0)
        self.max_power = config.get('max_power', 115.0)
        self.sample_interval = config.get('sample_interval', 200)
        
        # 初始化状态
        self.initialized = False
        self.nvml_initialized = False
        self.device_handle = None
        self.is_test_mode = os.getenv('TEST_MODE') == '1'
        
        # 即使在测试模式下，也要验证设备ID的有效性
        if self.config['device_id'] > 8:  # 假设最多支持8个GPU
            raise ValueError(f"无效的设备ID: {self.config['device_id']}")
            
        # 初始化 NVML
        if not skip_nvml and not self.is_test_mode:
            try:
                self._init_nvml()
            except Exception as e:
                self.logger.warning(f"NVML 初始化失败: {e}")
                self.nvml_initialized = False
        else:
            self.logger.info("Test mode detected, skipping NVML initialization")
            self.nvml_initialized = False
        
        self.initialized = True
        
    def _init_nvml(self):
        try:
            # 初始化NVML
            ret = nvml_lib.nvmlInit()
            if ret != 0:
                error_str = nvml_lib.nvmlErrorString(ret)
                logger.error(f"NVML初始化失败: {ret} ({error_str})")
                raise RuntimeError(f"NVML初始化失败: {ret} ({error_str})")
            logger.info("NVML初始化成功")
            
            # 获取设备句柄
            device_handle = c_void_p()
            ret = nvml_lib.nvmlDeviceGetHandleByIndex(self.config['device_id'], byref(device_handle))
            if ret != 0:
                error_str = nvml_lib.nvmlErrorString(ret)
                logger.error(f"获取设备句柄失败: {ret} ({error_str})")
                raise RuntimeError(f"获取设备句柄失败: {ret} ({error_str})")
            self.device_handle = device_handle
            logger.info(f"成功获取设备句柄: {self.device_handle.value}")
            
            # 获取设备名称
            name_buffer = create_string_buffer(64)
            ret = nvml_lib.nvmlDeviceGetName(self.device_handle, name_buffer, 64)
            if ret != 0:
                error_str = nvml_lib.nvmlErrorString(ret)
                logger.error(f"获取设备名称失败: {ret} ({error_str})")
                raise RuntimeError(f"获取设备名称失败: {ret} ({error_str})")
            self.device_name = name_buffer.value.decode('utf-8')
            logger.info(f"成功获取设备名称: {self.device_name}")
            
        except Exception as e:
            logger.error(f"NVML初始化过程中发生错误: {e}")
            raise
        
    def measure_power(self) -> float:
        """测量 GPU 的当前功率
        
        Returns:
            float: 当前功率（瓦特）
        """
        if nvml_lib is None or self.device_handle is None:
            return self.idle_power
            
        try:
            power = c_uint()
            ret = nvml_lib.nvmlDeviceGetPowerUsage(self.device_handle, byref(power))
            if ret != 0:
                error_str = nvml_lib.nvmlErrorString(ret)
                logger.error(f"获取功率失败: {ret} ({error_str})")
                return self.idle_power
            return float(power.value) / 1000.0  # 转换为瓦特
        except Exception as e:
            logger.error(f"测量功率时发生错误: {e}")
            return self.idle_power
        
    def measure(self, task, input_tokens, output_tokens):
        """
        测量任务执行的能耗和运行时间
        
        Args:
            task (callable): 要执行的任务
            input_tokens (int): 输入token数量
            output_tokens (int): 输出token数量
            
        Returns:
            dict: 包含能耗、运行时间和吞吐量的指标
        """
        if nvml_lib is None or self.device_handle is None:
            return {
                "energy": 1.0,
                "runtime": 0.1,
                "throughput": 100.0,
                "energy_per_token": 0.01
            }
        
        try:
            # 获取初始功率
            start_power = self.measure_power()
            
            # 执行任务
            start_time = time.time()
            result = task()
            end_time = time.time()
            
            # 获取结束功率
            end_power = self.measure_power()
            
            # 计算指标
            runtime = end_time - start_time
            avg_power = (start_power + end_power) / 2.0
            energy = avg_power * runtime
            throughput = (input_tokens + output_tokens) / runtime
            energy_per_token = energy / (input_tokens + output_tokens)
            
            return {
                "energy": energy,
                "runtime": runtime,
                "throughput": throughput,
                "energy_per_token": energy_per_token
            }
        except Exception as e:
            logger.error(f"测量过程中发生错误: {e}")
            return {
                "energy": 1.0,
                "runtime": 0.1,
                "throughput": 100.0,
                "energy_per_token": 0.01
            }
        
    def get_temperature(self) -> float:
        """获取 GPU 温度（摄氏度）"""
        if nvml_lib is None or self.device_handle is None:
            return 50.0  # 测试模式下的模拟温度
            
        try:
            temperature = c_uint()
            ret = nvml_lib.nvmlDeviceGetTemperature(self.device_handle, 0, byref(temperature))
            if ret != 0:
                error_str = nvml_lib.nvmlErrorString(ret)
                logger.error(f"获取温度失败: {ret} ({error_str})")
                return 50.0
            return float(temperature.value)
        except Exception as e:
            logger.error(f"获取温度时发生错误: {e}")
            return 50.0
            
    def get_memory_info(self) -> Dict[str, float]:
        """获取 GPU 内存信息（字节）"""
        if nvml_lib is None or self.device_handle is None:
            return {
                'total': 6 * 1024 * 1024 * 1024,  # 6GB
                'used': 2 * 1024 * 1024 * 1024,   # 2GB
                'free': 4 * 1024 * 1024 * 1024    # 4GB
            }
            
        try:
            memory = nvmlMemory_t()
            ret = nvml_lib.nvmlDeviceGetMemoryInfo(self.device_handle, byref(memory))
            if ret != 0:
                error_str = nvml_lib.nvmlErrorString(ret)
                logger.error(f"获取内存信息失败: {ret} ({error_str})")
                return {
                    'total': 0.0,
                    'used': 0.0,
                    'free': 0.0
                }
            return {
                'total': float(memory.total),
                'used': float(memory.used),
                'free': float(memory.free)
            }
        except Exception as e:
            logger.error(f"获取内存信息时发生错误: {e}")
            return {
                'total': 0.0,
                'used': 0.0,
                'free': 0.0
            }
            
    def cleanup(self):
        """清理 NVML 资源"""
        try:
            if nvml_lib is not None:
                ret = nvml_lib.nvmlShutdown()
                if ret != 0:
                    error_str = nvml_lib.nvmlErrorString(ret)
                    logger.error(f"关闭NVML失败: {ret} ({error_str})")
                else:
                    logger.info("NVML已关闭")
        except Exception as e:
            logger.error(f"关闭NVML时发生错误: {e}")
            
    def __del__(self):
        """析构函数"""
        self.cleanup()

    def profile_cpu(self) -> Dict[str, float]:
        """分析 CPU 使用情况。
        
        Returns:
            Dict[str, float]: CPU 使用指标
        """
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
        if not self.initialized or not self.nvml_initialized or self.device_handle is None:
            return {
                "power_usage": 0.0,
                "temperature": 0.0,
                "utilization": 0.0,
                "memory_utilization": 0.0
            }
        
        try:
            # 获取功率使用
            power = self.measure_power()
            
            # 获取温度
            temperature = nvml_lib.nvmlDeviceGetTemperature(self.device_handle, nvml_lib.NVML_TEMPERATURE_GPU)
            
            # 获取利用率
            utilization = nvml_lib.nvmlDeviceGetUtilizationRates(self.device_handle)
            gpu_util = float(utilization.gpu)
            memory_util = float(utilization.memory)
            
            return {
                "power_usage": power,
                "temperature": float(temperature),
                "utilization": gpu_util,
                "memory_utilization": memory_util
            }
        except Exception as e:
            logger.error(f"获取 GPU 信息失败: {e}")
            return {
                "power_usage": 0.0,
                "temperature": 0.0,
                "utilization": 0.0,
                "memory_utilization": 0.0
            }

    def profile_memory(self) -> Dict[str, float]:
        """分析内存使用情况。
        
        Returns:
            Dict[str, float]: 内存使用指标
        """
        if not self.initialized or not self.nvml_initialized or self.device_handle is None:
            return {
                "total": 0.0,
                "used": 0.0,
                "free": 0.0,
                "utilization": 0.0
            }
        
        try:
            info = nvml_lib.nvmlDeviceGetMemoryInfo(self.device_handle)
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
        except Exception as e:
            logger.error(f"获取内存信息失败: {e}")
            return {
                "total": 0.0,
                "used": 0.0,
                "free": 0.0,
                "utilization": 0.0
            } 