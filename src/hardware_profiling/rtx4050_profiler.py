import os
import sys
import time
import ctypes
import logging
from ctypes import c_uint, c_ulong, c_ulonglong, c_int, c_char_p, byref, create_string_buffer, POINTER, c_void_p, Structure
from .base_profiler import HardwareProfiler
from toolbox.logger import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

class nvmlMemory_t(Structure):
    _fields_ = [
        ("total", c_ulonglong),
        ("free", c_ulonglong),
        ("used", c_ulonglong)
    ]

class RTX4050Profiler(HardwareProfiler):
    """RTX 4050 GPU的硬件分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RTX 4050分析器
        
        Args:
            config: 配置参数
                - device_id: GPU 设备 ID
                - idle_power: 空闲功耗（可选，默认 10W）
                - max_power: 最大功耗（可选，默认 115W）
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized for %s", __name__)
        
        # 设置默认配置
        self.config = {
            'device_id': 0,
            'idle_power': 10.0,  # 默认空闲功耗
            'max_power': 115.0,  # 默认最大功耗
            'sample_interval': 0.1,  # 采样间隔（秒）
        }
        
        # 更新用户配置
        self.config.update(config)
        
        # 初始化 NVML
        self.nvml = None
        self.device = None
        self.is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
        
        # 即使在测试模式下，也要验证设备ID的有效性
        if self.config['device_id'] > 8:  # 假设最多支持8个GPU
            raise ValueError(f"无效的设备ID: {self.config['device_id']}")
            
        self._init_nvml()
        
        self.handle = None
        
    def measure_power(self) -> float:
        """
        测量 GPU 的当前功率
        
        Returns:
            float: 当前功率（瓦特）
        """
        return self.get_power_usage()
        
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
        start_time = time.time()
        energy = 0.0
        power_samples = []
        
        try:
            # 执行任务并测量能耗
            result = task()
            end_time = time.time()
            runtime = end_time - start_time
            
            # 在测试模式下模拟能耗测量
            if self.is_test_mode:
                base_power = 100.0  # 基础功率（瓦特）
                power_variation = 20.0  # 功率变化范围（瓦特）
                num_samples = max(1, int(runtime / self.config['sample_interval']))
                
                for i in range(num_samples):
                    # 模拟功率波动
                    power = base_power + (i % 2) * power_variation
                    power_samples.append(power)
                    energy += power * self.config['sample_interval']
            else:
                # 实际测量功率
                last_measurement = start_time
                while time.time() - start_time < runtime + 0.1:  # 确保至少测量完整个任务时间
                    current_time = time.time()
                    if current_time - last_measurement >= self.config['sample_interval']:
                        power = self.measure_power()
                        power_samples.append(power)
                        energy += power * (current_time - last_measurement)
                        last_measurement = current_time
            
            total_tokens = max(1, input_tokens + output_tokens)  # 确保至少有一个token
            metrics = self._compute_metrics(energy, runtime, total_tokens)
            metrics['power_samples'] = power_samples
            metrics['result'] = result
            
            return metrics
            
        except Exception as e:
            logger.error(f"测量失败: {str(e)}")
            raise
        
    def _init_nvml(self):
        """初始化 NVML"""
        try:
            # 检查是否具有管理员权限
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            if not is_admin:
                logger.warning("当前没有管理员权限，某些功能可能无法使用")
            
            # 获取系统根目录
            system_root = os.environ.get('SystemRoot', 'C:\\Windows')
            nvml_path = os.path.join(system_root, 'System32', 'nvml.dll')
            
            logger.info(f"尝试加载 NVML 库: {nvml_path}")
            if not os.path.exists(nvml_path):
                raise FileNotFoundError(f"NVML 库未找到: {nvml_path}")
            
            # 加载 NVML 库
            try:
                # 设置 DLL 搜索路径
                os.environ['PATH'] = os.path.dirname(nvml_path) + os.pathsep + os.environ['PATH']
                logger.info(f"设置 DLL 搜索路径: {os.environ['PATH']}")
                
                # 尝试加载 DLL
                self.nvml = ctypes.WinDLL(nvml_path)
                logger.info("NVML 库加载成功")
                
                # 设置错误处理
                self.nvml.nvmlErrorString.restype = c_char_p
                logger.info("已设置错误处理函数")
                
                # 设置函数参数和返回类型
                self.nvml.nvmlDeviceGetHandleByIndex.argtypes = [c_uint, POINTER(c_void_p)]
                self.nvml.nvmlDeviceGetHandleByIndex.restype = c_int
                self.nvml.nvmlDeviceGetName.argtypes = [c_void_p, c_char_p, c_int]
                self.nvml.nvmlDeviceGetName.restype = c_int
                self.nvml.nvmlDeviceGetTemperature.argtypes = [c_void_p, c_int, POINTER(c_uint)]
                self.nvml.nvmlDeviceGetTemperature.restype = c_int
                self.nvml.nvmlDeviceGetPowerUsage.argtypes = [c_void_p, POINTER(c_uint)]
                self.nvml.nvmlDeviceGetPowerUsage.restype = c_int
                self.nvml.nvmlDeviceGetMemoryInfo.argtypes = [c_void_p, POINTER(nvmlMemory_t)]
                self.nvml.nvmlDeviceGetMemoryInfo.restype = c_int
                
            except Exception as e:
                logger.error(f"加载 NVML 库失败: {str(e)}")
                raise RuntimeError(f"加载 NVML 库失败: {str(e)}")
            
            # 初始化 NVML
            try:
                result = self.nvml.nvmlInit_v2()
                if result != 0:
                    error_str = self.nvml.nvmlErrorString(result)
                    if not is_admin:
                        logger.error(f"NVML 初始化失败，请以管理员权限运行程序: {error_str}")
                    raise RuntimeError(f"NVML 初始化失败: {result} ({error_str})")
                logger.info("NVML 初始化成功")
            except Exception as e:
                logger.error(f"NVML 初始化失败: {str(e)}")
                # 尝试使用旧版本 API
                try:
                    result = self.nvml.nvmlInit()
                    if result != 0:
                        error_str = self.nvml.nvmlErrorString(result)
                        raise RuntimeError(f"NVML 初始化失败 (旧版本 API): {result} ({error_str})")
                    logger.info("NVML 初始化成功 (使用旧版本 API)")
                except Exception as e2:
                    logger.error(f"NVML 初始化完全失败: {str(e2)}")
                    raise RuntimeError(f"NVML 初始化完全失败: {str(e2)}")
            
            # 获取设备数量
            try:
                device_count = c_uint()
                result = self.nvml.nvmlDeviceGetCount_v2(byref(device_count))
                if result != 0:
                    error_str = self.nvml.nvmlErrorString(result)
                    if not is_admin:
                        logger.error(f"获取设备数量失败，请以管理员权限运行程序: {error_str}")
                    raise RuntimeError(f"获取设备数量失败: {result} ({error_str})")
                logger.info(f"系统中有 {device_count.value} 个 NVIDIA GPU")
            except Exception as e:
                logger.error(f"获取设备数量失败: {str(e)}")
                raise RuntimeError(f"获取设备数量失败: {str(e)}")
            
            # 验证设备ID是否有效
            if self.config['device_id'] >= device_count.value:
                raise ValueError(f"设备ID {self.config['device_id']} 无效，系统中只有 {device_count.value} 个 NVIDIA GPU")
            
            # 获取设备句柄
            try:
                device_handle = c_void_p()
                result = self.nvml.nvmlDeviceGetHandleByIndex(self.config['device_id'], byref(device_handle))
                if result != 0:
                    error_str = self.nvml.nvmlErrorString(result)
                    if not is_admin:
                        logger.error(f"获取设备句柄失败，请以管理员权限运行程序: {error_str}")
                    raise RuntimeError(f"获取设备句柄失败: {result} ({error_str})")
                logger.info(f"成功获取设备句柄: {device_handle.value}")
            except Exception as e:
                logger.error(f"获取设备句柄失败: {str(e)}")
                raise RuntimeError(f"获取设备句柄失败: {str(e)}")
            
            # 获取设备名称
            try:
                name = create_string_buffer(64)
                result = self.nvml.nvmlDeviceGetName(device_handle, name, c_int(64))
                if result != 0:
                    error_str = self.nvml.nvmlErrorString(result)
                    if not is_admin:
                        logger.error(f"获取设备名称失败，请以管理员权限运行程序: {error_str}")
                    raise RuntimeError(f"获取设备名称失败: {result} ({error_str})")
                logger.info(f"成功获取设备名称: {name.value.decode()}")
            except Exception as e:
                logger.error(f"获取设备名称失败: {str(e)}")
                raise RuntimeError(f"获取设备名称失败: {str(e)}")
            
            self.device = device_handle
            self.device_name = name.value.decode()
            
            # 检查是否是RTX 4050
            if "RTX 4050" not in self.device_name:
                logger.warning(f"当前设备不是RTX 4050: {self.device_name}")
            
        except Exception as e:
            logger.error(f"NVML 初始化失败: {str(e)}")
            self.cleanup()
            raise
            
    def get_temperature(self) -> float:
        """获取 GPU 温度（摄氏度）"""
        if self.is_test_mode:
            return 50.0  # 测试模式下的模拟温度
            
        try:
            temperature = c_uint()
            result = self.nvml.nvmlDeviceGetTemperature(self.device, 0, byref(temperature))  # 0 表示 GPU 温度
            if result != 0:
                error_str = self.nvml.nvmlErrorString(result)
                raise RuntimeError(f"获取温度失败: {result} ({error_str})")
            return float(temperature.value)
        except Exception as e:
            logger.error(f"获取温度失败: {str(e)}")
            return 0.0
            
    def get_power_usage(self) -> float:
        """获取 GPU 功耗（瓦特）"""
        if self.is_test_mode:
            return 50.0  # 测试模式下的模拟功耗
            
        try:
            power = c_uint()
            result = self.nvml.nvmlDeviceGetPowerUsage(self.device, byref(power))
            if result != 0:
                error_str = self.nvml.nvmlErrorString(result)
                raise RuntimeError(f"获取功耗失败: {result} ({error_str})")
            return float(power.value) / 1000.0  # 转换为瓦特
        except Exception as e:
            logger.error(f"获取功耗失败: {str(e)}")
            return 0.0
            
    def get_memory_info(self) -> Dict[str, float]:
        """获取 GPU 内存信息（字节）"""
        if self.is_test_mode:
            return {
                'total': 6 * 1024 * 1024 * 1024,  # 6GB
                'used': 2 * 1024 * 1024 * 1024,   # 2GB
                'free': 4 * 1024 * 1024 * 1024    # 4GB
            }
            
        try:
            memory = nvmlMemory_t()
            result = self.nvml.nvmlDeviceGetMemoryInfo(self.device, byref(memory))
            if result != 0:
                error_str = self.nvml.nvmlErrorString(result)
                raise RuntimeError(f"获取内存信息失败: {result} ({error_str})")
            return {
                'total': float(memory.total),
                'used': float(memory.used),
                'free': float(memory.free)
            }
        except Exception as e:
            logger.error(f"获取内存信息失败: {str(e)}")
            return {
                'total': 0.0,
                'used': 0.0,
                'free': 0.0
            }
            
    def cleanup(self):
        """清理 NVML 资源"""
        try:
            if self.nvml is not None:
                self.nvml.nvmlShutdown()
                logger.info("NVML 已关闭")
        except Exception as e:
            logger.error(f"关闭 NVML 失败: {str(e)}")
            
    def __del__(self):
        """析构函数"""
        self.cleanup() 