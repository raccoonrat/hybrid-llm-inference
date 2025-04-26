"""RTX4050 NVML功能测试。"""

import os
import pytest
import pynvml
from unittest.mock import patch, MagicMock

from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.toolbox.logger import get_logger

logger = get_logger(__name__)

# 测试配置
TEST_CONFIG = {
    "device_id": 0,
    "device_type": "RTX4050",
    "idle_power": 20.0,
    "sample_interval": 0.1,
    "memory_limit": 6 * 1024 * 1024 * 1024,  # 6GB
    "tdp": 115.0,  # 115W
    "log_level": "DEBUG"
}

@pytest.fixture(scope="function")
def mock_nvml():
    """模拟NVML库的fixture。"""
    with patch("pynvml.nvmlInit") as mock_init, \
         patch("pynvml.nvmlDeviceGetHandleByIndex") as mock_get_handle, \
         patch("pynvml.nvmlDeviceGetName") as mock_get_name, \
         patch("pynvml.nvmlDeviceGetPowerUsage") as mock_get_power, \
         patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_get_memory, \
         patch("pynvml.nvmlDeviceGetUtilizationRates") as mock_get_util, \
         patch("pynvml.nvmlShutdown") as mock_shutdown:
        
        # 设置模拟返回值
        mock_handle = MagicMock()
        mock_get_handle.return_value = mock_handle
        mock_get_name.return_value = b"NVIDIA GeForce RTX 4050"
        mock_get_power.return_value = 50000  # 50W in milliwatts
        
        class MemoryInfo:
            def __init__(self):
                self.total = 6 * 1024 * 1024 * 1024  # 6GB
                self.used = 2 * 1024 * 1024 * 1024   # 2GB
                self.free = 4 * 1024 * 1024 * 1024   # 4GB
        mock_get_memory.return_value = MemoryInfo()
        
        class UtilizationRates:
            def __init__(self):
                self.gpu = 75
                self.memory = 50
        mock_get_util.return_value = UtilizationRates()
        
        yield {
            "init": mock_init,
            "get_handle": mock_get_handle,
            "get_name": mock_get_name,
            "get_power": mock_get_power,
            "get_memory": mock_get_memory,
            "get_util": mock_get_util,
            "shutdown": mock_shutdown,
            "handle": mock_handle
        }

def test_rtx4050_nvml():
    """测试 RTX4050 的 NVML 功能"""
    try:
        # 系统信息
        logger.info(f"操作系统: {platform.system()} {platform.release()}")
        logger.info(f"Python 版本: {sys.version}")

        # 检查 NVML 库路径
        nvml_path = r"C:\Windows\System32\nvml.dll"
        if not os.path.exists(nvml_path):
            logger.error(f"NVML 库不存在: {nvml_path}")
            return False
        
        logger.info(f"找到 NVML 库: {nvml_path}")

        # 直接加载 NVML DLL
        try:
            logger.info("正在加载 NVML DLL...")
            nvml = ctypes.CDLL(nvml_path)
            logger.info("成功加载 NVML DLL")

            # 初始化 NVML
            logger.info("正在初始化 NVML...")
            result = nvml.nvmlInit_v2()
            if result != 0:
                logger.error(f"NVML 初始化失败，错误代码: {result}")
                return False
            logger.info("NVML 初始化成功")

            # 获取设备数量
            device_count = c_uint(0)
            result = nvml.nvmlDeviceGetCount_v2(byref(device_count))
            if result != 0:
                logger.error(f"获取设备数量失败，错误代码: {result}")
                return False
            
            logger.info(f"发现 {device_count.value} 个 NVIDIA GPU 设备")

            # 查找 RTX 4050
            rtx4050_found = False
            for i in range(device_count.value):
                device = c_void_p(0)
                result = nvml.nvmlDeviceGetHandleByIndex_v2(c_uint(i), byref(device))
                if result != 0:
                    logger.error(f"获取设备 {i} 句柄失败，错误代码: {result}")
                    continue

                # 获取设备名称
                name = create_string_buffer(96)
                result = nvml.nvmlDeviceGetName(device, name, c_uint(96))
                if result == 0:
                    device_name = name.value.decode()
                    logger.info(f"设备 {i}: {device_name}")
                    if "RTX 4050" in device_name:
                        rtx4050_found = True
                        logger.info("找到 RTX 4050 设备")
                        
                        # 获取内存信息
                        memory = c_ulonglong * 3  # total, free, used
                        memory_info = memory()
                        result = nvml.nvmlDeviceGetMemoryInfo(device, byref(memory_info))
                        if result == 0:
                            total = memory_info[0] / (1024 * 1024)  # 转换为 MB
                            used = memory_info[2] / (1024 * 1024)
                            free = memory_info[1] / (1024 * 1024)
                            logger.info(f"  总显存: {total:.0f} MB")
                            logger.info(f"  已用显存: {used:.0f} MB")
                            logger.info(f"  可用显存: {free:.0f} MB")
                        else:
                            logger.error(f"获取显存信息失败，错误代码: {result}")

                        # 获取功耗信息
                        power = c_uint(0)
                        result = nvml.nvmlDeviceGetPowerUsage(device, byref(power))
                        if result == 0:
                            logger.info(f"  当前功耗: {power.value/1000.0:.2f} W")
                        else:
                            logger.error(f"获取功耗信息失败，错误代码: {result}")

                        # 获取利用率信息
                        class nvmlUtilization_t(Structure):
                            _fields_ = [("gpu", c_uint), ("memory", c_uint)]

                        utilization = nvmlUtilization_t()
                        result = nvml.nvmlDeviceGetUtilizationRates(device, byref(utilization))
                        if result == 0:
                            logger.info(f"  GPU 利用率: {utilization.gpu}%")
                            logger.info(f"  显存利用率: {utilization.memory}%")
                        else:
                            logger.error(f"获取利用率信息失败，错误代码: {result}")

            if not rtx4050_found:
                logger.error("未找到 RTX 4050 设备")
                return False

            # 关闭 NVML
            nvml.nvmlShutdown()
            logger.info("NVML 测试完成")
            return True

        except Exception as e:
            logger.error(f"NVML 测试失败: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        return False

if __name__ == "__main__":
    test_rtx4050_nvml() 