import os
import sys
import logging
import ctypes
import platform
from ctypes import *

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_nvml():
    """测试 NVML 库的加载和使用"""
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

            # 获取系统驱动版本
            driver_version = create_string_buffer(80)
            result = nvml.nvmlSystemGetDriverVersion(driver_version, c_uint(80))
            if result == 0:
                logger.info(f"NVIDIA 驱动版本: {driver_version.value.decode()}")
            else:
                logger.error(f"获取驱动版本失败，错误代码: {result}")

            # 获取设备数量
            device_count = c_uint(0)
            result = nvml.nvmlDeviceGetCount_v2(byref(device_count))
            if result != 0:
                logger.error(f"获取设备数量失败，错误代码: {result}")
                return False
            
            logger.info(f"发现 {device_count.value} 个 NVIDIA GPU 设备")

            # 遍历每个设备
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
                    logger.info(f"设备 {i}: {name.value.decode()}")
                else:
                    logger.error(f"获取设备 {i} 名称失败，错误代码: {result}")

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
                    logger.error(f"获取设备 {i} 内存信息失败，错误代码: {result}")

                # 获取功耗信息
                power = c_uint(0)
                result = nvml.nvmlDeviceGetPowerUsage(device, byref(power))
                if result == 0:
                    logger.info(f"  当前功耗: {power.value/1000.0:.2f} W")
                else:
                    logger.error(f"获取设备 {i} 功耗失败，错误代码: {result}")

                # 获取利用率信息
                class nvmlUtilization_t(Structure):
                    _fields_ = [("gpu", c_uint), ("memory", c_uint)]

                utilization = nvmlUtilization_t()
                result = nvml.nvmlDeviceGetUtilizationRates(device, byref(utilization))
                if result == 0:
                    logger.info(f"  GPU 利用率: {utilization.gpu}%")
                    logger.info(f"  显存利用率: {utilization.memory}%")
                else:
                    logger.error(f"获取设备 {i} 利用率失败，错误代码: {result}")

            # 清理
            result = nvml.nvmlShutdown()
            if result != 0:
                logger.error(f"NVML 关闭失败，错误代码: {result}")
            else:
                logger.info("NVML 测试完成")

            return True

        except Exception as e:
            logger.error(f"加载或使用 NVML DLL 时出错: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"NVML 测试失败: {str(e)}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_nvml() 