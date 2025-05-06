# hybrid-llm-inference/src/hardware_profiling/__init__.py
"""硬件性能分析模块。"""

from .base_profiler import HardwareProfiler
from .m1_profiler import M1Profiler
from .m1_pro_profiler import M1ProProfiler
from .rtx4050_profiler import RTX4050Profiler
from .a100_profiler import A100Profiler
from .a800_profiling import A800Profiler

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.info("Logger initialized for %s", __name__)

def get_profiler(device_name: str, config: Dict[str, Any]) -> HardwareProfiler:
    """
    根据设备名称和配置获取相应的性能分析器
    
    Args:
        device_name: 设备名称
        config: 设备配置
        
    Returns:
        HardwareProfiler: 性能分析器实例
    """
    # 兼容 nvidia_rtx4050
    if device_name == "nvidia_rtx4050":
        device_name = "rtx4050"
    if device_name == "m1_pro":
        return M1Profiler(config)
    elif device_name == "rtx4050":
        return RTX4050Profiler(config)
    elif device_name == "a100":
        return A100Profiler(config)
    elif device_name == "a800":
        return A800Profiler(config)
    else:
        raise ValueError(f"不支持的设备类型: {device_name}")
