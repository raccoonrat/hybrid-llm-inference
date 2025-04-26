# hybrid-llm-inference/src/hardware_profiling/__init__.py
"""硬件性能分析模块。"""

from .base_profiler import HardwareProfiler
from .rtx4050_profiler import RTX4050Profiler
from .m1_profiler import M1Profiler
from .a800_profiling import A800Profiler

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.info("Logger initialized for %s", __name__)

def get_profiler(device_type: str, config: Dict[str, Any]) -> HardwareProfiler:
    """获取性能分析器实例。
    
    Args:
        device_type: 设备类型，支持 "rtx4050"、"a800" 和 "m1_pro"
        config: 配置字典，必须包含设备特定的配置参数
        
    Returns:
        性能分析器实例
        
    Raises:
        ValueError: 当设备类型不支持时
    """
    if device_type == "rtx4050":
        return RTX4050Profiler(config)
    elif device_type == "a800":
        return A800Profiler(config)
    elif device_type == "m1_pro":
        return M1Profiler(config)
    else:
        raise ValueError(f"不支持的设备类型: {device_type}")
