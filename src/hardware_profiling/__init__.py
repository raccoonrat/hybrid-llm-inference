# hybrid-llm-inference/src/hardware_profiling/__init__.py
from .base_profiler import HardwareProfiler as BaseProfiler
from .rtx4050_profiler import RTX4050Profiler
from .a800_profiling import A800Profiler
from .m1_pro_profiler import M1ProProfiler
from .a100_profiler import A100Profiler

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.info("Logger initialized for %s", __name__)

def get_profiler(device_id: int = 0, skip_nvml: bool = False, config: Dict[str, Any] = None) -> Any:
    """
    获取性能分析器
    
    Args:
        device_id: GPU 设备 ID
        skip_nvml: 是否跳过 NVML 初始化
        config: 配置参数
        
    Returns:
        Any: 性能分析器实例
    """
    # 检查是否在测试模式下
    is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
    
    # 设置默认配置
    default_config = {
        'device_id': device_id,
        'idle_power': 15.0,
        'max_power': 115.0,
        'sample_interval': 0.1
    }
    
    # 合并配置
    if config:
        default_config.update(config)
    
    # 根据设备类型返回对应的性能分析器
    if os.name == "posix" and os.uname().machine == "arm64":
        return M1ProProfiler(config=default_config, skip_nvml=skip_nvml or is_test_mode)
    elif os.name == "nt":
        return RTX4050Profiler(config=default_config, skip_nvml=skip_nvml or is_test_mode)
    else:
        return A100Profiler(config=default_config, skip_nvml=skip_nvml or is_test_mode)
