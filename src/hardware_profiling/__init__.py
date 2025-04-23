# hybrid-llm-inference/src/hardware_profiling/__init__.py
from .base_profiler import BaseProfiler
from .nvidia_profiling import NvidiaProfiler
from .apple_silicon_profiling import AppleSiliconProfiler
from .intel_cpu_profiling import IntelCPUProfiler
from .amd_cpu_profiling import AMDCPUProfiler
from .rtx4050_profiling import RTX4050Profiler
from .a800_profiling import A800Profiler

def get_profiler(hardware_type, config):
    """
    Return the appropriate profiler based on hardware type.
    
    Args:
        hardware_type (str): Hardware identifier (e.g., "m1_pro", "a100", "rtx4050", "a800").
        config (dict): Hardware configuration.
    
    Returns:
        BaseProfiler: Profiler instance.
    """
    profilers = {
        "m1_pro": AppleSiliconProfiler,
        "a100": NvidiaProfiler,
        "rtx4050": RTX4050Profiler,
        "a800": A800Profiler,
        "intel_cpu": IntelCPUProfiler,
        "amd_cpu": AMDCPUProfiler
    }
    profiler_class = profilers.get(hardware_type)
    if not profiler_class:
        raise ValueError(f"Unsupported hardware type: {hardware_type}")
    return profiler_class(config)
