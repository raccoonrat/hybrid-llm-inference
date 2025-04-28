# hybrid-llm-inference/src/hardware_profiling/base_profiler.py
"""硬件性能分析基类。"""

import os
import psutil
import platform
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from src.toolbox.logger import get_logger

logger = get_logger(__name__)

class HardwareProfiler(ABC):
    """硬件性能分析基类。"""
    
    def __init__(self, config=None):
        """初始化硬件性能分析器。

        Args:
            config: 配置字典，包含以下字段：
                - device_id: 设备 ID
                - device_type: 设备类型
                - idle_power: 空闲功率
                - sample_interval: 采样间隔
        """
        self.logger = logger
        self.config = config or {}
        self.device_id = self.config.get("device_id", 0)
        self.device_type = self.config.get("device_type", "gpu")
        self.idle_power = self.config.get("idle_power", 15.0)
        self.sample_interval = self.config.get("sample_interval", 200)
        self.initialized = False
        self.device = None
        self.handle = None
        self.nvml_initialized = False
        self.is_measuring = False
        self.is_test_mode = os.getenv('TEST_MODE') == '1'
        self.start_time = None
        self.start_energy = None
        self.gpu_handles = []
        self.system_info = self._get_system_info()
        self.metrics = {}
        
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息。
        
        Returns:
            Dict[str, Any]: 系统信息字典
        """
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        }
        
        if os.getenv('TEST_MODE') != '1':
            if torch.cuda.is_available():
                info.update({
                    "cuda_available": True,
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_device_memory": torch.cuda.get_device_properties(0).total_memory
                })
            else:
                info.update({
                    "cuda_available": False
                })
        else:
            info.update({
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_device_name": "MOCK GPU",
                "cuda_device_memory": 1024 * 1024 * 1024  # 1GB
            })
            
        return info
        
    @abstractmethod
    def profile_memory(self) -> Dict[str, float]:
        """分析内存使用情况。
        
        Returns:
            Dict[str, float]: 内存使用指标
        """
        pass
        
    @abstractmethod
    def profile_cpu(self) -> Dict[str, float]:
        """分析 CPU 使用情况。
        
        Returns:
            Dict[str, float]: CPU 使用指标
        """
        pass
        
    @abstractmethod
    def profile_gpu(self) -> Dict[str, float]:
        """分析 GPU 使用情况。
        
        Returns:
            Dict[str, float]: GPU 使用指标
        """
        pass
        
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标。
        
        Returns:
            Dict[str, Any]: 性能指标字典
        """
        return {
            "system_info": self.system_info,
            "memory": self.profile_memory(),
            "cpu": self.profile_cpu(),
            "gpu": self.profile_gpu()
        }
        
    def reset_metrics(self) -> None:
        """重置性能指标。"""
        self.metrics = {}

