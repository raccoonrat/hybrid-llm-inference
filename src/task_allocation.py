"""
任务分配模块。
"""

import logging
from typing import Dict, Any, Optional
from .model_zoo.base_model import BaseModel
from .hardware_profiling import BaseProfiler

logger = logging.getLogger(__name__)

class TaskAllocator:
    """
    任务分配器，负责根据硬件性能和模型特性分配推理任务。
    """
    
    def __init__(self, model: BaseModel, profiler: BaseProfiler):
        """
        初始化任务分配器。

        Args:
            model: 基础模型实例
            profiler: 硬件性能分析器实例
        """
        self.model = model
        self.profiler = profiler
        logger.info("任务分配器初始化完成")
        
    def allocate(self, task: str) -> Dict[str, Any]:
        """
        为给定任务分配资源。

        Args:
            task: 任务描述

        Returns:
            Dict[str, Any]: 包含分配结果的字典
        """
        # 获取当前硬件状态
        power = self.profiler.measure_power()
        memory = self.profiler.measure_memory()
        temperature = self.profiler.measure_temperature()
        
        # 获取任务的token数量
        token_count = self.model.get_token_count(task)
        
        # 根据硬件状态和任务规模决定分配策略
        allocation = {
            "power_limit": power * 0.9,  # 设置功率限制为当前功率的90%
            "memory_required": token_count * 2,  # 估算所需内存
            "can_execute": memory["free"] > token_count * 2 and temperature < 80.0
        }
        
        logger.info(f"任务分配结果: {allocation}")
        return allocation
        
    def cleanup(self):
        """
        清理资源。
        """
        logger.info("清理任务分配器资源") 