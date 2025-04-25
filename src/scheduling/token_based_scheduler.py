# hybrid-llm-inference/src/scheduling/token_based_scheduler.py
"""基于令牌的调度器模块。"""

import os
from typing import Dict, Any, List, Optional
from toolbox.logger import get_logger
from .base_scheduler import BaseScheduler

logger = get_logger(__name__)

class TokenBasedScheduler(BaseScheduler):
    """基于令牌的调度器。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化基于令牌的调度器。

        Args:
            config: 配置字典，包含以下字段：
                - token_threshold: 令牌阈值
                - hardware_config: 硬件配置
                - model_config: 模型配置
        """
        # 初始化基本属性
        self.token_threshold = config.get("token_threshold", 1000)
        self.hardware_config = config.get("hardware_config", {})
        self.model_config = config.get("model_config", {})
        self.initialized = False

        # 验证配置
        self._validate_config()
        
        # 调用父类构造函数
        super().__init__(config)
        
        # 初始化调度器
        self._init_scheduler()
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.token_threshold, (int, float)) or self.token_threshold <= 0:
            raise ValueError("令牌阈值必须是正数")
        if not isinstance(self.hardware_config, dict):
            raise ValueError("hardware_config 必须是字典")
        if not isinstance(self.model_config, dict):
            raise ValueError("model_config 必须是字典")
    
    def _init_scheduler(self) -> None:
        """初始化调度器。"""
        if self.initialized:
            return
            
        self.initialized = True
        logger.info("基于令牌的调度器初始化完成")
    
    def schedule(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """调度任务。

        Args:
            tasks: 任务列表，每个任务包含以下字段：
                - tokens: 令牌数量

        Returns:
            调度后的任务列表，每个任务包含以下字段：
                - tokens: 令牌数量
                - model: 分配的模型
                - hardware: 分配的硬件
        """
        if not self.initialized:
            raise RuntimeError("调度器未初始化")
        if not tasks:
            return []
        if not isinstance(tasks, list):
            raise TypeError("tasks 必须是列表类型")

        scheduled_tasks = []
        for task in tasks:
            if task is None:
                raise ValueError("任务不能为 None")
            if not isinstance(task, dict):
                raise TypeError("任务必须是字典类型")
            if "tokens" not in task:
                raise ValueError("任务必须包含 tokens 字段")
                
            tokens = task["tokens"]
            if not isinstance(tokens, (int, float)):
                raise ValueError("令牌数量必须是数字")
            if tokens < 0:
                raise ValueError("令牌数量不能为负数")
                
            if tokens <= self.token_threshold:
                model = "tinyllama"
                hardware = "apple_m1_pro"
            else:
                model = "llama3"
                hardware = "rtx4050"
            
            scheduled_tasks.append({
                "tokens": tokens,
                "model": model,
                "hardware": hardware
            })

        return scheduled_tasks

    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False
        logger.info("基于令牌的调度器清理完成")
