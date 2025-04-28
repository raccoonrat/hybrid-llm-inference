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
                - decoded_text: 解码后的文本
                - input_tokens: 输入令牌列表

        Returns:
            调度后的任务列表，每个任务包含以下字段：
                - input: 输入文本
                - input_tokens_count: 输入令牌数
                - output_tokens_count: 输出令牌数
                - model: 分配的模型
                - hardware: 分配的硬件
        """
        if not self.initialized:
            raise RuntimeError("调度器未初始化")
            
        # 如果是 DataFrame，转换为列表
        if hasattr(tasks, 'empty'):
            if tasks.empty:
                return []
            tasks = tasks.to_dict('records')
        elif not tasks:
            return []
            
        if not isinstance(tasks, list):
            raise TypeError("tasks 必须是列表类型")

        scheduled_tasks = []
        for task in tasks:
            if task is None:
                raise ValueError("任务不能为 None")
            if not isinstance(task, dict):
                raise TypeError("任务必须是字典类型")
            if "decoded_text" not in task or "input_tokens" not in task:
                raise ValueError("任务必须包含 decoded_text 和 input_tokens 字段")
                
            # 计算输入令牌数量
            input_tokens_count = len(task["input_tokens"])
            # 估计输出令牌数量（这里简单地假设为输入令牌数量的 1.5 倍）
            output_tokens_count = int(input_tokens_count * 1.5)
            total_tokens = input_tokens_count + output_tokens_count
            
            # 基于总令牌数和硬件配置进行分配
            if total_tokens <= 1000:
                model = "tinyllama"
                hardware = "apple_m1_pro"
            elif total_tokens <= 5000:
                model = "tinyllama"
                hardware = "nvidia_rtx4050"
            else:
                model = "llama3"
                hardware = "nvidia_rtx4090"
            
            scheduled_task = task.copy()
            scheduled_task.update({
                "input": task["decoded_text"],  # 原始输入文本
                "input_tokens_count": input_tokens_count,  # 输入令牌数
                "output_tokens_count": output_tokens_count,  # 输出令牌数
                "model": model,
                "hardware": hardware,
                "estimated_tokens": total_tokens
            })
            scheduled_tasks.append(scheduled_task)

        return scheduled_tasks

    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False
        logger.info("基于令牌的调度器清理完成")
