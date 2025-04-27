"""混合推理模块。"""

import os
from typing import Dict, Any, List, Optional
from toolbox.logger import get_logger
from src.model_zoo.tinyllama import TinyLlama
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler

logger = get_logger(__name__)

class HybridInference:
    """混合推理类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化混合推理。

        Args:
            config: 配置字典，包含：
                - models: 模型配置列表
                - scheduler_config: 调度器配置
        """
        self.config = config
        self.models = []
        self.scheduler = None
        self.is_initialized = False
        self.profiler = RTX4050Profiler(config["scheduler_config"]["hardware_config"])
        
        # 验证配置
        self._validate_config()
        
        # 初始化组件
        self._init_components()
        
        logger.info("混合推理初始化完成")
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not self.config:
            raise ValueError("配置不能为空")
            
        if "models" not in self.config:
            raise ValueError("模型配置不能为空")
            
        if not isinstance(self.config["models"], list):
            raise ValueError("模型配置必须是列表")
            
        if not self.config["models"]:
            raise ValueError("模型配置列表不能为空")
    
    def _init_components(self) -> None:
        """初始化组件。"""
        # 初始化模型
        for model_config in self.config["models"]:
            if model_config["name"].lower() == "tinyllama":
                model = TinyLlama(model_config["model_config"])
            else:
                raise ValueError(f"不支持的模型: {model_config['name']}")
            self.models.append(model)
        
        # 初始化调度器
        self.scheduler = TokenBasedScheduler(self.config.get("scheduler_config", {}))
        self.scheduler.initialize()
        
        self.is_initialized = True
        logger.info("组件初始化完成")
    
    def infer(self, task: Dict[str, Any]) -> str:
        """执行推理。

        Args:
            task: 任务字典，包含以下字段：
                - input: 输入文本
                - max_tokens: 最大生成令牌数

        Returns:
            生成的文本
        """
        if not self.is_initialized:
            raise RuntimeError("HybridInference 未初始化")
        if task is None:
            raise ValueError("任务不能为 None")
        if not isinstance(task, dict):
            raise TypeError("任务必须是字典类型")
        if "input" not in task or "max_tokens" not in task:
            raise ValueError("任务必须包含 input 和 max_tokens 字段")

        # 调度任务
        scheduled_tasks = self.scheduler.schedule([task])
        if not scheduled_tasks:
            raise RuntimeError("调度器未返回任何任务")
        
        scheduled_task = scheduled_tasks[0]
        model_name = scheduled_task["model"]
        hardware = scheduled_task["hardware"]
        
        # 选择对应的模型
        selected_model = None
        for model in self.models:
            if model.__class__.__name__.lower() == model_name.lower():
                selected_model = model
                break
        
        if selected_model is None:
            raise RuntimeError(f"未找到模型：{model_name}")
        
        # 执行推理
        return selected_model.generate(task["input"], max_tokens=task["max_tokens"])
    
    def cleanup(self) -> None:
        """清理资源。"""
        try:
            # 清理模型
            for model in self.models:
                model.cleanup()
                
            # 清理调度器
            if self.scheduler:
                self.scheduler.cleanup()
                
            # 清理性能分析器
            if self.profiler:
                self.profiler.cleanup()
                
            self.is_initialized = False
            logger.info("混合推理清理完成")
        except Exception as e:
            logger.error(f"清理失败: {e}")
            raise 