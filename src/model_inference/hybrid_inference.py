"""混合推理模块。"""

import os
from typing import Dict, Any, List, Optional
from toolbox.logger import get_logger
from src.model_zoo.base_model import BaseModel
from src.scheduling.token_based_scheduler import TokenBasedScheduler

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
        self.initialized = False
        
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
            model = BaseModel(model_config)
            self.models.append(model)
        
        # 初始化调度器
        self.scheduler = TokenBasedScheduler(self.config.get("scheduler_config", {}))
        self.scheduler.initialize()
        
        logger.info("组件初始化完成")
    
    def inference(self, input_text: str, max_tokens: int = 512) -> Dict[str, Any]:
        """执行推理。

        Args:
            input_text: 输入文本
            max_tokens: 最大令牌数

        Returns:
            推理结果
        """
        if not self.initialized:
            raise RuntimeError("混合推理未初始化")
            
        try:
            # 创建任务
            task = {
                "input": input_text,
                "max_tokens": max_tokens
            }
            
            # 调度任务
            scheduled_tasks = self.scheduler.schedule([task], self.models[0].__class__.__name__)
            
            if not scheduled_tasks:
                raise RuntimeError("任务调度失败")
                
            # 执行推理
            result = self.models[0].inference(input_text, max_tokens)
            
            return {
                "input": input_text,
                "output": result,
                "model": self.models[0].__class__.__name__
            }
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def cleanup(self) -> None:
        """清理资源。"""
        if self.initialized:
            try:
                # 清理模型
                for model in self.models:
                    model.cleanup()
                    
                # 清理调度器
                if self.scheduler:
                    self.scheduler.cleanup()
                    
                self.initialized = False
                logger.info("混合推理清理完成")
            except Exception as e:
                logger.error(f"清理失败: {e}")
                raise 