# hybrid-llm-inference/src/optimization_engine/threshold_optimizer.py
import pickle
import numpy as np
from pathlib import Path
from toolbox.logger import get_logger
from .cost_function import CostFunction
from model_zoo import get_model
from typing import Dict, Any, List, Optional

logger = get_logger(__name__)

class ThresholdOptimizer:
    """阈值优化器类。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化阈值优化器。

        Args:
            config: 配置字典，包含以下字段：
                - model_config: 模型配置
                - measure_fn: 测量函数
                - min_threshold: 最小阈值
                - max_threshold: 最大阈值
                - step_size: 步长
        """
        self.model_config = config.get("model_config", {})
        self.measure_fn = config.get("measure_fn")
        self.min_threshold = config.get("min_threshold", 100)
        self.max_threshold = config.get("max_threshold", 1000)
        self.step_size = config.get("step_size", 100)
        self._validate_config()
        self.cost_function = CostFunction({
            "model_config": self.model_config,
            "measure_fn": self.measure_fn
        })
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.model_config, dict):
            raise ValueError("model_config 必须是字典")
        if not self.model_config:
            raise ValueError("model_config 不能为空")
        if not callable(self.measure_fn):
            raise ValueError("measure_fn 必须是可调用对象")
        if not isinstance(self.min_threshold, (int, float)) or self.min_threshold <= 0:
            raise ValueError("min_threshold 必须是正数")
        if not isinstance(self.max_threshold, (int, float)) or self.max_threshold <= self.min_threshold:
            raise ValueError("max_threshold 必须大于 min_threshold")
        if not isinstance(self.step_size, (int, float)) or self.step_size <= 0:
            raise ValueError("step_size 必须是正数")
    
    def optimize(self, tasks: List[Dict[str, Any]]) -> float:
        """优化阈值。

        Args:
            tasks: 任务列表，每个任务包含以下字段：
                - input_tokens: 输入令牌数
                - output_tokens: 输出令牌数

        Returns:
            最优阈值
        """
        try:
            best_threshold = self.min_threshold
            best_cost = float('inf')
            
            # 遍历所有可能的阈值
            for threshold in range(self.min_threshold, self.max_threshold + 1, self.step_size):
                total_cost = 0.0
                
                # 计算每个任务的成本
                for task in tasks:
                    input_tokens = task["input_tokens"]
                    output_tokens = task["output_tokens"]
                    
                    # 根据阈值选择模型
                    if input_tokens + output_tokens <= threshold:
                        cost = self.cost_function.calculate(input_tokens, output_tokens)
                    else:
                        cost = self.cost_function.calculate(input_tokens, output_tokens) * 1.5
                    
                    total_cost += cost
                
                # 更新最优阈值
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_threshold = threshold
            
            return best_threshold
        except Exception as e:
            logger.error(f"阈值优化失败: {str(e)}")
            raise


