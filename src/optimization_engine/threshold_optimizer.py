# hybrid-llm-inference/src/optimization_engine/threshold_optimizer.py
import pickle
import numpy as np
from pathlib import Path
from toolbox.logger import get_logger
from .cost_function import CostFunction
from model_zoo import get_model
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd

logger = get_logger(__name__)

class ThresholdOptimizer:
    """阈值优化器类。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化阈值优化器。

        Args:
            config: 配置字典，包含以下字段：
                - model_config: 模型配置
                - measure_fn: 测量函数
                - device_id: 设备ID
                - search_range: 搜索范围
                - num_points: 搜索点数
        """
        self.model_config = config.get("model_config", {})
        self.measure_fn = config.get("measure_fn")
        self.device_id = config.get("device_id", "cuda:0")
        self.search_range = config.get("search_range", (0.1, 0.9))
        self.num_points = config.get("num_points", 10)
        self._validate_config()
        
        # 初始化成本函数
        cost_fn_config = {
            "model_config": self.model_config,
            "measure_fn": self.measure_fn,
            "device_id": self.device_id
        }
        self.cost_function = CostFunction(cost_fn_config)
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.model_config, dict):
            raise ValueError("model_config 必须是字典")
        if not self.model_config:
            raise ValueError("model_config 不能为空")
        if not callable(self.measure_fn):
            raise ValueError("measure_fn 必须是可调用对象")
        if not isinstance(self.device_id, str):
            raise ValueError("device_id 必须是字符串")
        if not isinstance(self.search_range, tuple) or len(self.search_range) != 2:
            raise ValueError("search_range 必须是包含两个元素的元组")
        if not isinstance(self.num_points, int) or self.num_points <= 0:
            raise ValueError("num_points 必须是正整数")
    
    def _validate_task(self, task: Union[Dict[str, Any], pd.DataFrame]) -> Tuple[int, int]:
        """验证任务数据并提取令牌数。

        Args:
            task: 任务数据字典或 DataFrame

        Returns:
            输入令牌数和输出令牌数的元组
        """
        if isinstance(task, pd.DataFrame):
            # 计算所有行的输入和输出令牌总数
            total_input_tokens = 0
            total_output_tokens = 0
            
            for _, row in task.iterrows():
                input_tokens = len(row['input_tokens']) if isinstance(row['input_tokens'], list) else 0
                output_tokens = len(row['decoded_text']) if isinstance(row['decoded_text'], list) else 0
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
            return total_input_tokens, total_output_tokens
            
        elif isinstance(task, dict):
            # 尝试不同的键名
            input_tokens = task.get("input_tokens", 
                                  task.get("input_length",
                                  task.get("prompt_tokens", 0)))
            output_tokens = task.get("output_tokens",
                                   task.get("output_length",
                                   task.get("completion_tokens", 0)))
                                   
            if isinstance(input_tokens, str):
                input_tokens = len(input_tokens.split())
            if isinstance(output_tokens, str):
                output_tokens = len(output_tokens.split())
                
            return int(input_tokens), int(output_tokens)
        else:
            raise ValueError("任务数据必须是字典或 DataFrame 类型")
    
    def optimize(self, task: Dict[str, Any]) -> float:
        """优化阈值。

        Args:
            task: 任务字典，包含输入和输出标记

        Returns:
            最优阈值
        """
        input_tokens, output_tokens = self._validate_task(task)
        
        # 生成搜索点
        thresholds = np.linspace(
            self.search_range[0],
            self.search_range[1],
            self.num_points
        )
        
        # 计算每个阈值的成本
        costs = []
        for threshold in thresholds:
            self.model_config["threshold"] = threshold
            cost = self.cost_function.calculate(input_tokens, output_tokens)
            costs.append(cost)
        
        # 找到最小成本对应的阈值
        min_cost_idx = np.argmin(costs)
        optimal_threshold = thresholds[min_cost_idx]
        
        return optimal_threshold


