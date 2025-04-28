# hybrid-llm-inference/src/optimization_engine/threshold_optimizer.py
import pickle
import numpy as np
from pathlib import Path
from toolbox.logger import get_logger
from .cost_function import CostFunction
from model_zoo import get_model
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import pandas as pd

logger = get_logger(__name__)

class ThresholdOptimizer:
    """阈值优化器类。
    
    用于优化混合推理系统的阈值参数。
    """
    
    def __init__(
        self,
        search_range: Tuple[float, float] = (0.0, 1.0),
        num_points: int = 10,
        device_id: str = "cuda:0",
        measure_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        """初始化阈值优化器。

        Args:
            search_range: 搜索范围元组 (min_threshold, max_threshold)
            num_points: 搜索点数量
            device_id: GPU设备ID
            measure_fn: 性能测量函数，接受任务字典，返回包含metrics的字典
        """
        self.search_range = search_range
        self.num_points = num_points
        self.device_id = device_id
        self.measure_fn = measure_fn or self._default_measure_fn
        self._validate_config()
        
        self.cost_function = CostFunction(
            config={
                "lambda_param": 0.5,
                "measure_fn": self.measure_fn,
                "device_id": self.device_id
            }
        )
    
    def _validate_config(self):
        """验证配置参数。"""
        if not isinstance(self.search_range, tuple) or len(self.search_range) != 2:
            raise ValueError("search_range 必须是包含两个元素的元组")
        
        min_threshold, max_threshold = self.search_range
        if not (isinstance(min_threshold, (int, float)) and isinstance(max_threshold, (int, float))):
            raise ValueError("阈值必须是数值类型")
            
        if min_threshold >= max_threshold:
            raise ValueError("最小阈值必须小于最大阈值")
            
        if not isinstance(self.num_points, int) or self.num_points < 2:
            raise ValueError("num_points 必须是大于1的整数")
            
        if not isinstance(self.device_id, str):
            raise ValueError("device_id 必须是字符串类型")
    
    def _default_measure_fn(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """默认性能测量函数。"""
        return {
            "metrics": {
                "latency": 1.0,
                "energy": 1.0,
                "accuracy": 0.9
            }
        }
    
    def optimize(self, task: Dict[str, Any]) -> float:
        """优化阈值。

        Args:
            task: 任务字典，包含输入文本和其他参数

        Returns:
            float: 最优阈值
        """
        if not isinstance(task, dict):
            raise ValueError("task 必须是字典类型")
        
        # 生成搜索点
        thresholds = np.linspace(
            self.search_range[0],
            self.search_range[1],
            self.num_points
        )
        
        # 计算每个阈值的成本
        costs = []
        for threshold in thresholds:
            # 获取输入和输出token数量
            input_tokens = task.get("input_tokens", [50])[0]
            output_tokens = task.get("max_tokens", 100)
            
            # 执行性能测量
            result = self.measure_fn(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                device_id=self.device_id
            )
            metrics = result.get("metrics", {})
            
            # 计算成本
            cost = self.cost_function.calculate(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                task=lambda: None
            )
            costs.append(cost)
        
        # 找到最优阈值
        optimal_idx = np.argmin(costs)
        return float(thresholds[optimal_idx])


