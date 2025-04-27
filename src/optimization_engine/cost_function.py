# hybrid-llm-inference/src/optimization_engine/cost_function.py
from typing import Dict, Any, Callable
from toolbox.logger import get_logger

logger = get_logger(__name__)

class CostFunction:
    """成本函数类。"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化成本函数。

        Args:
            config: 配置字典，包含以下字段：
                - model_config: 模型配置
                - measure_fn: 测量函数
        """
        self.model_config = config.get("model_config", {})
        self.measure_fn = config.get("measure_fn")
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.model_config, dict):
            raise ValueError("model_config 必须是字典")
        if not self.model_config:
            raise ValueError("model_config 不能为空")
        if not callable(self.measure_fn):
            raise ValueError("measure_fn 必须是可调用对象")
    
    def calculate(self, input_tokens: int, output_tokens: int) -> float:
        """计算成本。

        Args:
            input_tokens: 输入令牌数
            output_tokens: 输出令牌数

        Returns:
            成本值
        """
        try:
            # 使用测量函数获取性能指标
            metrics = self.measure_fn(input_tokens, output_tokens)
            
            # 计算成本
            cost = metrics["energy"] + metrics["runtime"] * 0.1
            
            return cost
        except Exception as e:
            logger.error(f"成本计算失败: {str(e)}")
            raise
