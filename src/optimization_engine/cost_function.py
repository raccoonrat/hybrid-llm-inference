# hybrid-llm-inference/src/optimization_engine/cost_function.py
from typing import Dict, Any, Callable, Union, Optional
from toolbox.logger import get_logger

logger = get_logger(__name__)

class CostFunction:
    """成本函数类。"""
    
    def __init__(self, config: Union[Dict[str, Any], float], hardware_config: Optional[Dict[str, Any]] = None) -> None:
        """初始化成本函数。

        Args:
            config: 配置字典或 lambda 参数
            hardware_config: 硬件配置字典（当 config 为 lambda 参数时使用）
        """
        if isinstance(config, (float, int)):
            self.lambda_param = float(config)
            if hardware_config is None:
                raise ValueError("当 config 为 lambda 参数时，必须提供 hardware_config")
            self.model_config = {}
            self.measure_fn = lambda x, y, z: {"energy": x * y * 0.1, "runtime": x * y * 0.01}  # 简化的测量函数
            self.device_id = hardware_config.get("device_id", "cuda:0")
        else:
            self.model_config = config.get("model_config", {})
            self.measure_fn = config.get("measure_fn")
            self.device_id = config.get("device_id", "cuda:0")
            self.lambda_param = config.get("lambda_param", 0.5)
            
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.lambda_param, (float, int)):
            raise ValueError("lambda_param 必须是数值类型")
        if self.lambda_param < 0 or self.lambda_param > 1:
            raise ValueError("lambda_param 必须在 [0, 1] 范围内")
            
        if not isinstance(self.device_id, str):
            raise ValueError("device_id 必须是字符串")
            
        if not callable(self.measure_fn):
            raise ValueError("measure_fn 必须是可调用对象")
    
    def calculate(
        self,
        input_tokens: int,
        output_tokens: int,
        task: Optional[Callable] = None,
        system: Optional[str] = None,
        return_metrics: bool = False
    ) -> Union[float, Dict[str, float]]:
        """计算性能指标和成本。

        Args:
            input_tokens: 输入令牌数
            output_tokens: 输出令牌数
            task: 任务函数（可选）
            system: 系统名称（可选）
            return_metrics: 是否返回原始性能指标

        Returns:
            如果 return_metrics 为 True，返回性能指标字典
            否则返回成本值
        """
        try:
            # 使用测量函数获取性能指标
            metrics = self.measure_fn(input_tokens, output_tokens, self.device_id)
            
            if return_metrics:
                return metrics
            
            # 计算成本
            cost = self._calculate_cost(metrics)
            return cost
            
        except Exception as e:
            logger.error(f"计算失败: {str(e)}")
            raise

    def _calculate_cost(self, metrics: Dict[str, float]) -> float:
        """计算成本。

        Args:
            metrics: 性能指标字典，包含 "latency" 和 "energy" 键

        Returns:
            成本值
        """
        return self.lambda_param * metrics["energy"] + (1 - self.lambda_param) * metrics["latency"]

    def compute(self, task: Callable, input_tokens: int, output_tokens: int, system: str) -> Dict[str, float]:
        """计算任务的性能指标。

        Args:
            task: 任务函数
            input_tokens: 输入令牌数
            output_tokens: 输出令牌数
            system: 系统名称

        Returns:
            性能指标字典
        """
        try:
            # 使用测量函数获取性能指标
            metrics = self.measure_fn(input_tokens, output_tokens, self.device_id)
            return metrics
        except Exception as e:
            logger.error(f"性能指标计算失败: {str(e)}")
            raise
