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

            # 处理 device_id，兼容字符串和整数
            device_id = hardware_config.get("device_id", "cuda:0")
            if isinstance(device_id, str) and device_id.startswith("cuda:"):
                try:
                    device_id_int = int(device_id.split(":")[1])
                except Exception:
                    raise ValueError(f"无法解析 device_id: {device_id}")
            elif isinstance(device_id, int):
                device_id_int = device_id
            else:
                raise ValueError("device_id 必须是整数或形如 'cuda:N' 的字符串")
            hardware_config = dict(hardware_config)  # 拷贝，避免副作用
            hardware_config["device_id"] = device_id_int
            self.device_id = device_id_int
            self.hardware_config = hardware_config

            # 根据硬件配置初始化对应的 profiler
            device_type = hardware_config.get("device_type", "rtx4050")
            logger.info(f"初始化 profiler，设备类型: {device_type}")
            
            try:
                if device_type == "rtx4050":
                    from ..hardware_profiling.rtx4050_profiler import RTX4050Profiler
                    self.profiler = RTX4050Profiler(hardware_config)
                elif device_type == "a100":
                    from ..hardware_profiling.a100_profiler import A100Profiler
                    self.profiler = A100Profiler(hardware_config)
                elif device_type == "a800":
                    from ..hardware_profiling.a800_profiling import A800Profiler
                    self.profiler = A800Profiler(hardware_config)
                elif device_type == "m1_pro":
                    from ..hardware_profiling.m1_pro_profiler import M1ProProfiler
                    self.profiler = M1ProProfiler(hardware_config)
                else:
                    raise ValueError(f"不支持的设备类型: {device_type}")
                
                self.measure_fn = self.profiler.measure
                logger.info(f"成功初始化 {device_type} profiler")
            except Exception as e:
                logger.error(f"初始化 profiler 失败: {e}")
                raise
            
            self.model_config = {}
        else:
            self.model_config = config.get("model_config", {})
            self.measure_fn = config.get("measure_fn")
            self.device_id = config.get("device_id", "cuda:0")
            self.lambda_param = config.get("lambda_param", 0.5)
            self.hardware_config = hardware_config
            
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.lambda_param, (float, int)):
            raise ValueError("lambda_param 必须是数值类型")
        if self.lambda_param < 0 or self.lambda_param > 1:
            raise ValueError("lambda_param 必须在 [0, 1] 范围内")
        
        if self.hardware_config is not None:
            device_type = self.hardware_config.get("device_type", "")
            if device_type in ["nvidia", "a100", "rtx4050"]:
                if not (isinstance(self.device_id, int) or (isinstance(self.device_id, str) and self.device_id.startswith("cuda:"))):
                    raise ValueError("NVIDIA 设备的 device_id 必须是整数或形如 'cuda:N' 的字符串")
            else:
                if not isinstance(self.device_id, int):
                    raise ValueError("非NVIDIA设备的 device_id 必须是整数")
        
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
            logger.info(f"开始计算性能指标: input_tokens={input_tokens}, output_tokens={output_tokens}")
            if task is None:
                logger.error("必须提供任务函数")
                raise ValueError("必须提供任务函数")
            # 根据是否有 profiler 区分参数顺序
            if hasattr(self, "profiler"):
                metrics = self.measure_fn(task, input_tokens, output_tokens)
            else:
                metrics = self.measure_fn(input_tokens, output_tokens, self.device_id)
            logger.info(f"测量结果: input={input_tokens}, output={output_tokens}, metrics={metrics}")
            if return_metrics:
                return metrics
            cost = self._calculate_cost(metrics)
            logger.info(f"计算得到成本: {cost}")
            return cost
        except Exception as e:
            logger.error(f"计算失败: {e}")
            raise

    def _calculate_cost(self, metrics: Dict[str, float]) -> float:
        """计算成本。

        Args:
            metrics: 性能指标字典，包含 "latency" 或 "runtime" 和 "energy" 键

        Returns:
            成本值
        """
        latency = metrics.get("latency", metrics.get("runtime"))
        if latency is None:
            raise ValueError("metrics 中缺少 latency/runtime 字段")
        return self.lambda_param * metrics["energy"] + (1 - self.lambda_param) * latency

    def compute(self, task: Callable, input_tokens: int, output_tokens: int, system: str) -> Dict[str, float]:
        """计算任务的性能指标。"""
        try:
            if hasattr(self, "profiler"):
                metrics = self.measure_fn(task, input_tokens, output_tokens)
            else:
                metrics = self.measure_fn(input_tokens, output_tokens, self.device_id)
            return metrics
        except Exception as e:
            logger.error(f"性能指标计算失败: {str(e)}")
            raise
