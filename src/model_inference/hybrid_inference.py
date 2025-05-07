"""混合推理模块。"""

import os
from typing import Dict, Any, List, Optional
from src.toolbox.logger import get_logger
from src.model_zoo.tinyllama import TinyLlama
from src.model_zoo.mistral import LocalMistral
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler

logger = get_logger(__name__)

class HybridInference:
    """混合推理类。"""
    
    def __init__(self, config: Dict[str, Any], test_mode: bool = False):
        """初始化混合推理。

        Args:
            config: 配置字典，包含：
                - model_name: 模型名称（必需）
                - model_path: 模型路径（必需）
                - device: 设备类型，默认为 "cuda"
                - mode: 运行模式，默认为 "local"
                - batch_size: 批处理大小，默认为 1
                - dtype: 数据类型，默认为 "float32"
            test_mode: 是否为测试模式
        """
        if not config:
            raise ValueError("配置不能为空")
        
        if "model_name" not in config:
            raise ValueError("配置必须包含 model_name")
            
        if "model_path" not in config:
            raise ValueError("配置必须包含 model_path")
            
        self.model_name = config["model_name"]
        self.model_path = config["model_path"]
        self.config = config
        self.test_mode = test_mode
        self.is_initialized = False
        self.profiler = RTX4050Profiler(config["scheduler_config"]["hardware_config"])
        
        # 初始化组件
        self._init_components()
        
        logger.info("混合推理初始化完成")
    
    def _init_components(self) -> None:
        """初始化组件。"""
        try:
            # 初始化模型
            from model_zoo import get_model
            self.model = get_model(self.model_name, {
                "model_path": self.model_path,
                "device": self.config.get("device", "cuda"),
                "mode": self.config.get("mode", "local"),
                "batch_size": self.config.get("batch_size", 1),
                "dtype": self.config.get("dtype", "float32")
            })
            logger.info(f"成功初始化模型 {self.model_name}，路径: {self.model_path}")
            
            # 初始化性能分析器
            self.profiler = RTX4050Profiler({
                "device_id": 0,
                "idle_power": 15.0,
                "sample_interval": 200
            })
            logger.info("性能分析器初始化完成")
            
            self.is_initialized = True
            logger.info("组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            raise
    
    def infer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理。

        Args:
            task: 任务字典，包含以下字段：
                - input: 输入文本
                - max_tokens: 最大生成令牌数

        Returns:
            Dict[str, Any]: 包含以下字段：
                - output: 生成的文本
                - metrics: 性能指标字典
        """
        if not self.is_initialized:
            raise RuntimeError("HybridInference 未初始化")
        if task is None:
            raise ValueError("任务不能为 None")
        if not isinstance(task, dict):
            raise TypeError("任务必须是字典类型")
        if "input" not in task or "max_tokens" not in task:
            raise ValueError("任务必须包含 input 和 max_tokens 字段")

        # 在测试模式下返回固定文本
        if self.test_mode:
            return {
                "output": "这是测试生成的文本",
                "metrics": {
                    "latency": 0.1,
                    "energy": 0.1
                }
            }

        try:
            # 开始性能测量
            self.profiler.start_monitoring()
            
            # 执行推理
            output = self.model.generate(
                task["input"],
                max_tokens=task["max_tokens"]
            )
            
            # 结束性能测量并获取指标
            metrics = self.profiler.stop_monitoring()
            
            return {
                "output": output,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def cleanup(self) -> None:
        """清理资源。"""
        try:
            # 清理模型
            if hasattr(self, "model"):
                self.model.cleanup()
                
            # 清理性能分析器
            if hasattr(self, "profiler"):
                self.profiler.cleanup()
                
            self.is_initialized = False
            logger.info("混合推理清理完成")
        except Exception as e:
            logger.error(f"清理失败: {e}")
            raise

    def generate(self, input_text: str, max_tokens: int) -> str:
        """生成文本。

        Args:
            input_text: 输入文本
            max_tokens: 最大生成令牌数

        Returns:
            str: 生成的文本
        """
        try:
            # 在测试模式下返回固定文本
            if self.test_mode:
                return "这是测试生成的文本"
            
            # 实际生成逻辑
            result = self.model.generate(
                input_text,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            return result
            
        except Exception as e:
            logger.error(f"生成文本时出错: {e}")
            return "" 