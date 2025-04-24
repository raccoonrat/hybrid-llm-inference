# hybrid-llm-inference/src/model_zoo/base_model.py
"""基础模型模块。"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

class BaseModel(ABC):
    """基础模型类，为所有模型实现提供通用功能。
    
    属性:
        config (Dict[str, Any]): 模型配置
        logger (logging.Logger): 日志记录器
        is_test_mode (bool): 是否在测试模式下运行
        _total_tokens (int): 处理的总token数
        _total_time (float): 总运行时间
        _call_count (int): 调用次数
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化基础模型。
        
        Args:
            config: 模型配置，必须包含:
                - model_name: 模型名称
                - model_path: 模型路径
                - model_type: 模型类型
                - mode: 运行模式 ('local' 或 'remote')
                - batch_size: 批处理大小
                - max_length: 最大序列长度
                
        Raises:
            ValueError: 当配置参数无效时
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized for %s", __name__)
        
        # 检查是否在测试模式下
        self.is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
        
        # 验证配置
        self._validate_config(config)
        
        # 保存配置
        self.config = config
        
        # 初始化性能指标
        self._total_tokens = 0
        self._total_time = 0.0
        self._call_count = 0
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证模型配置。
        
        Args:
            config: 要验证的配置
            
        Raises:
            ValueError: 当配置无效时
        """
        required_fields = ["model_name", "model_path", "mode", "batch_size", "max_length"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"缺少必需的配置字段: {field}")
                
        if config["mode"] not in ["local", "remote"]:
            raise ValueError("mode必须是'local'或'remote'")
            
        if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
            raise ValueError("batch_size必须是正整数")
            
        if not isinstance(config["max_length"], int) or config["max_length"] <= 0:
            raise ValueError("max_length必须是正整数")
            
    @abstractmethod
    def _do_inference(self, input_text: str) -> str:
        """执行实际的推理操作。
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 生成的文本
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
        
    def inference(self, input_text: str) -> str:
        """执行推理并记录性能指标。
        
        Args:
            input_text: 输入文本
            
        Returns:
            str: 生成的文本
            
        Raises:
            ValueError: 当输入无效时
            RuntimeError: 当推理失败时
        """
        if not input_text:
            raise ValueError("输入文本不能为空")
            
        try:
            start_time = time.time()
            output = self._do_inference(input_text)
            end_time = time.time()
            
            # 更新性能指标
            input_tokens = self.get_token_count(input_text)
            output_tokens = self.get_token_count(output)
            self._total_tokens += input_tokens + output_tokens
            self._total_time += end_time - start_time
            self._call_count += 1
            
            return output
        except Exception as e:
            self.logger.error(f"推理失败: {str(e)}")
            raise RuntimeError(f"推理失败: {str(e)}")
            
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标。
        
        Returns:
            Dict[str, float]: 包含以下指标:
                - total_tokens: 处理的总token数
                - total_time: 总运行时间（秒）
                - avg_tokens_per_second: 平均每秒处理的token数
                - avg_time_per_call: 平均每次调用时间（秒）
        """
        if self._call_count == 0:
            return {
                "total_tokens": 0,
                "total_time": 0.0,
                "avg_tokens_per_second": 0.0,
                "avg_time_per_call": 0.0
            }
            
        return {
            "total_tokens": self._total_tokens,
            "total_time": self._total_time,
            "avg_tokens_per_second": self._total_tokens / self._total_time if self._total_time > 0 else 0.0,
            "avg_time_per_call": self._total_time / self._call_count
        }
        
    def reset_metrics(self) -> None:
        """重置性能指标。"""
        self._total_tokens = 0
        self._total_time = 0.0
        self._call_count = 0
        
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """获取文本的token数量。
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
        
    def cleanup(self) -> None:
        """清理资源。子类可以重写此方法以实现自定义清理逻辑。"""
        pass
