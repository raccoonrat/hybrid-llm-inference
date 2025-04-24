# hybrid-llm-inference/src/scheduling/token_based_scheduler.py
"""基于 token 的调度器模块。"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TokenBasedScheduler:
    """基于 token 的调度器类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化调度器。

        Args:
            config: 配置字典，必须包含：
                - thresholds: 阈值配置，包含 T_in 和 T_out
                - hardware_map: 硬件映射，包含 small、medium 和 large 的硬件类型
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not self.config:
            raise ValueError("配置不能为空")
            
        # 验证阈值配置
        if "thresholds" not in self.config:
            raise ValueError("配置缺少 thresholds")
            
        thresholds = self.config["thresholds"]
        if "T_in" not in thresholds or "T_out" not in thresholds:
            raise ValueError("thresholds 必须包含 T_in 和 T_out")
            
        if not isinstance(thresholds["T_in"], int) or thresholds["T_in"] <= 0:
            raise ValueError("T_in 必须是正整数")
            
        if not isinstance(thresholds["T_out"], int) or thresholds["T_out"] <= 0:
            raise ValueError("T_out 必须是正整数")
            
        # 验证硬件映射
        if "hardware_map" not in self.config:
            raise ValueError("配置缺少 hardware_map")
            
        hardware_map = self.config["hardware_map"]
        required_hardware = ["small", "medium", "large"]
        for hw in required_hardware:
            if hw not in hardware_map:
                raise ValueError(f"hardware_map 缺少 {hw} 硬件类型")
    
    def schedule(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """调度任务。

        Args:
            tasks: 任务列表，每个任务必须包含 input_tokens 和 output_tokens

        Returns:
            调度结果列表，每个结果包含：
                - hardware: 分配的硬件类型
                - input_tokens: 输入 token 数
                - output_tokens: 输出 token 数
                - task: 原始任务
        """
        results = []
        for task in tasks:
            input_tokens = task.get("input_tokens", 0)
            output_tokens = task.get("output_tokens", 0)
            
            # 根据输入 token 数确定硬件
            if input_tokens <= self.config["thresholds"]["T_in"]:
                hardware = self.config["hardware_map"]["small"]
            elif input_tokens <= self.config["thresholds"]["T_in"] * 2:
                hardware = self.config["hardware_map"]["medium"]
            else:
                hardware = self.config["hardware_map"]["large"]
            
            results.append({
                "hardware": hardware,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "task": task
            })
            
        return results
    
    def cleanup(self) -> None:
        """
        清理资源。
        """
        pass
