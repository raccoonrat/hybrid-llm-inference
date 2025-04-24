# hybrid-llm-inference/src/scheduling/token_based_scheduler.py
"""基于 token 的调度器模块。"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TokenBasedScheduler:
    """基于 token 的调度器。"""
    
    def __init__(self, thresholds: Dict[str, int], config: Dict[str, Any]):
        """
        初始化调度器。

        Args:
            thresholds: token 阈值，包含:
                - T_in: 输入 token 阈值
                - T_out: 输出 token 阈值
            config: 调度器配置，包含:
                - hardware_map: 硬件映射
        """
        self.thresholds = thresholds
        self.config = config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.thresholds, dict):
            raise ValueError("阈值必须是字典类型")
            
        if "T_in" not in self.thresholds or "T_out" not in self.thresholds:
            raise ValueError("阈值必须包含 T_in 和 T_out")
            
        if self.thresholds["T_in"] <= 0 or self.thresholds["T_out"] <= 0:
            raise ValueError("阈值必须为正数")
            
        if not isinstance(self.config, dict):
            raise ValueError("配置必须是字典类型")
            
        if "hardware_map" not in self.config:
            raise ValueError("配置必须包含 hardware_map")
            
    def schedule(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        调度任务。

        Args:
            tasks: 任务列表，每个任务包含:
                - prompt: 提示文本
                - response: 响应文本
                - input_tokens: 输入 token 数量
                - output_tokens: 输出 token 数量

        Returns:
            List[Dict[str, Any]]: 调度结果，每个任务包含:
                - hardware: 分配的硬件
                - query: 原始任务
        """
        results = []
        
        for task in tasks:
            try:
                # 获取 token 数量
                input_tokens = task.get("input_tokens", 0)
                output_tokens = task.get("output_tokens", 0)
                
                # 根据 token 数量选择硬件
                if input_tokens <= self.thresholds["T_in"] and output_tokens <= self.thresholds["T_out"]:
                    hardware = "nvidia_rtx4050"  # 小任务
                elif input_tokens <= self.thresholds["T_in"] * 2 and output_tokens <= self.thresholds["T_out"] * 2:
                    hardware = "apple_m1_pro"  # 中等任务
                else:
                    hardware = "nvidia_a100"  # 大任务
                    
                # 添加结果
                results.append({
                    "hardware": hardware,
                    "query": task
                })
                
            except Exception as e:
                logger.error(f"任务调度失败: {str(e)}")
                continue
                
        return results
    
    def cleanup(self) -> None:
        """
        清理资源。
        """
        pass
