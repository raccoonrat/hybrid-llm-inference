# hybrid-llm-inference/src/scheduling/task_allocator.py
from toolbox.logger import get_logger
from src.hardware_profiling import get_profiler
from model_zoo import get_model
from typing import Dict, Any, List, Optional
import numpy as np
import logging
from pathlib import Path
from src.model_zoo.base_model import BaseModel
import os
from src.scheduling.token_based_scheduler import TokenBasedScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskAllocator:
    """任务分配器类。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化任务分配器。

        Args:
            config: 配置字典，必须包含：
                - thresholds: 阈值配置，包含 T_in 和 T_out
                - hardware_map: 硬件映射，包含 small、medium 和 large 的硬件类型
                - hardware_config: 硬件配置，包含每个硬件类型的配置
        """
        self.config = config
        self._validate_config()
        
        # 初始化调度器
        self.scheduler = TokenBasedScheduler({
            "thresholds": self.config["thresholds"],
            "hardware_map": self.config["hardware_map"]
        })
        
        # 初始化性能分析器
        self.profilers = {}
        for hw_type, hw_config in self.config["hardware_config"].items():
            try:
                self.profilers[hw_type] = get_profiler(hw_type, hw_config)
            except Exception as e:
                logger.warning(f"初始化 {hw_type} 性能分析器失败: {e}")
        
        self.device_stats = {
            "gpu": {"total_tasks": 0, "total_tokens": 0},
            "cpu": {"total_tasks": 0, "total_tokens": 0}
        }
        self.gpu_power_threshold = None
        self.last_device = None
        self.switch_penalty = 0.2  # 任务切换惩罚系数
        self.dynamic_threshold = 128  # 初始阈值
        self.threshold_history = []  # 阈值历史记录
        self.performance_history = []  # 性能历史记录
        self.min_threshold = 64  # 最小阈值
        self.max_threshold = 512  # 最大阈值
        self.adjustment_step = 16  # 阈值调整步长
        
        # 初始化模型
        self.models = {
            name: get_model(
                model_name=name,
                mode=cfg.get("mode", "local"),
                config=cfg
            ) for name, cfg in config["models"].items()
        }
    
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
                
        # 验证硬件配置
        if "hardware_config" not in self.config:
            raise ValueError("配置缺少 hardware_config")
            
        hardware_config = self.config["hardware_config"]
        for hw_type in hardware_map.values():
            if hw_type not in hardware_config:
                raise ValueError(f"hardware_config 缺少 {hw_type} 的配置")
    
    def update_threshold(self, current_performance: float) -> None:
        """根据性能历史动态调整阈值"""
        if len(self.performance_history) < 5:
            self.performance_history.append(current_performance)
            return
        
        # 计算最近5次性能的平均值
        recent_avg = np.mean(self.performance_history[-5:])
        
        if current_performance > recent_avg * 1.1:  # 性能下降超过10%
            # 降低阈值，减少GPU任务
            self.dynamic_threshold = max(
                self.min_threshold,
                self.dynamic_threshold - self.adjustment_step
            )
        elif current_performance < recent_avg * 0.9:  # 性能提升超过10%
            # 提高阈值，增加GPU任务
            self.dynamic_threshold = min(
                self.max_threshold,
                self.dynamic_threshold + self.adjustment_step
            )
        
        self.threshold_history.append(self.dynamic_threshold)
        self.performance_history.append(current_performance)
        
        # 保持历史记录长度
        if len(self.threshold_history) > 100:
            self.threshold_history.pop(0)
            self.performance_history.pop(0)
    
    def allocate_task(self, instruction: str, input_text: str, token_threshold: int) -> str:
        """分配任务到合适的设备
        
        Args:
            instruction: 指令文本
            input_text: 输入文本
            token_threshold: token数量阈值
            
        Returns:
            str: "cpu" 或 "gpu"
        """
        # 计算总token数
        total_tokens = len(instruction.split()) + len(input_text.split())
        
        # 使用动态阈值
        effective_threshold = min(token_threshold, self.dynamic_threshold)
        
        # 基础分数
        base_score = total_tokens / effective_threshold
        
        # 考虑任务切换开销
        if self.last_device is not None:
            if self.last_device == "cpu" and base_score > 1.0:
                # CPU -> GPU切换需要更高的阈值
                base_score *= (1 - self.switch_penalty)
            elif self.last_device == "gpu" and base_score <= 1.0:
                # GPU -> CPU切换需要更低的阈值
                base_score *= (1 + self.switch_penalty)
        
        # 根据分数决定设备
        device = "gpu" if base_score > 1.0 else "cpu"
        
        # 更新上一次使用的设备
        self.last_device = device
        
        # 更新统计信息
        self.device_stats[device]["total_tasks"] += 1
        self.device_stats[device]["total_tokens"] += total_tokens
        
        return device
    
    def get_device_stats(self) -> Dict[str, Dict[str, int]]:
        """获取设备使用统计"""
        return self.device_stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        for device in self.device_stats:
            self.device_stats[device] = {"total_tasks": 0, "total_tokens": 0}

    def allocate(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分配任务。

        Args:
            tasks: 任务列表，每个任务必须包含 input_tokens 和 output_tokens

        Returns:
            分配结果列表，每个结果包含：
                - hardware: 分配的硬件类型
                - input_tokens: 输入 token 数
                - output_tokens: 输出 token 数
                - task: 原始任务
                - profiler: 对应的性能分析器
        """
        try:
            # 使用调度器分配任务
            allocations = self.scheduler.schedule(tasks)
            
            # 为每个分配添加性能分析器
            for allocation in allocations:
                hw_type = allocation["hardware"]
                if hw_type in self.profilers:
                    allocation["profiler"] = self.profilers[hw_type]
                else:
                    logger.warning(f"未找到 {hw_type} 的性能分析器")
            
            return allocations
        except Exception as e:
            logger.error(f"分配任务失败: {e}")
            raise
    
    def cleanup(self) -> None:
        """清理资源。"""
        for profiler in self.profilers.values():
            try:
                profiler.cleanup()
            except Exception as e:
                logger.warning(f"清理性能分析器失败: {e}")
