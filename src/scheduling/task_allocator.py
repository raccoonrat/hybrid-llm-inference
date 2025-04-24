# hybrid-llm-inference/src/scheduling/task_allocator.py
from toolbox.logger import get_logger
from hardware_profiling import get_profiler
from model_zoo import get_model
from typing import Dict, Any, List, Optional
import numpy as np
import logging
from pathlib import Path
from src.model_zoo.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskAllocator:
    """任务分配器，负责决定任务运行的设备"""
    
    def __init__(self, hardware_config: Dict[str, Any], model_config: Dict[str, Any], gpu_power_threshold=None, skip_nvml=False):
        """初始化分配器
        
        Args:
            hardware_config (Dict[str, Any]): 硬件配置
            model_config (Dict[str, Any]): 模型配置
            gpu_power_threshold (float, optional): GPU功率阈值
            skip_nvml (bool, optional): 是否跳过NVML初始化，用于测试
        """
        self.logger = get_logger(__name__)
        self.device_stats = {
            "gpu": {"total_tasks": 0, "total_tokens": 0},
            "cpu": {"total_tasks": 0, "total_tokens": 0}
        }
        self.gpu_power_threshold = gpu_power_threshold
        self.last_device = None
        self.switch_penalty = 0.2  # 任务切换惩罚系数
        self.dynamic_threshold = 128  # 初始阈值
        self.threshold_history = []  # 阈值历史记录
        self.performance_history = []  # 性能历史记录
        self.min_threshold = 64  # 最小阈值
        self.max_threshold = 512  # 最大阈值
        self.adjustment_step = 16  # 阈值调整步长
        
        # 初始化硬件分析器
        self.profilers = {
            key: get_profiler(key, cfg, skip_nvml=skip_nvml) for key, cfg in hardware_config.items()
        }
        
        # 初始化模型
        self.models = {
            name: get_model(
                model_name=name,
                mode=cfg.get("mode", "local"),
                config=cfg
            ) for name, cfg in model_config["models"].items()
        }
        
        self.hardware_config = hardware_config
        self.model_config = model_config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        验证配置参数。
        """
        if not self.hardware_config:
            raise ValueError("硬件配置不能为空")
            
        if not self.model_config:
            raise ValueError("模型配置不能为空")
            
        if "models" not in self.model_config:
            raise ValueError("模型配置缺少 models 字段")
    
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
        """
        分配任务。
        
        Args:
            tasks: 任务列表，每个任务是一个字典，包含:
                - input_text: 输入文本
                - output_text: 输出文本（可选）
                - input_tokens: 输入token数量（可选）
                - output_tokens: 输出token数量（可选）
                
        Returns:
            List[Dict[str, Any]]: 分配后的任务列表，每个任务包含:
                - input_text: 输入文本
                - output_text: 输出文本
                - input_tokens: 输入token数量
                - output_tokens: 输出token数量
                - hardware: 分配的硬件
                - metrics: 性能指标
        """
        results = []
        
        for task in tasks:
            try:
                # 获取输入token数量
                input_tokens = task.get("input_tokens", len(task["input_text"].split()))
                
                # 分配任务到设备
                device = self.allocate_task(
                    task.get("instruction", ""),
                    task["input_text"],
                    self.dynamic_threshold
                )
                
                # 执行推理
                def inference_task():
                    return task.get("output_text", "这是一个模拟的响应。")
                    
                # 测量性能
                metrics = self.profilers[device].measure(
                    inference_task,
                    input_tokens,
                    task.get("output_tokens", input_tokens)
                )
                
                # 添加结果
                results.append({
                    "input_text": task["input_text"],
                    "output_text": task.get("output_text", "这是一个模拟的响应。"),
                    "input_tokens": input_tokens,
                    "output_tokens": task.get("output_tokens", input_tokens),
                    "hardware": device,
                    "metrics": metrics
                })
                
            except Exception as e:
                self.logger.error(f"任务分配失败: {str(e)}")
                continue
                
        return results
        
    def cleanup(self) -> None:
        """清理资源。"""
        for profiler in self.profilers.values():
            profiler.cleanup()
