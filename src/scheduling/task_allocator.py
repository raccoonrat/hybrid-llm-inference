# hybrid-llm-inference/src/scheduling/task_allocator.py
from toolbox.logger import get_logger
from hardware_profiling import get_profiler
from model_zoo import get_model
from typing import Dict, Any
import numpy as np

class TaskAllocator:
    """任务分配器，负责决定任务运行的设备"""
    
    def __init__(self, hardware_config: Dict[str, Any], model_config: Dict[str, Any], gpu_power_threshold=None):
        """初始化分配器
        
        Args:
            hardware_config (Dict[str, Any]): 硬件配置
            model_config (Dict[str, Any]): 模型配置
            gpu_power_threshold (float, optional): GPU功率阈值
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
            key: get_profiler(key, cfg) for key, cfg in hardware_config.items()
        }
        
        # 初始化模型
        self.models = {
            name: get_model(
                model_name=name,
                mode=cfg.get("mode", "local"),
                config=cfg
            ) for name, cfg in model_config["models"].items()
        }
    
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

    def allocate(self, allocations, model_name="llama3"):
        """
        Execute tasks on assigned hardware and collect metrics.
        
        Args:
            allocations (list): List of allocations [{"query": dict, "hardware": str}].
            model_name (str): Model to use for inference.
        
        Returns:
            list: Results [{"query": dict, "hardware": str, "metrics": dict}].
        """
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")
        
        if not allocations:
            self.logger.warning("No allocations provided, returning empty results")
            return []
        
        model = self.models[model_name]
        results = []

        for allocation in allocations:
            query = allocation.get("query", {})
            hardware = allocation.get("hardware")
            if not query or not hardware:
                self.logger.warning(f"Skipping invalid allocation: {allocation}")
                continue
            
            prompt = query.get("prompt", "")
            input_tokens = query.get("input_tokens", 0)
            output_tokens = query.get("output_tokens", 0)
            
            if hardware not in self.profilers:
                self.logger.warning(f"Hardware {hardware} not supported, skipping")
                continue
            
            profiler = self.profilers[hardware]
            task = lambda: model.infer(prompt)
            
            try:
                metrics = profiler.measure(task, input_tokens, output_tokens)
                result = {
                    "query": query,
                    "hardware": hardware,
                    "metrics": metrics
                }
                results.append(result)
                self.logger.debug(f"Executed task on {hardware}: {metrics}")
            except Exception as e:
                self.logger.error(f"Failed to execute task on {hardware}: {e}")
                continue
        
        self.logger.info(f"Allocated and executed {len(results)} tasks")
        return results
