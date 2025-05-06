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
from src.scheduling.base_allocator import BaseAllocator
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查是否在测试模式下
is_test_mode = os.environ.get('TEST_MODE', '0') == '1'

class TaskAllocator(BaseAllocator):
    """任务分配器类"""
    
    def __init__(self, hardware_config: Dict[str, Any], model_config: Dict[str, Any], device_name: str = "m1_pro"):
        """
        初始化任务分配器
        
        Args:
            hardware_config: 硬件配置
            model_config: 模型配置
            device_name: 设备名称，默认为 m1_pro
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized for %s", __name__)
        
        # 检查是否在测试模式下
        self.is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
        
        # 保存配置
        self.hardware_config = hardware_config
        self.model_config = model_config
        
        # 加载调度器配置
        scheduler_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'scheduler_config.yaml')
        with open(scheduler_config_path, 'r', encoding='utf-8') as f:
            self.scheduler_config = yaml.safe_load(f)
        
        # 获取设备配置
        if not self.hardware_config:
            # 从配置文件中读取硬件配置
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'hardware_config.yaml')
            with open(config_path, 'r') as f:
                hardware_configs = yaml.safe_load(f)
            self.hardware_config = hardware_configs.get(device_name, {})
            if not self.hardware_config:
                raise ValueError(f"Missing {device_name} configuration in hardware_config.yaml")
        
        # 初始化性能分析器
        self.profiler = get_profiler(device_name, self.hardware_config)
        
        # 初始化基本属性
        self.initialized = False
        self.token_threshold = self.scheduler_config.get("scheduler", {}).get("token_threshold", 128)  # 从调度器配置中获取阈值
        
        # 验证配置
        self._validate_config()
        
        # 初始化分配器
        self._init_allocator()
    
    def _validate_config(self) -> None:
        """验证配置。"""
        if not isinstance(self.hardware_config, dict):
            raise ValueError("hardware_config 必须是字典")
        if not isinstance(self.model_config, dict):
            raise ValueError("model_config 必须是字典")
        if not self.hardware_config:
            raise ValueError("hardware_config 不能为空")
        if not self.model_config:
            raise ValueError("model_config 不能为空")
        
        # 验证硬件配置
        for hardware_name, hardware_info in self.hardware_config.items():
            if not isinstance(hardware_info, dict):
                raise ValueError(f"硬件 {hardware_name} 的配置必须是字典")
            
            # 设置默认值
            if "device_type" not in hardware_info:
                hardware_info["device_type"] = "gpu"
            if "idle_power" not in hardware_info:
                hardware_info["idle_power"] = 15.0
            if "sample_interval" not in hardware_info:
                hardware_info["sample_interval"] = 200
            
            required_fields = ["device_type", "idle_power", "sample_interval"]
            for field in required_fields:
                if field not in hardware_info:
                    raise ValueError(f"硬件 {hardware_name} 缺少必要字段: {field}")
                if field == "idle_power" and hardware_info[field] <= 0:
                    raise ValueError(f"硬件 {hardware_name} 的 idle_power 必须大于 0")
                if field == "sample_interval" and hardware_info[field] <= 0:
                    raise ValueError(f"硬件 {hardware_name} 的 sample_interval 必须大于 0")
        
        # 验证模型配置
        if "models" not in self.model_config:
            raise ValueError("model_config 必须包含 models 字段")
        for model_name, model_info in self.model_config["models"].items():
            if not isinstance(model_info, dict):
                raise ValueError(f"模型 {model_name} 的配置必须是字典")
            required_fields = ["model_name", "model_path", "mode", "batch_size", "max_length"]
            for field in required_fields:
                if field not in model_info:
                    raise ValueError(f"模型 {model_name} 缺少必要字段: {field}")
    
    def _init_allocator(self) -> None:
        """初始化分配器。"""
        self.initialized = True
        logger.info("任务分配器初始化完成")
    
    def _select_hardware(self, total_tokens: int) -> str:
        """
        根据总令牌数选择硬件。
        
        Args:
            total_tokens: 总令牌数
            
        Returns:
            选择的硬件名称
        """
        if total_tokens <= 0:
            raise ValueError("总令牌数必须大于 0")
        
        # 由于只有 RTX 4050，所以直接返回
        return "nvidia_rtx4050"
    
    def allocate(self, tasks: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """
        分配任务。
        
        Args:
            tasks: 任务列表，每个任务包含以下字段：
                - input_tokens: 输入令牌数
                - output_tokens: 输出令牌数
                - model: 模型名称
            model_name: 模型名称
                
        Returns:
            分配后的任务列表，每个任务包含以下字段：
                - input_tokens: 输入令牌数
                - output_tokens: 输出令牌数
                - model: 模型名称
                - hardware: 分配的硬件
        """
        if not self.initialized:
            raise RuntimeError("分配器未初始化")
        if not tasks:
            return []
        
        allocated_tasks = []
        for task in tasks:
            # 验证任务
            if not isinstance(task, dict):
                raise ValueError("任务必须是字典")
            if "input_tokens_count" not in task or "output_tokens_count" not in task:
                raise ValueError("任务必须包含 input_tokens_count 和 output_tokens_count 字段")
            
            input_tokens = task["input_tokens_count"]
            output_tokens = task["output_tokens_count"]
            
            # 验证令牌数
            if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
                raise ValueError("input_tokens_count 和 output_tokens_count 必须是整数")
            if input_tokens < 0 or output_tokens < 0:
                raise ValueError("input_tokens_count 和 output_tokens_count 不能为负数")
            
            # 验证模型
            if model_name not in self.model_config["models"]:
                raise ValueError(f"未知的模型: {model_name}")
            
            # 选择硬件
            total_tokens = input_tokens + output_tokens
            hardware = self._select_hardware(total_tokens)
            
            # 验证硬件
            if hardware not in self.hardware_config:
                raise ValueError(f"未知的硬件: {hardware}")
            
            allocated_task = task.copy()
            allocated_task.update({
                "model": model_name,
                "hardware": hardware
            })
            allocated_tasks.append(allocated_task)
        
        return allocated_tasks
    
    def cleanup(self) -> None:
        """清理资源。"""
        self.initialized = False
        logger.info("任务分配器已清理")
