"""任务分配器模块。"""

import os
import logging
import yaml
from typing import Dict, Any
from .hardware_profiling import get_profiler

class TaskAllocator:
    """任务分配器类"""
    
    def __init__(self, device_name: str = "m1_pro"):
        """
        初始化任务分配器
        
        Args:
            device_name: 设备名称，默认为 m1_pro
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized for %s", __name__)
        
        # 检查是否在测试模式下
        self.is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
        
        # 从配置文件中读取硬件配置
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'hardware_config.yaml')
        with open(config_path, 'r') as f:
            hardware_configs = yaml.safe_load(f)
        
        # 获取设备配置
        self.hardware_config = hardware_configs.get(device_name, {})
        if not self.hardware_config:
            raise ValueError(f"Missing {device_name} configuration in hardware_config.yaml")
        
        # 初始化性能分析器
        self.profiler = get_profiler(device_name, self.hardware_config)
        
    def allocate(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        分配任务
        
        Args:
            task_config: 任务配置
            
        Returns:
            dict: 分配结果
        """
        # 测量性能
        power = self.profiler.measure_power()
        memory = self.profiler.measure_memory()
        temperature = self.profiler.measure_temperature()
        
        # 记录性能数据
        self.logger.info("Power: %.2fW, Memory: %d/%d bytes, Temperature: %.1f°C",
                        power, memory['used'], memory['total'], temperature)
        
        # 返回分配结果
        return {
            'power': power,
            'memory': memory,
            'temperature': temperature
        }
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'profiler'):
            self.profiler.cleanup()
            
    def __del__(self):
        """析构函数"""
        self.cleanup() 