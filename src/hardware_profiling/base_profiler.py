# hybrid-llm-inference/src/hardware_profiling/base_profiler.py
from abc import ABC, abstractmethod
import time
import logging

logger = logging.getLogger(__name__)

class HardwareProfiler(ABC):
    """硬件分析器的基类"""
    
    def __init__(self, config):
        """
        初始化硬件分析器
        
        Args:
            config (dict): 硬件配置
        """
        self.config = config
        self.idle_power = config.get("idle_power", 0.0)
        self.sample_interval = config.get("sample_interval", 200)  # ms
        
    @abstractmethod
    def measure_power(self):
        """
        测量当前功率
        
        Returns:
            float: 当前功率（瓦特）
        """
        pass
        
    def measure(self, task, input_tokens, output_tokens):
        """
        测量任务执行的能耗和运行时间
        
        Args:
            task (callable): 要执行的任务
            input_tokens (int): 输入token数量
            output_tokens (int): 输出token数量
            
        Returns:
            dict: 包含能耗、运行时间和吞吐量的指标
        """
        start_time = time.time()
        energy = 0.0
        
        try:
            # 执行任务并测量能耗
            task()
            end_time = time.time()
            runtime = end_time - start_time
            
            # 模拟能耗测量（实际实现中应该使用真实的硬件测量）
            power = self.measure_power()
            energy = power * runtime
            
            total_tokens = input_tokens + output_tokens
            throughput = total_tokens / runtime if runtime > 0 else 0
            energy_per_token = energy / total_tokens if total_tokens > 0 else 0
            
            return {
                "energy": energy,
                "runtime": runtime,
                "throughput": throughput,
                "energy_per_token": energy_per_token
            }
            
        except Exception as e:
            logger.error(f"测量失败: {str(e)}")
            raise

    def _compute_metrics(self, energy, runtime, total_tokens):
        """Compute standard metrics."""
        throughput = total_tokens / runtime if runtime > 0 else 0
        energy_per_token = energy / total_tokens if total_tokens > 0 else 0
        return {
            "energy": energy,
            "runtime": runtime,
            "throughput": throughput,
            "energy_per_token": energy_per_token
        }

