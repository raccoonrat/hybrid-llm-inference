import logging
from typing import Dict, Any, Optional
import psutil

from src.hardware_profiling.base_profiler import HardwareProfiler

logger = logging.getLogger(__name__)

class M1Profiler(HardwareProfiler):
    """Apple M1 处理器性能分析器。"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 M1 性能分析器。

        Args:
            config: 配置字典，必须包含：
                - device_type: 设备类型，必须为 "m1"
                - idle_power: 空闲功耗（瓦特）
                - sample_interval: 采样间隔（毫秒）
        """
        # 调用父类初始化
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        self.device_type = config.get("device_type", "m1")
        self.idle_power = config.get("idle_power", 10.0)  # M1的默认空闲功率较低
        self.sample_interval = config.get("sample_interval", 200)  # 毫秒
        
        # 验证配置
        self._validate_config()
        
    def _validate_config(self) -> None:
        """验证配置参数。"""
        if not self.config:
            raise ValueError("配置不能为空")
            
        if self.config.get("device_type") != "m1":
            raise ValueError("设备类型必须为 m1")
            
        if "idle_power" in self.config and not isinstance(self.config["idle_power"], (int, float)):
            raise ValueError("空闲功耗必须是数字")
            
        if "sample_interval" in self.config and not isinstance(self.config["sample_interval"], (int, float)):
            raise ValueError("采样间隔必须是数字")
    
    def measure_power(self) -> float:
        """测量当前功耗。

        Returns:
            当前功耗（瓦特）
        """
        # 由于无法直接测量 M1 功耗，返回预设值
        return self.idle_power
    
    def measure(self, task: Dict[str, Any]) -> Dict[str, float]:
        """测量任务执行的性能指标。

        Args:
            task: 任务信息字典

        Returns:
            性能指标字典，包含：
            - energy: 能耗（焦耳）
            - runtime: 运行时间（秒）
            - throughput: 吞吐量（token/秒）
            - energy_per_token: 每 token 能耗（焦耳/token）
        """
        # 模拟测量结果
        runtime = 1.0  # 假设运行时间为 1 秒
        energy = self.idle_power * runtime
        
        input_tokens = task.get("input_tokens", 0)
        output_tokens = task.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        
        return self._compute_metrics(energy, runtime, total_tokens)
    
    def _compute_metrics(self, energy: float, runtime: float, total_tokens: int) -> Dict[str, float]:
        """计算性能指标。

        Args:
            energy: 能耗（焦耳）
            runtime: 运行时间（秒）
            total_tokens: 总token数

        Returns:
            Dict[str, float]: 包含以下字段的性能指标：
                - energy: 能耗（焦耳）
                - runtime: 运行时间（秒）
                - throughput: 吞吐量（token/秒）
                - energy_per_token: 每token能耗（焦耳/token）
        """
        throughput = total_tokens / runtime if runtime > 0 else 0
        energy_per_token = energy / total_tokens if total_tokens > 0 else 0
        
        return {
            "energy": energy,
            "runtime": runtime,
            "throughput": throughput,
            "energy_per_token": energy_per_token
        }
    
    def cleanup(self) -> None:
        """清理资源。"""
        pass
    
    def profile_cpu(self) -> Dict[str, float]:
        """分析CPU使用情况。
        
        Returns:
            Dict[str, float]: CPU使用率和温度信息
        """
        # 使用psutil获取CPU信息
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        return {
            "cpu_usage": sum(cpu_percent) / len(cpu_percent),
            "cpu_temp": self._get_cpu_temperature()
        }
        
    def profile_gpu(self) -> Dict[str, float]:
        """分析GPU使用情况。
        
        Returns:
            Dict[str, float]: GPU使用率和性能指标
        """
        # M1的GPU集成在SoC中，使用系统命令获取信息
        return {
            "gpu_usage": self._get_gpu_usage(),
            "gpu_power": self._get_gpu_power()
        }
        
    def profile_memory(self) -> Dict[str, float]:
        """分析内存使用情况。
        
        Returns:
            Dict[str, float]: 内存使用信息
        """
        memory = psutil.virtual_memory()
        return {
            "total_memory": memory.total / (1024**3),  # GB
            "used_memory": memory.used / (1024**3),
            "free_memory": memory.available / (1024**3)
        }
        
    def _get_cpu_temperature(self) -> float:
        """获取CPU温度。"""
        # 这里需要使用macOS特定的命令
        try:
            import subprocess
            result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'smc', '-i0', '-n1'],
                                 capture_output=True, text=True, timeout=2)
            for line in result.stdout.split('\n'):
                if 'CPU die temperature' in line:
                    return float(line.split(':')[1].strip().replace('°C', ''))
        except:
            self.logger.warning("无法获取CPU温度")
            return 0.0
            
    def _get_gpu_usage(self) -> float:
        """获取GPU使用率。"""
        try:
            import subprocess
            result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'gpu', '-i0', '-n1'],
                                 capture_output=True, text=True, timeout=2)
            for line in result.stdout.split('\n'):
                if 'GPU active residency' in line:
                    return float(line.split(':')[1].strip().replace('%', ''))
        except:
            self.logger.warning("无法获取GPU使用率")
            return 0.0
            
    def _get_gpu_power(self) -> float:
        """获取GPU功耗。"""
        try:
            import subprocess
            result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'gpu', '-i0', '-n1'],
                                 capture_output=True, text=True, timeout=2)
            for line in result.stdout.split('\n'):
                if 'GPU power' in line:
                    return float(line.split(':')[1].strip().replace('W', ''))
        except:
            self.logger.warning("无法获取GPU功耗")
            return 0.0 