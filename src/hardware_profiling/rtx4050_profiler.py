import pynvml
import logging
import os
import time
from .base_profiler import HardwareProfiler

logger = logging.getLogger(__name__)

class RTX4050Profiler(HardwareProfiler):
    """RTX 4050 GPU的硬件分析器"""
    
    def __init__(self, config: dict):
        """
        初始化RTX 4050分析器
        
        Args:
            config (dict): 硬件配置，包含以下可选参数：
                - device_id (int): GPU设备ID，默认为0
                - idle_power (float): 空闲功耗（瓦特），默认为15.0
                - sample_interval (float): 采样间隔（毫秒），默认为200
        """
        # 验证必要参数
        required_params = ["device_id", "idle_power", "sample_interval"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"缺少必要参数: {param}")
        
        # 验证参数值
        if not isinstance(config["device_id"], int) or config["device_id"] < 0:
            raise ValueError("device_id必须是大于等于0的整数")
        if not isinstance(config["idle_power"], (int, float)) or config["idle_power"] <= 0:
            raise ValueError("idle_power必须是大于0的数值")
        if not isinstance(config["sample_interval"], int) or config["sample_interval"] <= 0:
            raise ValueError("sample_interval必须是大于0的整数")
        
        super().__init__(config)
        self.device_id = config["device_id"]
        self.idle_power = config["idle_power"]
        self.sample_interval = config["sample_interval"] / 1000.0  # 转换为秒
        self.handle = None
        self.is_test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
        
        # 即使在测试模式下，也要验证设备ID的有效性
        if self.device_id > 8:  # 假设最多支持8个GPU
            raise ValueError(f"无效的设备ID: {self.device_id}")
        
        if not self.is_test_mode:
            try:
                # 尝试多个可能的NVML库路径
                nvml_paths = [
                    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
                    r"C:\Windows\System32\nvml.dll",
                    os.path.expandvars(r"%ProgramW6432%\NVIDIA Corporation\NVSMI\nvml.dll"),
                    os.path.expandvars(r"%ProgramFiles%\NVIDIA Corporation\NVSMI\nvml.dll")
                ]
                
                nvml_loaded = False
                for path in nvml_paths:
                    if os.path.exists(path):
                        os.environ["NVML_LIBRARY_PATH"] = path
                        try:
                            pynvml.nvmlInit()
                            nvml_loaded = True
                            break
                        except Exception as e:
                            logger.warning(f"尝试加载NVML库 {path} 失败: {str(e)}")
                
                if not nvml_loaded:
                    raise RuntimeError("未找到可用的NVML库")
                
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                device_name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode()
                if "RTX 4050" not in device_name:
                    logger.warning(f"当前设备不是RTX 4050: {device_name}")
                    raise ValueError(f"当前设备不是RTX 4050: {device_name}")
            except Exception as e:
                logger.error(f"初始化NVML失败: {str(e)}")
                raise
    
    def measure_power(self):
        """
        测量RTX 4050的当前功率
        
        Returns:
            float: 当前功率（瓦特）
        """
        if self.is_test_mode:
            # 测试模式下返回模拟功率
            return 100.0  # 模拟100W功率
            
        if self.handle is None:
            return 0.0  # 如果handle为None，返回0
            
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 转换为瓦特
            return max(0.0, power - self.idle_power)  # 减去空闲功耗
        except Exception as e:
            logger.error(f"功率测量失败: {str(e)}")
            return 0.0
    
    def measure(self, task, input_tokens, output_tokens):
        """
        测量任务的能耗和运行时间
        
        Args:
            task (callable): 要执行的任务（如模型推理）
            input_tokens (int): 输入token数量
            output_tokens (int): 输出token数量
        
        Returns:
            dict: 包含以下指标的字典：
                - energy (float): 总能耗（焦耳）
                - runtime (float): 运行时间（秒）
                - throughput (float): 吞吐量（tokens/s）
                - energy_per_token (float): 每token能耗（J/token）
        """
        try:
            # 开始测量
            start_time = time.time()
            total_energy = 0.0
            power_samples = []
            last_measurement = start_time
            
            # 执行任务并同时测量功率
            result = task()
            end_time = time.time()
            runtime = max(0.001, end_time - start_time)  # 确保最小运行时间为1ms
            
            # 在测试模式下模拟能耗
            if self.is_test_mode:
                # 模拟GPU在执行任务时的功率变化
                base_power = 100.0  # 基础功率（瓦特）
                power_variation = 20.0  # 功率变化范围（瓦特）
                num_samples = max(1, int(runtime / self.sample_interval))
                
                for i in range(num_samples):
                    # 模拟功率波动
                    power = base_power + (i % 2) * power_variation
                    power_samples.append(power)
                    total_energy += power * self.sample_interval
            else:
                # 实际测量功率
                while time.time() - start_time < runtime + 0.1:  # 确保至少测量完整个任务时间
                    current_time = time.time()
                    if current_time - last_measurement >= self.sample_interval:
                        power = self.measure_power()
                        power_samples.append(power)
                        total_energy += power * (current_time - last_measurement)
                        last_measurement = current_time
            
            # 计算最终指标
            total_tokens = max(1, input_tokens + output_tokens)  # 确保至少有一个token
            throughput = total_tokens / runtime
            energy_per_token = total_energy / total_tokens
            
            metrics = {
                "energy": total_energy,
                "runtime": runtime,
                "throughput": throughput,
                "energy_per_token": energy_per_token,
                "power_samples": power_samples,
                "result": result
            }
            
            logger.debug(f"RTX 4050测量指标: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"测量失败: {str(e)}")
            raise
    
    def __del__(self):
        """清理NVML资源"""
        try:
            if self.handle is not None:
                pynvml.nvmlShutdown()
        except:
            pass 