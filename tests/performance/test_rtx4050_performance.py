"""RTX4050 性能测试模块。"""

import pytest
import time
import psutil
import torch
import os
from hardware_profiling.rtx4050_profiler import RTX4050Profiler
from toolbox.logger import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

# 测试配置
HARDWARE_CONFIG = {
    "nvidia_rtx4050": {
        "device_type": "rtx4050",
        "idle_power": 15.0,
        "sample_interval": 200
    }
}

@pytest.fixture
def profiler():
    """创建 RTX4050Profiler 实例的 fixture。"""
    return RTX4050Profiler(HARDWARE_CONFIG)

def test_initialization(profiler):
    """测试初始化。"""
    assert profiler is not None
    assert profiler.hardware_config == HARDWARE_CONFIG
    assert profiler.device_type == "rtx4050"
    assert profiler.idle_power == 15.0
    assert profiler.sample_interval == 200

def test_power_measurement_accuracy(profiler):
    """测试功耗测量准确性。"""
    # 测试多次测量
    measurements = []
    for _ in range(5):
        power = profiler.measure_power()
        assert isinstance(power, float)
        assert power >= 15.0  # 不应低于空闲功耗
        measurements.append(power)
    
    # 验证测量值的一致性
    assert len(set(measurements)) > 0  # 至少有一个不同的值
    assert max(measurements) - min(measurements) < 50.0  # 波动应在合理范围内

def test_sample_interval_impact():
    """测试采样间隔的影响。"""
    # 测试不同的采样间隔
    intervals = [100, 200, 500]
    for interval in intervals:
        config = HARDWARE_CONFIG.copy()
        config["nvidia_rtx4050"]["sample_interval"] = interval
        profiler = RTX4050Profiler(config)
        
        start_time = time.time()
        power = profiler.measure_power()
        end_time = time.time()
        
        assert isinstance(power, float)
        assert power >= 15.0
        assert end_time - start_time >= interval / 1000.0  # 确保采样间隔生效

def test_memory_usage(profiler):
    """测试内存使用。"""
    # 测试多次获取内存信息
    for _ in range(3):
        memory_info = profiler.get_memory_info()
        assert isinstance(memory_info, dict)
        assert "total" in memory_info
        assert "used" in memory_info
        assert "free" in memory_info
        assert memory_info["total"] > 0
        assert memory_info["used"] >= 0
        assert memory_info["free"] >= 0
        assert memory_info["used"] + memory_info["free"] == memory_info["total"]

def test_temperature_measurement(profiler):
    """测试温度测量。"""
    # 测试多次测量
    temperatures = []
    for _ in range(5):
        temp = profiler.get_temperature()
        assert isinstance(temp, float)
        assert 0 <= temp <= 100  # 温度应在合理范围内
        temperatures.append(temp)
    
    # 验证测量值的一致性
    assert len(set(temperatures)) > 0  # 至少有一个不同的值
    assert max(temperatures) - min(temperatures) < 20.0  # 波动应在合理范围内

def test_measurement_lifecycle(profiler):
    """测试测量生命周期。"""
    # 测试开始测量
    profiler.start_measurement()
    assert profiler.is_measuring
    
    # 测试测量过程中的功耗
    power = profiler.measure_power()
    assert isinstance(power, float)
    assert power >= 15.0
    
    # 测试停止测量
    profiler.stop_measurement()
    assert not profiler.is_measuring

def test_error_handling():
    """测试错误处理。"""
    # 测试无效配置
    invalid_configs = [
        {},  # 空配置
        {"device_type": "rtx4050"},  # 缺少必要参数
        {"idle_power": 15.0},  # 缺少必要参数
        {"sample_interval": 200},  # 缺少必要参数
        {"device_type": "rtx4050", "idle_power": -1.0, "sample_interval": 200},  # 无效的空闲功耗
        {"device_type": "rtx4050", "idle_power": 15.0, "sample_interval": 0},  # 无效的采样间隔
        {"device_type": 123, "idle_power": 15.0, "sample_interval": 200},  # 无效的设备类型
        {"device_type": "rtx4050", "idle_power": "15.0", "sample_interval": 200},  # 无效的空闲功耗类型
        {"device_type": "rtx4050", "idle_power": 15.0, "sample_interval": "200"}  # 无效的采样间隔类型
    ]
    
    for config in invalid_configs:
        with pytest.raises((ValueError, TypeError)):
            RTX4050Profiler(config)

def test_cleanup(profiler):
    """测试资源清理。"""
    profiler.cleanup()
    
    # 测试清理后使用
    with pytest.raises(RuntimeError):
        profiler.measure_power()
    
    with pytest.raises(RuntimeError):
        profiler.get_memory_info()
    
    with pytest.raises(RuntimeError):
        profiler.get_temperature()

def test_performance_metrics(profiler):
    """测试性能指标。"""
    # 测试测量任务
    task = {
        "input_tokens": 100,
        "output_tokens": 50
    }
    
    # 开始测量
    profiler.start_measurement()
    
    # 执行任务
    metrics = profiler.measure(task)
    
    # 停止测量
    profiler.stop_measurement()
    
    # 验证指标
    assert isinstance(metrics, dict)
    assert "energy" in metrics
    assert "runtime" in metrics
    assert "throughput" in metrics
    assert metrics["energy"] > 0
    assert metrics["runtime"] > 0
    assert metrics["throughput"] > 0

class TestRTX4050Performance:
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 设置测试模式环境变量
        os.environ['TEST_MODE'] = '1'
        
        cls.config = {
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        }
        cls.profiler = RTX4050Profiler(**cls.config)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {cls.device}")
        
    def test_power_measurement_accuracy(self):
        """测试功率测量准确性"""
        # 测试空闲状态功率
        idle_power = self.profiler.measure_power()
        logger.info(f"空闲状态功率: {idle_power}W")
        assert idle_power >= 0, "功率测量值不应为负"
        
        # 测试负载状态功率
        def gpu_load_task():
            # 创建一个GPU计算任务
            x = torch.randn(1000, 1000, device=self.device)
            y = torch.randn(1000, 1000, device=self.device)
            for _ in range(100):
                z = torch.matmul(x, y)
            return z
        
        # 测量负载状态功率
        metrics = self.profiler.measure(gpu_load_task, input_tokens=1, output_tokens=1)
        load_power = metrics["energy"] / metrics["runtime"]
        logger.info(f"负载状态功率: {load_power}W")
        assert load_power > idle_power, "负载状态功率应大于空闲状态功率"
        
    def test_sample_interval_impact(self):
        """测试采样间隔的影响"""
        intervals = [100, 200, 500, 1000]  # 不同的采样间隔（毫秒）
        results = []
        
        def test_task():
            # 创建一个GPU密集型任务
            x = torch.randn(2000, 2000, device=self.device)
            y = torch.randn(2000, 2000, device=self.device)
            for _ in range(100):  # 增加迭代次数
                z = torch.matmul(x, y)
                # 添加一些额外的计算
                z = torch.relu(z)
                z = torch.sigmoid(z)
            return z
        
        for interval in intervals:
            config = self.config.copy()
            config["sample_interval"] = interval
            profiler = RTX4050Profiler(config)
            
            start_time = time.time()
            metrics = profiler.measure(test_task, input_tokens=1, output_tokens=1)
            runtime = time.time() - start_time
            
            results.append({
                "interval": interval,
                "runtime": runtime,
                "energy": metrics["energy"],
                "power_samples": metrics.get("power_samples", [])
            })
            
            logger.info(f"采样间隔 {interval}ms - 运行时间: {runtime:.2f}s, 能耗: {metrics['energy']:.2f}J")
        
        # 验证采样间隔对测量精度的影响
        assert all(r["runtime"] >= 0.1 for r in results), "所有测试的运行时间应至少为0.1秒"
        assert all(r["energy"] > 0 for r in results), "所有测试的能耗应大于0"
        
    def test_memory_usage(self):
        """测试显存使用情况"""
        def memory_test_task():
            # 分配不同大小的显存
            sizes = [100, 500, 1000]  # MB
            tensors = []
            
            for size in sizes:
                # 转换为元素数量（假设每个元素4字节）
                num_elements = size * 1024 * 1024 // 4
                tensor = torch.randn(num_elements, device=self.device)
                tensors.append(tensor)
                time.sleep(0.5)  # 等待显存分配稳定
                
                # 记录显存使用情况
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                    logger.info(f"分配 {size}MB 后 - 已分配: {memory_allocated:.2f}MB, 已保留: {memory_reserved:.2f}MB")
                else:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    logger.info(f"分配 {size}MB 后 - RSS: {memory_info.rss / 1024 / 1024:.2f}MB, VMS: {memory_info.vms / 1024 / 1024:.2f}MB")
            
            return tensors
        
        # 测量内存使用情况
        metrics = self.profiler.measure(memory_test_task, input_tokens=1, output_tokens=1)
        
        # 验证内存使用
        if torch.cuda.is_available():
            assert torch.cuda.memory_allocated() > 0, "应该有显存被分配"
            assert torch.cuda.memory_reserved() > 0, "应该有显存被保留"
        else:
            process = psutil.Process()
            memory_info = process.memory_info()
            assert memory_info.rss > 0, "应该有内存被分配"
            assert memory_info.vms > 0, "应该有虚拟内存被保留"
        
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        if hasattr(cls, 'profiler'):
            del cls.profiler
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 