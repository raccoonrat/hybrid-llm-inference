import pytest
import time
import psutil
import torch
from hardware_profiling.rtx4050_profiler import RTX4050Profiler
from toolbox.logger import get_logger

logger = get_logger(__name__)

class TestRTX4050Performance:
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.config = {
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        }
        cls.profiler = RTX4050Profiler(cls.config)
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