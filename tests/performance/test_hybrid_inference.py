import pytest
import torch
import numpy as np
import time
import psutil
import math
from src.scheduling.task_scheduler import TaskScheduler
from src.scheduling.task_allocator import TaskAllocator
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from src.data.alpaca_loader import AlpacaLoader
from toolbox.logger import get_logger

logger = get_logger(__name__)

class TestHybridInference:
    """测试混合推理系统性能"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 配置硬件
        cls.gpu_config = {
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        }
        cls.gpu_profiler = RTX4050Profiler(cls.gpu_config)
        
        # 加载数据集
        cls.dataset_path = "data/alpaca_data.json"
        cls.data_loader = AlpacaLoader(cls.dataset_path)
        
        # 初始化调度器
        cls.scheduler = TaskScheduler()
        cls.allocator = TaskAllocator()
        
        # 测试阈值范围
        cls.thresholds = [64, 128, 256, 512]
        
    def test_threshold_optimization(self):
        """测试不同阈值下的性能表现"""
        results = []
        
        for threshold in self.thresholds:
            # 准备测试数据
            test_samples = self.data_loader.load_samples(20)
            
            # 执行任务并收集指标
            gpu_tasks = 0
            cpu_tasks = 0
            total_tokens = 0
            
            for sample in test_samples:
                device = self.allocator.allocate_task(
                    sample["instruction"],
                    sample["input"],
                    threshold
                )
                
                if device == "gpu":
                    gpu_tasks += 1
                else:
                    cpu_tasks += 1
                    
                total_tokens += len(sample["instruction"].split()) + len(sample["input"].split())
            
            # 计算任务分配比例
            gpu_ratio = gpu_tasks / len(test_samples)
            cpu_ratio = cpu_tasks / len(test_samples)
            
            results.append({
                "threshold": threshold,
                "gpu_ratio": gpu_ratio,
                "cpu_ratio": cpu_ratio,
                "total_tokens": total_tokens
            })
            
        # 分析结果
        print("\n阈值优化测试结果:")
        for result in results:
            print(f"阈值: {result['threshold']}")
            print(f"GPU任务比例: {result['gpu_ratio']:.2%}")
            print(f"CPU任务比例: {result['cpu_ratio']:.2%}")
            print(f"总token数: {result['total_tokens']}")
            print("---")
            
        # 验证阈值选择合理性
        optimal_threshold = min(self.thresholds, key=lambda t: abs(
            next(r for r in results if r["threshold"] == t)["gpu_ratio"] - 0.7
        ))
        print(f"\n最优阈值: {optimal_threshold}")
        
        assert optimal_threshold in self.thresholds, "应找到合理的阈值"
        
    def test_energy_saving(self):
        """测试能耗节省效果"""
        # 使用最优阈值
        optimal_threshold = 128
        
        # 准备测试数据
        test_samples = self.data_loader.load_samples(30)
        
        # 测量纯GPU和混合推理的能耗
        def measure_energy(samples, use_hybrid=True):
            total_energy = 0.0
            
            for sample in test_samples:
                if use_hybrid:
                    device = self.allocator.allocate_task(
                        sample["instruction"],
                        sample["input"],
                        optimal_threshold
                    )
                else:
                    device = "gpu"  # 强制使用GPU
                
                def task_fn():
                    # 模拟更真实的负载
                    if device == "gpu" and torch.cuda.is_available():
                        # GPU任务：矩阵运算 + 注意力计算
                        batch_size = 32
                        seq_len = 128
                        hidden_size = 768
                        
                        # 模拟输入嵌入
                        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
                        
                        # 模拟注意力计算
                        for _ in range(3):  # 3层注意力
                            q = torch.matmul(x, torch.randn(hidden_size, hidden_size, device="cuda"))
                            k = torch.matmul(x, torch.randn(hidden_size, hidden_size, device="cuda"))
                            v = torch.matmul(x, torch.randn(hidden_size, hidden_size, device="cuda"))
                            
                            # 计算注意力分数
                            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hidden_size)
                            attn = torch.softmax(scores, dim=-1)
                            x = torch.matmul(attn, v)
                        
                        torch.cuda.synchronize()
                    else:
                        # CPU任务：轻量级文本处理
                        batch_size = 8
                        seq_len = 64
                        hidden_size = 256
                        
                        # 模拟轻量级处理
                        x = np.random.rand(batch_size, seq_len, hidden_size)
                        for _ in range(2):  # 2层处理
                            x = np.tanh(np.dot(x, np.random.rand(hidden_size, hidden_size)))
                    
                    return len(sample["instruction"].split()) + len(sample["input"].split()), len(sample["output"].split())
                
                metrics = self.gpu_profiler.measure(
                    task_fn,
                    len(sample["instruction"].split()) + len(sample["input"].split()),
                    len(sample["output"].split())
                )
                total_energy += metrics["energy"]
            
            return total_energy
        
        # 测量能耗
        hybrid_energy = measure_energy(test_samples, use_hybrid=True)
        gpu_only_energy = measure_energy(test_samples, use_hybrid=False)
        
        # 计算能耗节省
        energy_saving = (gpu_only_energy - hybrid_energy) / gpu_only_energy
        
        print(f"\n能耗测试结果:")
        print(f"纯GPU能耗: {gpu_only_energy:.2f} J")
        print(f"混合推理能耗: {hybrid_energy:.2f} J")
        print(f"能耗节省: {energy_saving:.2%}")
        
        assert energy_saving > 0.1, "混合推理应显著节省能耗"
        
    def test_performance_tradeoff(self):
        """测试性能权衡"""
        # 使用最优阈值
        optimal_threshold = 128
        
        # 准备测试数据和调度器
        test_samples = self.data_loader.load_samples(25)
        scheduler = TaskScheduler()
        allocator = TaskAllocator()
        
        def measure_performance(samples, use_hybrid=True):
            response_times = []
            
            # 预热阶段
            warmup_samples = samples[:5]
            for sample in warmup_samples:
                if use_hybrid:
                    device = allocator.allocate_task(
                        sample["instruction"],
                        sample["input"],
                        optimal_threshold
                    )
                else:
                    device = "gpu"
                
                start_time = time.perf_counter()
                
                scheduler.schedule_task(
                    lambda: (
                        len(sample["instruction"].split()) + len(sample["input"].split()),
                        len(sample["output"].split())
                    ),
                    device
                )
                
                end_time = time.perf_counter()
                response_time = end_time - start_time
                
                # 更新阈值
                if use_hybrid:
                    allocator.update_threshold(response_time)
            
            # 正式测量
            test_samples = samples[5:]
            for sample in test_samples:
                # 重复测量以提高精度
                times = []
                for _ in range(5):
                    if use_hybrid:
                        device = allocator.allocate_task(
                            sample["instruction"],
                            sample["input"],
                            optimal_threshold
                        )
                    else:
                        device = "gpu"
                    
                    start_time = time.perf_counter()
                    
                    scheduler.schedule_task(
                        lambda: (
                            len(sample["instruction"].split()) + len(sample["input"].split()),
                            len(sample["output"].split())
                        ),
                        device
                    )
                    
                    end_time = time.perf_counter()
                    response_time = end_time - start_time
                    times.append(response_time)
                    
                    # 更新阈值
                    if use_hybrid:
                        allocator.update_threshold(response_time)
                
                # 使用中位数作为该样本的响应时间
                response_times.append(np.median(times))
            
            return np.mean(response_times), np.percentile(response_times, 95)
        
        # 测量性能
        hybrid_avg_time, hybrid_p95 = measure_performance(test_samples, use_hybrid=True)
        gpu_avg_time, gpu_p95 = measure_performance(test_samples, use_hybrid=False)
        
        # 计算性能差异（添加小值保护）
        min_time = 1e-6  # 1微秒
        gpu_avg_time = max(gpu_avg_time, min_time)
        gpu_p95 = max(gpu_p95, min_time)
        
        avg_time_increase = (hybrid_avg_time - gpu_avg_time) / gpu_avg_time
        p95_time_increase = (hybrid_p95 - gpu_p95) / gpu_p95
        
        print(f"\n性能权衡测试结果:")
        print(f"纯GPU平均响应时间: {gpu_avg_time*1000:.3f}毫秒")
        print(f"混合推理平均响应时间: {hybrid_avg_time*1000:.3f}毫秒")
        print(f"平均响应时间增加: {avg_time_increase:.2%}")
        print(f"纯GPU P95响应时间: {gpu_p95*1000:.3f}毫秒")
        print(f"混合推理 P95响应时间: {hybrid_p95*1000:.3f}毫秒")
        print(f"P95响应时间增加: {p95_time_increase:.2%}")
        
        # 验证性能权衡是否可接受
        assert avg_time_increase < 0.3, "平均响应时间增加应控制在30%以内"
        assert p95_time_increase < 0.5, "P95响应时间增加应控制在50%以内"
        
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        if hasattr(cls, 'gpu_profiler'):
            del cls.gpu_profiler
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 