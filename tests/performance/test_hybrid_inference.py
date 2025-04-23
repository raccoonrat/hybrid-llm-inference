import pytest
import os
from performance.hybrid_inference import HybridInference
from scheduling.task_allocator import TaskAllocator

@pytest.fixture
def hardware_config():
    """硬件配置"""
    return {
        "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
        "a100": {"type": "gpu", "device_id": 0},
        "rtx4050": {
            "type": "gpu",
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        }
    }

@pytest.fixture
def model_config():
    """模型配置"""
    return {
        "model_name": "tinyllama",
        "model_path": "models/TinyLlama-1.1B-Chat-v1.0",
        "mode": "local",
        "batch_size": 1,
        "max_length": 128
    }

class TestHybridInference:
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 设置测试模式环境变量
        os.environ['TEST_MODE'] = '1'
        
        cls.hardware_config = {
            "m1_pro": {"type": "cpu_gpu", "idle_power": 10.0},
            "a100": {"type": "gpu", "device_id": 0},
            "rtx4050": {
                "type": "gpu",
                "device_id": 0,
                "idle_power": 15.0,
                "sample_interval": 200
            }
        }
        
        cls.model_config = {
            "model_name": "tinyllama",
            "model_path": "models/TinyLlama-1.1B-Chat-v1.0",
            "mode": "local",
            "batch_size": 1,
            "max_length": 128
        }
        
        cls.allocator = TaskAllocator(
            hardware_config=cls.hardware_config,
            model_config=cls.model_config
        )
        
        cls.inference = HybridInference(cls.allocator)
    
    def test_threshold_optimization(self):
        """测试阈值优化"""
        thresholds = self.inference.optimize_thresholds()
        assert "T_in" in thresholds
        assert "T_out" in thresholds
        assert thresholds["T_in"] > 0
        assert thresholds["T_out"] > 0
    
    def test_energy_saving(self):
        """测试节能效果"""
        task = {
            "input_tokens": 32,
            "output_tokens": 32
        }
        
        device = self.allocator.allocate(task)
        assert device in ["m1_pro", "a100", "rtx4050"]
        
        metrics = self.inference.measure_performance(task)
        assert "energy" in metrics
        assert "runtime" in metrics
        assert metrics["energy"] >= 0
        assert metrics["runtime"] >= 0
    
    def test_performance_tradeoff(self):
        """测试性能权衡"""
        tradeoff = self.inference.analyze_tradeoff()
        assert "energy" in tradeoff
        assert "runtime" in tradeoff
        assert tradeoff["energy"] >= 0
        assert tradeoff["runtime"] >= 0 