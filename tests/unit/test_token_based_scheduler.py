import pytest
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from toolbox.logger import get_logger

logger = get_logger(__name__)

class TestTokenBasedScheduler:
    """测试基于token的调度器"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.scheduler_config = {
            "hardware_map": {
                "m1_pro": "apple_m1_pro",
                "a100": "nvidia_a100",
                "rtx4050": "nvidia_rtx4050",
                "a800": "nvidia_a800"
            }
        }
        
    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        # 测试有效阈值
        thresholds = {"T_in": 32, "T_out": 32}
        scheduler = TokenBasedScheduler(thresholds, self.scheduler_config)
        assert scheduler.input_threshold == 32
        assert scheduler.output_threshold == 32
        
        # 测试无效阈值
        with pytest.raises(ValueError, match="Thresholds must be positive"):
            TokenBasedScheduler({"T_in": -1, "T_out": 32}, self.scheduler_config)
            
        with pytest.raises(ValueError, match="Thresholds must include T_in and T_out"):
            TokenBasedScheduler({"T_in": 32}, self.scheduler_config)
            
    def test_task_scheduling(self):
        """测试任务调度"""
        thresholds = {"T_in": 32, "T_out": 32}
        scheduler = TokenBasedScheduler(thresholds, self.scheduler_config)
        
        # 测试小任务（应该分配给RTX4050）
        small_task = {
            "prompt": "Short prompt",
            "response": "Short response",
            "input_tokens": 10,
            "output_tokens": 15
        }
        
        # 测试中等任务（应该分配给M1 Pro）
        medium_task = {
            "prompt": "Medium prompt",
            "response": "Medium response",
            "input_tokens": 25,
            "output_tokens": 30
        }
        
        # 测试大任务（应该分配给A100/A800）
        large_task = {
            "prompt": "Large prompt",
            "response": "Large response",
            "input_tokens": 50,
            "output_tokens": 60
        }
        
        allocations = scheduler.schedule([small_task, medium_task, large_task])
        
        assert len(allocations) == 3
        assert allocations[0]["hardware"] == "nvidia_rtx4050"  # 小任务
        assert allocations[1]["hardware"] == "apple_m1_pro"    # 中等任务
        assert allocations[2]["hardware"] in ["nvidia_a100", "nvidia_a800"]  # 大任务
        
    def test_empty_task_list(self):
        """测试空任务列表"""
        thresholds = {"T_in": 32, "T_out": 32}
        scheduler = TokenBasedScheduler(thresholds, self.scheduler_config)
        
        allocations = scheduler.schedule([])
        assert len(allocations) == 0
        
    def test_invalid_tasks(self):
        """测试无效任务"""
        thresholds = {"T_in": 32, "T_out": 32}
        scheduler = TokenBasedScheduler(thresholds, self.scheduler_config)
        
        # 测试缺少token计数的任务
        invalid_task = {
            "prompt": "Test prompt",
            "response": "Test response"
        }
        
        allocations = scheduler.schedule([invalid_task])
        assert len(allocations) == 0
        
    def test_edge_cases(self):
        """测试边界情况"""
        thresholds = {"T_in": 32, "T_out": 32}
        scheduler = TokenBasedScheduler(thresholds, self.scheduler_config)
        
        # 测试恰好等于阈值的任务
        edge_task = {
            "prompt": "Edge case",
            "response": "Edge response",
            "input_tokens": 32,
            "output_tokens": 32
        }
        
        allocations = scheduler.schedule([edge_task])
        assert len(allocations) == 1
        assert allocations[0]["hardware"] == "apple_m1_pro"  # 应该分配给M1 Pro
        
    def test_performance_metrics(self):
        """测试性能指标"""
        thresholds = {"T_in": 32, "T_out": 32}
        scheduler = TokenBasedScheduler(thresholds, self.scheduler_config)
        
        # 创建多个任务
        tasks = []
        for i in range(100):
            input_tokens = i % 50 + 1  # 1-50
            output_tokens = i % 40 + 1  # 1-40
            tasks.append({
                "prompt": f"Task {i}",
                "response": f"Response {i}",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        
        # 测量调度时间
        import time
        start_time = time.time()
        allocations = scheduler.schedule(tasks)
        end_time = time.time()
        
        # 验证调度时间
        assert end_time - start_time < 1.0  # 100个任务的调度时间应小于1秒
        
        # 验证任务分配比例
        hardware_counts = {}
        for alloc in allocations:
            hardware = alloc["hardware"]
            hardware_counts[hardware] = hardware_counts.get(hardware, 0) + 1
            
        total_tasks = len(allocations)
        for hardware, count in hardware_counts.items():
            ratio = count / total_tasks
            logger.info(f"{hardware}任务比例: {ratio:.2%}")
            
        # 验证小任务主要分配给RTX4050
        assert hardware_counts.get("nvidia_rtx4050", 0) > 0
        # 验证中等任务主要分配给M1 Pro
        assert hardware_counts.get("apple_m1_pro", 0) > 0
        # 验证大任务主要分配给A100/A800
        assert (hardware_counts.get("nvidia_a100", 0) + 
                hardware_counts.get("nvidia_a800", 0)) > 0 