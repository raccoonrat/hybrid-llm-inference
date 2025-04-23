import pytest
import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hardware_profiling.rtx4050_profiler import RTX4050Profiler
from scheduling.task_allocator import TaskAllocator

def load_configs():
    """加载配置文件"""
    model_config_path = "configs/models/tinyllama.yaml"
    hardware_config_path = "configs/hardware.yaml"
    
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    with open(hardware_config_path, "r") as f:
        hardware_config = yaml.safe_load(f)
        
    return model_config, hardware_config

class TestTokenScheduling:
    @pytest.fixture(autouse=True)
    def setup(self):
        """设置测试环境"""
        self.model_config, self.hardware_config = load_configs()
        
        # 设置测试模式
        os.environ["TEST_MODE"] = "true"
        
        # 加载本地模型
        model_path = self.model_config["models"]["tinyllama"]["path"]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                torch_dtype=torch.float16,
                local_files_only=True
            )
        except Exception as e:
            pytest.skip(f"无法加载本地模型: {str(e)}")
        
        # 初始化任务分配器
        self.allocator = TaskAllocator(
            hardware_config=self.hardware_config['hardware'],
            model_config=self.model_config
        )
        
        yield
        # 清理资源
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
    
    def generate_task(self, prompt, max_new_tokens=20):
        """生成推理任务"""
        def task():
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return task
    
    def test_token_threshold_impact(self):
        """测试不同token阈值对调度决策的影响"""
        prompts = [
            "Hello",  # 短文本
            "What is the meaning of life?",  # 中等文本
            "Please write a detailed explanation of quantum computing and its applications in modern technology." # 长文本
        ]
        
        thresholds = [64, 128, 256]  # 不同的token阈值
        
        for prompt in prompts:
            input_tokens = len(self.tokenizer.encode(prompt))
            for threshold in thresholds:
                # 获取调度决策
                device = self.allocator.allocate_task(prompt, "", threshold)
                
                # 创建任务
                task = self.generate_task(prompt)
                
                # 使用对应设备的分析器测量性能
                profiler = RTX4050Profiler(self.hardware_config["hardware"]["rtx4050"])
                metrics = profiler.measure(task, input_tokens=input_tokens, output_tokens=20)
                
                # 验证结果
                assert metrics["energy"] > 0
                assert metrics["runtime"] > 0
                assert metrics["throughput"] > 0
                print(f"Prompt length: {input_tokens}, Threshold: {threshold}, Device: {device}")
                print(f"Energy: {metrics['energy']:.2f}J, Runtime: {metrics['runtime']:.2f}s")
    
    def test_dynamic_threshold_adjustment(self):
        """测试动态阈值调整"""
        prompt = "What is the capital of France?"
        input_tokens = len(self.tokenizer.encode(prompt))
        
        # 执行多次任务，观察阈值变化
        for _ in range(10):
            device = self.allocator.allocate_task(prompt, "", 128)
            task = self.generate_task(prompt)
            
            profiler = RTX4050Profiler(self.hardware_config["hardware"]["rtx4050"])
            metrics = profiler.measure(task, input_tokens=input_tokens, output_tokens=20)
            
            # 更新阈值
            self.allocator.update_threshold(metrics["throughput"])
            
            # 验证阈值在合理范围内
            assert self.allocator.dynamic_threshold >= self.allocator.min_threshold
            assert self.allocator.dynamic_threshold <= self.allocator.max_threshold
            print(f"Iteration {_}, Dynamic Threshold: {self.allocator.dynamic_threshold}")
    
    def test_batch_processing(self):
        """测试批处理场景"""
        prompts = [
            "What is Python?",
            "Explain machine learning.",
            "How does a computer work?",
            "What is the Internet?"
        ]
        
        # 创建批量任务
        allocations = []
        for prompt in prompts:
            input_tokens = len(self.tokenizer.encode(prompt))
            device = self.allocator.allocate_task(prompt, "", 128)
            allocations.append({
                "query": {
                    "prompt": prompt,
                    "input_tokens": input_tokens,
                    "output_tokens": 20
                },
                "hardware": device
            })
        
        # 执行批量任务
        results = self.allocator.allocate(allocations)
        
        # 验证结果
        assert len(results) == len(prompts)
        for result in results:
            assert "metrics" in result
            assert result["metrics"]["energy"] > 0
            assert result["metrics"]["runtime"] > 0
            print(f"Task on {result['hardware']}: Energy={result['metrics']['energy']:.2f}J") 