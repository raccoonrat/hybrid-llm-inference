import pytest
import torch
import os
from model_zoo import get_model
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from toolbox.logger import get_logger
import time

logger = get_logger(__name__)

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

class TestModelInference:
    """测试模型推理性能"""

    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 设置测试模式环境变量
        os.environ['TEST_MODE'] = '1'
        
        cls.model_config = {
            "model_name": "tinyllama",
            "model_path": "models/TinyLlama-1.1B-Chat-v1.0",
            "mode": "local",
            "batch_size": 1,
            "max_length": 128
        }
        
        # 获取模型实例
        cls.model = get_model(
            model_name=cls.model_config["model_name"],
            mode=cls.model_config["mode"],
            config=cls.model_config
        )
        
        # 确保模型在CPU上运行
        if hasattr(cls.model, 'model'):
            cls.model.model = cls.model.model.to('cpu')
        
        # 获取tokenizer
        if hasattr(cls.model, 'tokenizer'):
            cls.tokenizer = cls.model.tokenizer
        else:
            raise AttributeError("模型实例没有tokenizer属性")

        cls.config = {
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        }
        cls.profiler = RTX4050Profiler(cls.config)
        cls.device = torch.device("cpu")  # 强制使用CPU

    def _generate_text(self, prompt):
        """生成文本并返回token数量"""
        messages = [{"role": "user", "content": prompt}]
        chat_format = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(chat_format, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输入: {prompt}")
        print(f"输出: {decoded}\n")
        
        return len(inputs["input_ids"][0]), len(outputs[0])

    def test_inference_speed(self):
        """测试推理速度"""
        prompt = "Hello, how are you?"
        start_time = time.time()
        
        response = self.model.infer(prompt)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert inference_time > 0

    def test_memory_usage(self):
        """测试内存使用"""
        prompt = "Hello, how are you?"
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        response = self.model.infer(prompt)
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_usage = final_memory - initial_memory
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert memory_usage >= 0

    def test_precision_impact(self):
        """测试精度影响"""
        prompt = "Hello, how are you?"
        response = self.model.infer(prompt)
        
        assert isinstance(response, str)
        assert len(response) > 0

    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        if hasattr(cls, 'profiler'):
            del cls.profiler
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 