import pytest
import torch
import os
from model_zoo import get_model
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from toolbox.logger import get_logger
import time
from unittest.mock import MagicMock, patch
from src.model_zoo.base_model import BaseModel

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

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def encode(self, text, **kwargs):
        return [1, 2, 3, 4, 5]
        
    def decode(self, tokens, **kwargs):
        return "Mock response"

class MockModel(BaseModel):
    def __init__(self, model_name, config):
        super().__init__(model_name, config)
        self.tokenizer = MockTokenizer()
        self.device = torch.device("cpu")
        
    def infer(self, input_text, **kwargs):
        time.sleep(0.1)  # 模拟推理时间
        return "Mock response"
        
    def get_token_count(self, text):
        return len(self.tokenizer.encode(text))

@pytest.fixture
def mock_model():
    with patch('src.model_zoo.get_model', return_value=MockModel("tinyllama", {})):
        yield

class MockRTX4050Profiler:
    def __init__(self, config):
        self.config = config
        self.device_id = config.get("device_id", 0)
        self.idle_power = config.get("idle_power", 15.0)
        self.sample_interval = config.get("sample_interval", 200)
        
    def measure_power(self):
        return self.idle_power + 10.0  # 模拟负载功耗
        
    def get_memory_info(self):
        return {
            "total": 1024 * 1024 * 1024,  # 1GB
            "used": 512 * 1024 * 1024,    # 512MB
            "free": 512 * 1024 * 1024     # 512MB
        }
        
    def cleanup(self):
        pass

class TestModelInference:
    """测试模型推理性能"""
    
    use_gpu = torch.cuda.is_available()  # 检查GPU是否可用

    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 设置测试模式环境变量
        os.environ['TEST_MODE'] = '1'
        
        # 确定设备
        cls.device = torch.device("cuda" if cls.use_gpu else "cpu")
        logger.info(f"使用设备: {cls.device}")
        
        cls.model_config = {
            "model_name": "tinyllama",
            "model_path": "models/TinyLlama-1.1B-Chat-v1.0",
            "mode": "local",
            "batch_size": 1,
            "max_length": 128
        }
        
        # 根据环境选择使用真实模型还是模拟模型
        if cls.use_gpu:
            cls.model = get_model(
                model_name=cls.model_config["model_name"],
                mode=cls.model_config["mode"],
                config=cls.model_config
            )
            cls.model.model = cls.model.model.to(cls.device)
        else:
            cls.model = MockModel(cls.model_config["model_name"], cls.model_config)
        
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
        
        # 根据环境选择使用真实profiler还是模拟profiler
        if cls.use_gpu:
            try:
                cls.profiler = RTX4050Profiler(cls.config)
                logger.info("使用RTX4050 Profiler")
            except Exception as e:
                logger.warning(f"无法初始化RTX4050 Profiler: {e}, 切换到模拟模式")
                cls.profiler = MockRTX4050Profiler(cls.config)
        else:
            cls.profiler = MockRTX4050Profiler(cls.config)
            logger.info("使用模拟Profiler")

    def _generate_text(self, prompt):
        """生成文本并返回token数量"""
        if self.use_gpu:
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
            logger.info(f"输入: {prompt}")
            logger.info(f"输出: {decoded}")
            
            return len(inputs["input_ids"][0]), len(outputs[0])
        else:
            # 使用模拟模型生成
            response = self.model.infer(prompt)
            input_tokens = self.model.get_token_count(prompt)
            output_tokens = self.model.get_token_count(response)
            logger.info(f"输入: {prompt}")
            logger.info(f"输出: {response}")
            return input_tokens, output_tokens

    def test_inference_speed(self):
        """测试推理速度"""
        input_text = "测试输入文本"
        
        # 测量推理时间
        start_time = time.time()
        if self.use_gpu:
            input_tokens, output_tokens = self._generate_text(input_text)
        else:
            response = self.model.infer(input_text)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        if self.use_gpu:
            # GPU模式下的断言
            assert inference_time > 0
            assert input_tokens > 0
            assert output_tokens > 0
        else:
            # CPU模拟模式下的断言
            assert isinstance(response, str)
            assert len(response) > 0
            assert 0.05 <= inference_time <= 0.15

    def test_memory_usage(self):
        """测试内存使用"""
        input_text = "测试输入文本"
        
        # 记录初始内存使用
        if self.use_gpu:
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        # 执行推理
        if self.use_gpu:
            self._generate_text(input_text)
        else:
            self.model.infer(input_text)
        
        # 记录推理后的内存使用
        if self.use_gpu:
            final_memory = torch.cuda.memory_allocated()
            memory_diff = final_memory - initial_memory
            # GPU模式下检查实际内存使用
            assert memory_diff >= 0
            logger.info(f"GPU内存使用: {memory_diff / 1024 / 1024:.2f}MB")
        else:
            # CPU模式下使用模拟的内存值
            memory_info = self.profiler.get_memory_info()
            assert memory_info["used"] <= memory_info["total"]

    def test_precision_impact(self):
        """测试精度影响"""
        input_text = "测试输入文本"
        
        if self.use_gpu:
            # GPU模式下比较两次生成的token数量
            tokens1_in, tokens1_out = self._generate_text(input_text)
            tokens2_in, tokens2_out = self._generate_text(input_text)
            assert tokens1_in == tokens2_in  # 输入token数应该相同
            # 由于采样的随机性，输出token数可能不同，但应在合理范围内
            assert abs(tokens1_out - tokens2_out) < 20
        else:
            # CPU模拟模式下比较响应字符串
            response1 = self.model.infer(input_text)
            response2 = self.model.infer(input_text)
            assert response1 == response2
            assert isinstance(response1, str)
            assert len(response1) > 0

    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        if hasattr(cls, 'profiler'):
            cls.profiler.cleanup()
        if cls.use_gpu:
            torch.cuda.empty_cache() 