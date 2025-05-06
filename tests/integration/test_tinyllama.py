import pytest
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler

def load_config():
    return {
        "model_name": "TinyLlama-1.1B-Chat-v1.0",
        "type": "local",
        "path": r"\\wsl.localhost\Ubuntu-24.04\home\mpcblock\models\TinyLlama-1.1B-Chat-v1.0",
        "tokenizer": "auto",
        "max_length": 2048,
        "batch_size": 1,
        "device": "cuda",
        "dtype": "float16",
        "precision": "float16",
        "quantization": "none",
        "cache_dir": "data/models",
        "trust_remote_code": True,
        "mixed_precision": "fp16",
        "device_placement": True
    }

def test_model_loading():
    """测试模型加载"""
    config = load_config()
    
    # 设置测试模式
    os.environ["TEST_MODE"] = "true"
    
    try:
        # 加载模型和分词器
        try:
            tokenizer = AutoTokenizer.from_pretrained(config["path"])
            model = AutoModelForCausalLM.from_pretrained(
                config["path"],
                device_map=config["device"],
                torch_dtype=config["dtype"]
            )
            assert tokenizer is not None
            assert model is not None
        except Exception as e:
            pytest.fail(f"模型加载失败: {str(e)}")
    finally:
        if "TEST_MODE" in os.environ:
            del os.environ["TEST_MODE"]

def test_model_inference():
    """测试模型推理"""
    config = load_config()
    
    # 设置测试模式
    os.environ["TEST_MODE"] = "true"
    
    try:
        # 初始化硬件分析器
        profiler_config = {
            "device_id": 0,
            "idle_power": 15.0,
            "sample_interval": 200
        }
        profiler = RTX4050Profiler(profiler_config)
        
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(config["path"])
        model = AutoModelForCausalLM.from_pretrained(
            config["path"],
            device_map=config["device"],
            torch_dtype=config["dtype"]
        )
        
        # 定义推理任务
        def inference_task():
            prompt = "Hello, how are you?"
            inputs = tokenizer(prompt, return_tensors="pt").to(config["device"])
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        
        # 测量推理性能
        metrics = profiler.measure(inference_task, input_tokens=5, output_tokens=20)
        
        # 验证指标
        assert metrics["energy"] > 0
        assert metrics["runtime"] > 0
        assert metrics["throughput"] > 0
        assert metrics["energy_per_token"] > 0
        assert isinstance(metrics["result"], str)
        assert len(metrics["result"]) > 0
        
        # 清理资源
        profiler.cleanup()
    finally:
        if "TEST_MODE" in os.environ:
            del os.environ["TEST_MODE"] 