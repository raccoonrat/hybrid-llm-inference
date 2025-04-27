import pytest
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler

def load_config():
    """加载配置文件"""
    config_path = "configs/models/tinyllama.yaml"
    if not os.path.exists(config_path):
        # 创建默认配置
        config = {
            "models": {
                "tinyllama": {
                    "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "device": "cuda",
                    "dtype": "auto"
                }
            }
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config["models"]["tinyllama"]
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["models"]["tinyllama"]

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