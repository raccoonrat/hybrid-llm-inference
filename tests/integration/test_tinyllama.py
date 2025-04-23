import pytest
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from hardware_profiling.rtx4050_profiler import RTX4050Profiler

def load_config():
    """加载配置文件"""
    config_path = "configs/models/tinyllama.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["models"]["tinyllama"]

def test_model_loading():
    """测试模型加载"""
    config = load_config()
    
    # 加载模型和分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["path"])
        model = AutoModelForCausalLM.from_pretrained(
            config["path"],
            device_map="cuda",
            torch_dtype="auto"
        )
        assert tokenizer is not None
        assert model is not None
    except Exception as e:
        pytest.fail(f"模型加载失败: {str(e)}")

def test_model_inference():
    """测试模型推理"""
    config = load_config()
    
    # 初始化硬件分析器
    profiler_config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    os.environ["TEST_MODE"] = "true"
    profiler = RTX4050Profiler(profiler_config)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        config["path"],
        device_map="cuda",
        torch_dtype="auto"
    )
    
    # 定义推理任务
    def inference_task():
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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