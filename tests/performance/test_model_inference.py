import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from toolbox.logger import get_logger

logger = get_logger(__name__)

class TestModelInference:
    """测试模型推理性能"""

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

        # 加载本地TinyLlama模型
        cls.model_path = r"\\wsl.localhost\Ubuntu-24.04\home\mpcblock\models\TinyLlama-1.1B-Chat-v1.0"
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path, local_files_only=True)
            cls.model = AutoModelForCausalLM.from_pretrained(
                cls.model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if cls.device.type == "cuda" else torch.float32,
                device_map="auto" if cls.device.type == "cuda" else None
            )
            logger.info("成功加载TinyLlama模型")
        except Exception as e:
            pytest.skip(f"无法加载模型: {str(e)}")

    def _generate_text(self, prompt):
        """生成文本并返回token数量"""
        messages = [{"role": "user", "content": prompt}]
        chat_format = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(chat_format, return_tensors="pt")
        if self.device.type == "cuda":
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
        prompts = [
            "今天天气真好",
            "人工智能的发展",
            "机器学习模型训练"
        ]
        
        def inference_task():
            total_input_tokens = 0
            total_output_tokens = 0
            for prompt in prompts:
                input_tokens, output_tokens = self._generate_text(prompt)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            return total_input_tokens, total_output_tokens
        
        # 先运行一次获取token数量
        input_tokens, output_tokens = inference_task()
        # 使用实际的token数量运行测试
        metrics = self.profiler.measure(inference_task, input_tokens, output_tokens)
        print(f"推理性能指标: {metrics}")
        
        assert metrics["runtime"] > 0
        assert metrics["energy"] > 0
        assert metrics["throughput"] > 0

    def test_memory_usage(self):
        """测试内存使用"""
        if self.device.type != "cuda":
            pytest.skip("需要 CUDA 设备进行内存测试")
            
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        prompt = "测试模型内存使用情况"
        input_tokens, output_tokens = self._generate_text(prompt)
            
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        print(f"初始内存: {initial_memory / 1024**2:.2f} MB")
        print(f"峰值内存: {peak_memory / 1024**2:.2f} MB")
        print(f"当前内存: {current_memory / 1024**2:.2f} MB")
        
        assert peak_memory > initial_memory
        assert current_memory >= initial_memory

    def test_precision_impact(self):
        """测试不同精度的影响"""
        if self.device.type != "cuda":
            pytest.skip("需要 CUDA 设备进行精度测试")
            
        prompt = "测试不同精度对推理的影响"
        
        def fp16_task():
            return self._generate_text(prompt)
            
        def fp32_task():
            model_fp32 = self.model.to(torch.float32)
            try:
                return self._generate_text(prompt)
            finally:
                self.model = model_fp32.to(torch.float16)  # 恢复FP16
        
        # 先运行一次获取token数量
        input_tokens, output_tokens = fp16_task()
        
        # 测试 FP16 (默认)
        metrics_fp16 = self.profiler.measure(fp16_task, input_tokens, output_tokens)
        
        # 测试 FP32
        metrics_fp32 = self.profiler.measure(fp32_task, input_tokens, output_tokens)
        
        print(f"FP16 性能指标: {metrics_fp16}")
        print(f"FP32 性能指标: {metrics_fp32}")
        
        # 验证 FP16 推理速度更快
        assert metrics_fp16["runtime"] <= metrics_fp32["runtime"] * 1.1  # 允许 10% 的误差

    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        if hasattr(cls, 'profiler'):
            del cls.profiler
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 