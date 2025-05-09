import pytest
import os
import yaml
import torch
import logging
from pathlib import Path
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """加载配置"""
    try:
        # 尝试从多个位置加载模型
        possible_paths = [
            os.path.join(os.getcwd(), "models", "TinyLlama-1.1B-Chat-v1.0"),
            os.path.join(os.getcwd(), "data", "models", "TinyLlama-1.1B-Chat-v1.0"),
            r"\\wsl.localhost\Ubuntu-24.04\home\mpcblock\models\TinyLlama-1.1B-Chat-v1.0",
            "D:\\models\\TinyLlama-1.1B-Chat-v1.0"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"找到模型路径: {model_path}")
                break
                
        if model_path is None:
            raise FileNotFoundError("未找到可用的模型路径")
        
        config = {
            "model_name": "TinyLlama-1.1B-Chat-v1.0",
            "type": "local",
            "path": model_path,
            "tokenizer": "auto",
            "max_length": 2048,
            "batch_size": 1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": "bfloat16",  # 与config.json保持一致
            "precision": "bfloat16",
            "quantization": "none",
            "cache_dir": "data/models",
            "trust_remote_code": True,
            "mixed_precision": "bf16",
            "device_placement": True,
            "use_safetensors": True,
            "local_files_only": True
        }
        
        return config
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}")
        raise

class TestTinyLlama:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """设置和清理测试环境"""
        try:
            self.config = load_config()
            logger.info("配置加载成功")
            
            # 检查CUDA是否可用
            if not torch.cuda.is_available():
                pytest.skip("需要CUDA环境进行测试")
            
            logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
            logger.info(f"当前CUDA设备: {torch.cuda.current_device()}")
            logger.info(f"设备名称: {torch.cuda.get_device_name()}")
            
            # 检查模型文件
            model_dir = Path(self.config["path"])
            model_files = list(model_dir.glob("*.safetensors"))
            if not model_files:
                pytest.skip(f"模型文件不存在: {self.config['path']}")
            logger.info(f"找到模型文件: {[f.name for f in model_files]}")
            
            # 加载分词器
            logger.info(f"开始加载分词器，路径: {self.config['path']}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["path"],
                trust_remote_code=True,
                local_files_only=True
            )
            logger.info("分词器加载成功")
            
            # 加载模型
            logger.info("开始加载模型")
            logger.info(f"模型配置: {self.config}")
            
            # 首先加载配置
            config_path = os.path.join(self.config["path"], "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            logger.info(f"模型配置文件内容: {json.dumps(model_config, indent=2)}")

            # 使用 AutoConfig 创建配置对象
            config = AutoConfig.from_pretrained(
                self.config["path"],
                trust_remote_code=True,
                local_files_only=True
            )
            logger.info(f"自动加载的配置: {config}")

            # 确保配置与模型匹配
            config.torch_dtype = torch.bfloat16
            config.architectures = ["LlamaForCausalLM"]
            config.model_type = "llama"
            config.num_attention_heads = 32
            config.num_key_value_heads = 4
            config.hidden_size = 2048
            config.intermediate_size = 5632
            config.num_hidden_layers = 22
            config.max_position_embeddings = 2048
            config.rms_norm_eps = 1e-6
            config.vocab_size = 32000
            logger.info(f"修改后的配置: {config}")

            # 使用配置加载模型
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config["path"],
                    config=config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
                logger.info("模型加载成功")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                logger.error(f"错误类型: {type(e)}")
                logger.error(f"错误详情: {e.__dict__ if hasattr(e, '__dict__') else 'No details'}")
                raise
            
            # 初始化性能分析器
            self.profiler = RTX4050Profiler()
            logger.info("性能分析器初始化成功")
            
            yield
            
            # 清理资源
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"测试环境设置失败: {str(e)}")
            raise

    def test_basic_inference(self):
        """基本推理测试"""
        prompt = "你好，请介绍一下你自己。"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config["device"])
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\n输入: {prompt}")
        logger.info(f"输出: {response}")
        
        assert len(response) > 0, "生成的响应不应为空"
        assert isinstance(response, str), "响应应该是字符串类型"

    def test_batch_inference(self):
        """批量推理测试"""
        prompts = [
            "解释什么是人工智能。",
            "介绍一下机器学习的基本概念。"
        ]
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.config["device"])
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        for prompt, response in zip(prompts, responses):
            logger.info(f"\n输入: {prompt}")
            logger.info(f"输出: {response}")
            assert len(response) > 0, "生成的响应不应为空"
            assert isinstance(response, str), "响应应该是字符串类型"

    def test_performance(self):
        """性能测试"""
        prompt = "请写一篇关于环境保护的短文。"
        
        def inference_task():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config["device"])
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 测量性能指标
        metrics = self.profiler.measure(inference_task, input_tokens=len(self.tokenizer.encode(prompt)), output_tokens=100)
        
        logger.info("\n性能指标:")
        logger.info(f"- 能耗: {metrics['energy']:.2f} J")
        logger.info(f"- 运行时间: {metrics['runtime']:.2f} s")
        logger.info(f"- 吞吐量: {metrics['throughput']:.2f} tokens/s")
        
        assert metrics["energy"] > 0, "能耗指标异常"
        assert metrics["runtime"] > 0, "运行时间指标异常"
        assert metrics["throughput"] > 0, "吞吐量指标异常"

    def test_model_stress(self):
        """压力测试"""
        prompt = "写一篇关于人工智能的综合论文。"
        
        def stress_task():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config["device"])
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 连续执行多次推理，测试稳定性
        for i in range(3):
            metrics = self.profiler.measure(stress_task, input_tokens=len(self.tokenizer.encode(prompt)), output_tokens=200)
            
            assert metrics["energy"] > 0, f"第{i+1}次压力测试能耗指标异常"
            assert metrics["runtime"] > 0, f"第{i+1}次压力测试运行时间指标异常"
            
            logger.info(f"\n压力测试 #{i+1} 性能指标:")
            logger.info(f"- 能耗: {metrics['energy']:.2f} J")
            logger.info(f"- 运行时间: {metrics['runtime']:.2f} s")
            logger.info(f"- 吞吐量: {metrics['throughput']:.2f} tokens/s")
            
            # 短暂等待，避免GPU过热
            torch.cuda.empty_cache()
            import time
            time.sleep(2) 