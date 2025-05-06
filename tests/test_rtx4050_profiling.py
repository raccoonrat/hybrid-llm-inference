from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
from hardware_profiling.rtx4050_profiler import RTX4050Profiler
import pynvml
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_profiling():
    # 使用Windows下的WSL模型路径
    model_path = r"D:\Dev\cursor\github.com\hybrid-llm-inference\models\TinyLlama-1.1B-Chat-v1.0"
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"错误：模型路径不存在: {model_path}")
        return
    
    logger.info(f"使用模型路径: {model_path}")
    
    # 初始化RTX 4050分析器
    profiler_config = {
        "device_id": 0,
        "idle_power": 15.0,
        "sample_interval": 200
    }
    profiler = RTX4050Profiler(profiler_config)
    
    # 初始化tokenizer和模型
    logger.info("正在加载tokenizer和模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return
    
    # 定义测试输入
    test_inputs = [
        "你好，请介绍一下你自己。",
        "请用中文解释一下什么是人工智能。",
        "写一个简单的Python函数来计算斐波那契数列。"
    ]
    
    # 进行性能分析
    logger.info("\n开始性能分析测试...")
    total_energy = 0.0
    total_runtime = 0.0
    total_tokens = 0
    
    for i, test_input in enumerate(test_inputs, 1):
        logger.info(f"\n测试用例 {i}: {test_input}")
        
        # 测量推理过程
        def inference_task():
            try:
                inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                logger.error(f"推理失败: {str(e)}")
                return None
        
        # 使用分析器测量
        try:
            metrics = profiler.measure(
                inference_task,
                input_tokens=len(tokenizer.encode(test_input)),
                output_tokens=100  # 预估输出token数
            )
            
            # 更新总指标
            total_energy += metrics["energy"]
            total_runtime += metrics["runtime"]
            total_tokens += len(tokenizer.encode(test_input)) + 100
            
            # 打印结果
            logger.info(f"能耗: {metrics['energy']:.2f} J")
            logger.info(f"运行时间: {metrics['runtime']:.2f} s")
            logger.info(f"吞吐量: {metrics['throughput']:.2f} tokens/s")
            logger.info(f"每token能耗: {metrics['energy_per_token']:.2f} J/token")
            
            # 打印模型输出
            if metrics.get("result"):
                logger.info(f"模型输出: {metrics['result']}")
            
        except Exception as e:
            logger.error(f"测量失败: {str(e)}")
            continue
    
    # 打印总体性能指标
    if total_runtime > 0:
        logger.info("\n总体性能指标:")
        logger.info(f"总能耗: {total_energy:.2f} J")
        logger.info(f"总运行时间: {total_runtime:.2f} s")
        logger.info(f"平均吞吐量: {total_tokens/total_runtime:.2f} tokens/s")
        logger.info(f"平均每token能耗: {total_energy/total_tokens:.2f} J/token")

if __name__ == "__main__":
    test_model_profiling() 