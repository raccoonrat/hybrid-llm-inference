from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def test_model_load():
    # 使用Windows下的WSL模型路径
    model_path = r"\\wsl.localhost\Ubuntu-24.04\home\mpcblock\models\TinyLlama-1.1B-Chat-v1.0"
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在: {model_path}")
        return
    
    print(f"使用模型路径: {model_path}")
    
    # 初始化tokenizer
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 使用float16加载模型
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 简单的推理测试
    test_input = "你好，请介绍一下你自己。"
    print(f"\n测试输入: {test_input}")
    
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n模型输出: {response}")
    
    print("\n模型加载测试完成！")

if __name__ == "__main__":
    test_model_load() 