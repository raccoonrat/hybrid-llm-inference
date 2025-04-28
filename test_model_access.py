import os
import torch
from pathlib import Path

def test_model_access():
    """测试模型文件访问权限"""
    model_path = "D:/Dev/cursor/github.com/hybrid-llm-inference/models/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"测试模型路径: {model_path}")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在: {model_path}")
        return False
        
    # 检查是否是目录
    if not os.path.isdir(model_path):
        print(f"错误：路径不是目录: {model_path}")
        return False
        
    # 检查读取权限
    if not os.access(model_path, os.R_OK):
        print(f"错误：没有读取权限: {model_path}")
        return False
        
    # 列出目录内容
    try:
        print("\n目录内容:")
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"- {item} (文件, {size} 字节)")
            else:
                print(f"- {item} (目录)")
    except Exception as e:
        print(f"错误：无法列出目录内容: {e}")
        return False
        
    print("\n权限检查通过！")
    return True

if __name__ == "__main__":
    test_model_access() 