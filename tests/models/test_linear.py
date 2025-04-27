import pytest
import torch
import torch.nn as nn
from src.models.linear import Linear

def test_linear_initialization():
    """测试Linear类的初始化"""
    model = Linear(10, 20, bias=True)
    assert isinstance(model, nn.Linear)
    assert model.in_features == 10
    assert model.out_features == 20
    assert model.bias is not None

def test_generate_single_string():
    """测试generate方法处理单个字符串的情况"""
    model = Linear(10, 20)
    input_str = "测试文本"
    result = model.generate(input_str)
    assert isinstance(result, int)
    assert result == len(input_str) * 2

def test_generate_string_list():
    """测试generate方法处理字符串列表的情况"""
    model = Linear(10, 20)
    input_list = ["测试1", "测试文本2", "测试3"]
    result = model.generate(input_list)
    assert isinstance(result, list)
    assert len(result) == len(input_list)
    for i, s in enumerate(input_list):
        assert result[i] == len(s) * 2

def test_generate_invalid_input():
    """测试generate方法对无效输入的处理"""
    model = Linear(10, 20)
    with pytest.raises(TypeError):
        model.generate(123)  # 数字类型
    with pytest.raises(TypeError):
        model.generate([1, 2, 3])  # 数字列表
    with pytest.raises(TypeError):
        model.generate(None)  # None类型 