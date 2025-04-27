import torch
import torch.nn as nn
from typing import Union, List

class Linear(nn.Linear):
    """
    线性层实现，继承自torch.nn.Linear
    
    该类在保持原有线性层功能的基础上，添加了文本长度生成功能
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        初始化Linear层
        
        参数:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            bias (bool): 是否使用偏置项
        """
        super().__init__(in_features, out_features, bias)
        
    def generate(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        根据输入文本生成长度值
        
        参数:
            text: 输入文本，可以是单个字符串或字符串列表
            
        返回:
            如果输入是字符串，返回整数（文本长度的2倍）
            如果输入是字符串列表，返回整数列表（每个文本长度的2倍）
            
        异常:
            TypeError: 当输入类型不是字符串或字符串列表时抛出
        """
        if isinstance(text, str):
            return len(text) * 2
        elif isinstance(text, list):
            if not all(isinstance(item, str) for item in text):
                raise TypeError("列表中所有元素必须是字符串类型")
            return [len(item) * 2 for item in text]
        else:
            raise TypeError("输入必须是字符串或字符串列表") 