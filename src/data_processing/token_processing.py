# hybrid-llm-inference/src/data_processing/token_processing.py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from toolbox.logger import get_logger
from model_zoo import get_model
import os
import numpy as np
from typing import Dict, List, Optional, Union
from .token_processor import TokenProcessor
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logging.basicConfig(level=logging.INFO)

class TokenProcessing:
    def __init__(self, model_path: str):
        self.processor = TokenProcessor(model_path)
        self.logger = logging.getLogger(__name__)
        
    def process_tokens(self, texts: List[str]) -> pd.DataFrame:
        """处理文本并返回包含 token 数据的 DataFrame"""
        results = []
        for text in texts:
            result = self.processor.process_text(text)
            results.append(result)
        return pd.DataFrame(results)
        
    def compute_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, float]:
        """计算 token 分布
        
        Args:
            df: 包含 token 数据的 DataFrame
            save_path: 保存分布图的路径（可选）
            
        Returns:
            Dict[str, float]: token 分布字典，key 为 token，value 为频率
        """
        if df.empty or 'input_tokens' not in df.columns:
            self.logger.warning("输入数据为空或缺少 input_tokens 列")
            return {}
            
        try:
            # 收集所有 token
            all_tokens = []
            for tokens in df['input_tokens']:
                all_tokens.extend(tokens)
                
            # 计算唯一 token 及其频率
            unique_tokens, counts = np.unique(all_tokens, return_counts=True)
            total = sum(counts)
            
            # 计算频率分布
            distribution = {
                str(token): count/total 
                for token, count in zip(unique_tokens, counts)
            }
            
            # 如果提供了保存路径，生成并保存分布图
            if save_path:
                self._visualize_distribution(distribution, save_path)
                
            return distribution
            
        except Exception as e:
            self.logger.error(f"计算 token 分布失败: {str(e)}")
            return {}
            
    def _visualize_distribution(self, distribution: Dict[str, float], save_path: str):
        """可视化 token 分布
        
        Args:
            distribution: token 分布字典
            save_path: 保存图片的路径
        """
        try:
            # 准备数据
            tokens = list(distribution.keys())
            frequencies = list(distribution.values())
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(tokens)), frequencies)
            plt.xlabel('Token')
            plt.ylabel('频率')
            plt.title('Token 分布')
            
            # 设置 x 轴标签
            plt.xticks(range(len(tokens)), tokens, rotation=45)
            
            # 保存图片
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"已保存分布图到: {save_path}")
            
        except Exception as e:
            self.logger.error(f"生成分布图失败: {str(e)}")
        
    def get_token_data(self, df: pd.DataFrame, format: str = 'dataframe') -> Union[pd.DataFrame, Dict[str, List]]:
        """获取 token 数据
        
        Args:
            df: 输入 DataFrame
            format: 返回格式，可选 'dataframe' 或 'dict'
            
        Returns:
            Union[pd.DataFrame, Dict[str, List]]: 处理后的 token 数据
        """
        if df.empty:
            self.logger.warning("输入数据为空")
            return pd.DataFrame() if format == 'dataframe' else {}
            
        try:
            # 提取所需列
            token_data = df[['input_tokens', 'decoded_text']].copy()
            
            # 根据请求的格式返回数据
            if format == 'dict':
                return {
                    'input_tokens': token_data['input_tokens'].tolist(),
                    'decoded_text': token_data['decoded_text'].tolist()
                }
            else:
                return token_data
                
        except Exception as e:
            self.logger.error(f"获取 token 数据失败: {str(e)}")
            return pd.DataFrame() if format == 'dataframe' else {}
