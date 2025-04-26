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
        """处理文本并返回包含 token 数据的 DataFrame
        
        Args:
            texts: 输入文本列表
            
        Returns:
            pd.DataFrame: 包含 token 数据的 DataFrame
            
        Raises:
            ValueError: 当输入为 None 时
        """
        if texts is None:
            raise ValueError("输入文本列表不能为 None")
            
        results = []
        for text in texts:
            # 将非字符串类型转换为字符串
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            result = self.processor.process_text(text)
            results.append(result)
        return pd.DataFrame(results)
        
    def compute_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, float]:
        """计算token分布并可选地保存分布图"""
        if df is None:
            raise ValueError("输入数据不能为None")
            
        if df.empty:
            self.logger.warning("输入数据为空")
            return {}
            
        # 检查必要的列
        if 'input_tokens' not in df.columns and 'token' not in df.columns:
            self.logger.warning("输入数据缺少token列")
            return {}
            
        try:
            # 计算token频率分布
            if 'input_tokens' in df.columns:
                # 处理嵌套列表的情况
                all_tokens = []
                for tokens in df['input_tokens']:
                    if isinstance(tokens, list):
                        all_tokens.extend(tokens)
                if not all_tokens:
                    return {}
                distribution = pd.Series(all_tokens).value_counts(normalize=True).to_dict()
            else:
                distribution = df['token'].value_counts(normalize=True).to_dict()
            
            # 如果提供了保存路径，生成并保存分布图
            if save_path:
                try:
                    # 检查路径是否有效
                    if not os.path.isabs(save_path):
                        raise ValueError(f"保存路径必须是绝对路径：{save_path}")
                        
                    # 在Windows上，检查路径是否包含无效字符
                    if os.name == 'nt':
                        invalid_chars = '<>:"|?*'
                        if any(char in save_path for char in invalid_chars):
                            raise ValueError(f"保存路径包含无效字符：{save_path}")
                            
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        raise ValueError(f"保存路径的目录不存在：{save_dir}")
                    if not os.access(save_dir, os.W_OK):
                        raise ValueError(f"没有写入权限：{save_dir}")
                    self._visualize_distribution(distribution, save_path)
                except Exception as e:
                    self.logger.error(f"保存分布图失败: {str(e)}")
                    raise
                    
            return distribution
            
        except Exception as e:
            self.logger.error(f"计算分布失败: {str(e)}")
            raise
            
    def _visualize_distribution(self, distribution: Dict[str, float], save_path: str):
        """可视化 token 分布
        
        Args:
            distribution: token 分布字典
            save_path: 保存图片的路径
            
        Raises:
            ValueError: 当分布为空或路径无效时
        """
        if not distribution:
            raise ValueError("分布数据为空")
            
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
            raise
        
    def get_token_data(self, df: pd.DataFrame, format: str = 'dataframe') -> Union[pd.DataFrame, Dict[str, List]]:
        """获取 token 数据
        
        Args:
            df: 输入 DataFrame
            format: 返回格式，可选 'dataframe' 或 'dict'
            
        Returns:
            Union[pd.DataFrame, Dict[str, List]]: 处理后的 token 数据
            
        Raises:
            ValueError: 当格式无效或输入为 None 时
        """
        if df is None:
            raise ValueError("输入 DataFrame 不能为 None")
            
        if format not in ['dataframe', 'dict']:
            raise ValueError("不支持的格式，只能是 'dataframe' 或 'dict'")
            
        if df.empty:
            self.logger.warning("输入数据为空")
            return pd.DataFrame() if format == 'dataframe' else {}
            
        try:
            # 提取所需列
            required_columns = ['input_tokens', 'decoded_text']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"DataFrame 必须包含以下列：{required_columns}")
                
            token_data = df[required_columns].copy()
            
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
