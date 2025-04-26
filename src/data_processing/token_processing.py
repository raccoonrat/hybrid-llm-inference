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
from collections import Counter
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logging.basicConfig(level=logging.INFO)

class TokenProcessing:
    def __init__(self, model_path: str):
        """初始化TokenProcessing类。

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.processor = TokenProcessor(model_path)
        
    def process_tokens(self, texts: Union[List[str], pd.Series]) -> pd.DataFrame:
        """处理文本并返回包含token的DataFrame"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        if not texts:
            self.logger.warning("输入数据为空")
            return pd.DataFrame(columns=["input_tokens", "decoded_text"])
            
        # 将非字符串类型转换为字符串
        texts = [str(text) if text is not None else "" for text in texts]
            
        tokens = self.processor.batch_process(texts)
        decoded_texts = [self.processor.decode(t) for t in tokens]
        
        return pd.DataFrame({
            "input_tokens": tokens,
            "decoded_text": decoded_texts
        })
        
    def get_token_data(self, df: pd.DataFrame, format: str = 'dataframe') -> Union[pd.DataFrame, Dict]:
        """从DataFrame中获取token数据

        Args:
            df: 输入DataFrame
            format: 输出格式，'dataframe'或'dict'

        Returns:
            DataFrame或字典格式的token数据
        """
        if format not in ['dataframe', 'dict']:
            raise ValueError("不支持的格式，必须是'dataframe'或'dict'")
            
        if df is None or df.empty:
            self.logger.warning("输入DataFrame为空")
            if format == 'dataframe':
                return pd.DataFrame(columns=["input_tokens", "decoded_text"])
            return {}
            
        # 获取必要的列
        result = pd.DataFrame()
        if "input_tokens" in df.columns:
            result["input_tokens"] = df["input_tokens"]
            if "decoded_text" in df.columns:
                result["decoded_text"] = df["decoded_text"]
        elif "token" in df.columns:
            result["input_tokens"] = df["token"]
            if "decoded_text" in df.columns:
                result["decoded_text"] = df["decoded_text"]
        else:
            raise ValueError("DataFrame中必须包含input_tokens或token列")
            
        # 如果没有decoded_text列，添加空值
        if "decoded_text" not in result.columns:
            result["decoded_text"] = ""
            
        if format == 'dict':
            return {
                "input_tokens": result["input_tokens"].tolist(),
                "decoded_text": result["decoded_text"].tolist()
            }
        return result
            
    def compute_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict:
        """计算token分布并可选地保存可视化结果"""
        if df is None:
            raise ValueError("输入DataFrame不能为None")
            
        token_df = self.get_token_data(df)
        if token_df.empty:
            self.logger.warning("没有有效的token数据")
            return {}
            
        # 过滤无效的token数据
        valid_tokens = token_df["input_tokens"].apply(
            lambda x: isinstance(x, list) and all(isinstance(t, (int, float)) for t in x)
        )
        token_df = token_df[valid_tokens]
        
        if token_df.empty:
            self.logger.warning("没有有效的token数据")
            return {}
            
        # 计算token频率并转换为字符串键
        all_tokens = [t for tokens in token_df["input_tokens"] for t in tokens]
        token_counts = pd.Series(all_tokens).value_counts()
        distribution = {str(k): v for k, v in (token_counts / token_counts.sum()).to_dict().items()}
        
        # 如果需要保存可视化结果
        if save_path:
            self._save_distribution_plot(distribution, save_path)
            
        return distribution
        
    def _save_distribution_plot(self, distribution: Dict[str, float], save_path: str) -> None:
        """保存分布图到指定路径。

        Args:
            distribution (Dict[str, float]): token分布字典
            save_path (str): 保存路径

        Raises:
            ValueError: 当路径无效或没有写入权限时
        """
        try:
            # 转换为绝对路径
            abs_path = os.path.abspath(save_path)
            directory = os.path.dirname(abs_path)
            
            # 检查路径有效性
            if not os.path.isabs(abs_path):
                raise ValueError("必须提供绝对路径")
                
            # 检查目录是否存在
            if not os.path.exists(directory):
                raise ValueError(f"目录不存在: {directory}")
                
            # 检查文件名是否包含无效字符
            filename = os.path.basename(abs_path)
            invalid_chars = '<>:"/\\|?*' if os.name == 'nt' else '/'
            if any(char in filename for char in invalid_chars):
                raise ValueError(f"文件名包含无效字符: {filename}")
                
            # 检查目录写入权限
            if not os.access(directory, os.W_OK):
                raise ValueError(f"没有目录的写入权限: {directory}")
                
            # 如果文件已存在，检查是否可写
            if os.path.exists(abs_path) and not os.access(abs_path, os.W_OK):
                raise ValueError(f"没有文件的写入权限: {abs_path}")
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.bar(distribution.keys(), distribution.values())
            plt.title('Token Distribution')
            plt.xlabel('Token')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            
            # 保存图表
            plt.savefig(abs_path)
            plt.close()
            
            self.logger.info(f"分布图已保存到: {abs_path}")
            
        except Exception as e:
            self.logger.error(f"保存分布图时出错: {str(e)}")
            raise ValueError(f"保存分布图失败: {str(e)}")
