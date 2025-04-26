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
            
        tokens = self.processor.batch_process(texts)
        decoded_texts = [self.processor.decode(t) for t in tokens]
        
        return pd.DataFrame({
            "input_tokens": tokens,
            "decoded_text": decoded_texts
        })
        
    def get_token_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """从DataFrame中获取token数据"""
        if df is None or df.empty:
            self.logger.warning("输入DataFrame为空")
            return pd.DataFrame(columns=["input_tokens"])
            
        if "input_tokens" in df.columns:
            return df[["input_tokens"]].copy()
        elif "token" in df.columns:
            return df[["token"]].rename(columns={"token": "input_tokens"})
        else:
            raise ValueError("DataFrame中必须包含input_tokens或token列")
            
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
            
        # 计算token频率
        all_tokens = [t for tokens in token_df["input_tokens"] for t in tokens]
        token_counts = pd.Series(all_tokens).value_counts()
        distribution = (token_counts / token_counts.sum()).to_dict()
        
        # 如果需要保存可视化结果
        if save_path:
            self._save_distribution_plot(distribution, save_path)
            
        return distribution
        
    def _save_distribution_plot(self, distribution: Dict, save_path: str):
        """保存token分布图"""
        try:
            # 检查路径是否有效
            save_path = os.path.abspath(save_path)
            if not os.path.isabs(save_path):
                raise ValueError("保存路径必须是绝对路径")
                
            # 检查目录是否存在且有写入权限
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                raise ValueError(f"目录不存在: {save_dir}")
                
            if not os.access(save_dir, os.W_OK):
                raise ValueError(f"没有写入权限: {save_dir}")
                
            # 检查文件名是否有效
            if any(c in os.path.basename(save_path) for c in '<>:"|?*'):
                raise ValueError("文件名包含无效字符")
                
            # 创建可视化
            plt.figure(figsize=(10, 6))
            plt.bar(distribution.keys(), distribution.values())
            plt.xlabel("Token")
            plt.ylabel("频率")
            plt.title("Token分布")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"保存分布图时出错: {str(e)}")
            raise
