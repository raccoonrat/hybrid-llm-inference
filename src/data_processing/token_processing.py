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
import re
import time
import psutil

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
        """处理文本并返回包含token的DataFrame

        Args:
            texts: 输入文本列表或Series

        Returns:
            DataFrame包含input_tokens和decoded_text列

        Raises:
            ValueError: 当输入无效时
            MemoryError: 当内存不足时
            RuntimeError: 当处理过程中发生错误时
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        if not texts:
            self.logger.warning("输入数据为空")
            return pd.DataFrame(columns=["input_tokens", "decoded_text"])
            
        # 检查输入大小
        total_size = sum(len(str(text)) for text in texts)
        if total_size > 100 * 1024 * 1024:  # 100MB限制
            raise ValueError("输入数据太大，请分批处理")
            
        try:
            # 将非字符串类型转换为字符串
            texts = [str(text) if text is not None else "" for text in texts]
            
            # 检查编码
            for text in texts:
                try:
                    text.encode('utf-8')
                except UnicodeEncodeError:
                    self.logger.warning(f"发现非UTF-8编码的文本，将尝试使用其他编码")
                    try:
                        text.encode('gbk')
                    except UnicodeEncodeError:
                        raise ValueError("文本包含无法处理的字符编码")
            
            # 批量处理
            tokens = self.processor.batch_process(texts)
            
            # 检查内存使用
            process = psutil.Process()
            if process.memory_info().rss > 1024 * 1024 * 1024:  # 1GB限制
                raise MemoryError("内存使用过高，请减少输入数据量")
            
            # 解码
            decoded_texts = []
            for i, t in enumerate(tokens):
                try:
                    decoded = self.processor.decode(t)
                    decoded_texts.append(decoded)
                except Exception as e:
                    self.logger.error(f"解码第{i}个token时出错: {str(e)}")
                    decoded_texts.append("")
            
            return pd.DataFrame({
                "input_tokens": tokens,
                "decoded_text": decoded_texts
            })
            
        except Exception as e:
            self.logger.error(f"处理token时出错: {str(e)}")
            raise RuntimeError(f"处理token时出错: {str(e)}")
        
    def get_token_data(self, df: pd.DataFrame, format: str = 'dataframe') -> Union[pd.DataFrame, Dict]:
        """从DataFrame中获取token数据

        Args:
            df: 输入DataFrame
            format: 输出格式，'dataframe'或'dict'

        Returns:
            DataFrame或字典格式的token数据

        Raises:
            ValueError: 当输入无效或格式不支持时
            TypeError: 当数据类型转换失败时
            RuntimeError: 当处理过程中发生错误时
        """
        if format not in ['dataframe', 'dict']:
            raise ValueError("不支持的格式，必须是'dataframe'或'dict'")
            
        if df is None or df.empty:
            self.logger.warning("输入DataFrame为空")
            if format == 'dataframe':
                return pd.DataFrame(columns=["input_tokens", "decoded_text"])
            return {}
            
        try:
            # 数据验证
            required_columns = {'input_tokens', 'token'}
            if not any(col in df.columns for col in required_columns):
                raise ValueError("DataFrame中必须包含input_tokens或token列")
                
            # 检查数据类型
            for col in df.columns:
                if col in required_columns:
                    if not all(isinstance(x, (list, str)) or pd.isna(x) for x in df[col]):
                        raise TypeError(f"列 {col} 包含无效的数据类型")
            
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
            
            # 如果没有decoded_text列，添加空值
            if "decoded_text" not in result.columns:
                result["decoded_text"] = ""
            
            # 清理无效数据
            result = result.replace({np.nan: None})
            result = result.fillna("")
            
            # 性能监控
            start_time = time.time()
            
            if format == 'dict':
                output = {
                    "input_tokens": result["input_tokens"].tolist(),
                    "decoded_text": result["decoded_text"].tolist()
                }
            else:
                output = result
                
            # 记录处理时间
            process_time = time.time() - start_time
            self.logger.info(f"处理 {len(df)} 行数据用时: {process_time:.2f}秒")
            
            return output
            
        except Exception as e:
            self.logger.error(f"获取token数据时出错: {str(e)}")
            raise RuntimeError(f"获取token数据时出错: {str(e)}")
            
    def compute_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, float]:
        """计算token分布。

        Args:
            df (pd.DataFrame): 包含token数据的DataFrame
            save_path (Optional[str], optional): 保存分布图的路径. Defaults to None.

        Returns:
            Dict[str, float]: token分布字典

        Raises:
            ValueError: 当输入数据无效或保存路径无效时
            MemoryError: 当内存不足时
            RuntimeError: 当处理过程中发生错误时
        """
        if df is None:
            raise ValueError("输入DataFrame不能为None")

        if df.empty:
            self.logger.warning("输入数据为空")
            return {}

        try:
            # 检查内存使用
            process = psutil.Process()
            if process.memory_info().rss > 1024 * 1024 * 1024:  # 1GB限制
                raise MemoryError("内存使用过高，请减少输入数据量")

            # 确定使用哪一列
            token_column = 'input_tokens' if 'input_tokens' in df.columns else 'token'
            if token_column not in df.columns:
                self.logger.warning(f"DataFrame中缺少所需的列: {token_column}")
                return {}

            # 展平所有token列表并计算频率分布
            all_tokens = []
            total_tokens = 0
            batch_size = 1000  # 批处理大小
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                for tokens in batch[token_column]:
                    if isinstance(tokens, list):
                        all_tokens.extend(map(str, tokens))
                        total_tokens += len(tokens)
                    else:
                        all_tokens.append(str(tokens))
                        total_tokens += 1
                
                # 检查内存使用
                if process.memory_info().rss > 1024 * 1024 * 1024:  # 1GB限制
                    raise MemoryError("内存使用过高，请减少输入数据量")
                
                # 报告进度
                if i % 10000 == 0:
                    self.logger.info(f"已处理 {i}/{len(df)} 行数据")

            if not all_tokens:
                self.logger.warning("没有找到有效的token")
                return {}

            # 计算分布
            distribution = {}
            counter = Counter(all_tokens)
            for token, count in counter.items():
                distribution[token] = count / total_tokens

            # 如果提供了保存路径，则保存分布图
            if save_path:
                try:
                    self._save_distribution_plot(distribution, save_path)
                except ValueError as e:
                    self.logger.error(f"保存分布图时出错: {str(e)}")
                    raise

            return distribution
            
        except Exception as e:
            self.logger.error(f"计算分布时出错: {str(e)}")
            raise RuntimeError(f"计算分布时出错: {str(e)}")
        
    def _save_distribution_plot(self, distribution: Dict[str, float], save_path: str) -> None:
        """保存分布图到指定路径。

        Args:
            distribution (Dict[str, float]): token分布字典
            save_path (str): 保存路径

        Raises:
            ValueError: 当路径无效或没有写入权限时
        """
        if not save_path or save_path.isspace():
            raise ValueError("保存路径不能为空或只包含空格")

        # 标准化路径
        try:
            save_path = os.path.abspath(save_path)
        except Exception as e:
            raise ValueError(f"无效的路径格式: {str(e)}")

        # 检查文件扩展名
        if not save_path.lower().endswith('.png'):
            raise ValueError("文件扩展名必须是.png")

        # 分离目录和文件名
        dir_path = os.path.dirname(save_path)
        file_name = os.path.basename(save_path)

        # 检查文件名是否为空或只包含空格
        if not file_name or file_name.isspace():
            raise ValueError("文件名不能为空或只包含空格")

        # 检查目录是否存在
        if not os.path.exists(dir_path):
            raise ValueError(f"目录不存在: {dir_path}")

        # 检查目录权限
        if not os.access(dir_path, os.W_OK):
            raise ValueError(f"没有目录写入权限: {dir_path}")

        # Windows特定检查
        if os.name == 'nt':
            # 检查保留名称
            reserved_names = {'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
                            'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3',
                            'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'}
            base_name = file_name.split('.')[0].lower()
            if base_name in reserved_names:
                raise ValueError(f"文件名不能使用Windows保留名称: {base_name}")

            # 检查无效字符
            invalid_chars = '<>:"|?*\\'
            if any(char in file_name for char in invalid_chars):
                raise ValueError("文件名包含无效字符")

            # 检查长路径
            if len(save_path) > 260 and not save_path.startswith('\\\\?\\'):
                raise ValueError("路径长度超过Windows限制(260字符)")

            # 检查UNC路径
            if save_path.startswith('\\\\'):
                if not os.path.exists(dir_path):
                    raise ValueError("无法访问网络路径")
                try:
                    # 尝试访问网络路径
                    os.listdir(dir_path)
                except Exception as e:
                    raise ValueError(f"无法访问网络路径: {str(e)}")

        # 检查文件是否已存在且有写入权限
        if os.path.exists(save_path):
            if not os.access(save_path, os.W_OK):
                raise ValueError(f"没有文件写入权限: {save_path}")

        # 生成并保存图表
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(distribution.keys()), y=list(distribution.values()))
            plt.title('Token分布')
            plt.xlabel('Token')
            plt.ylabel('频率')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            raise ValueError(f"保存图表时出错: {str(e)}")
