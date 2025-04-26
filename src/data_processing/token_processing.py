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
import tempfile

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logging.basicConfig(level=logging.INFO)

class DataProcessor:
    """数据处理基类。"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_data(self, data: Union[pd.DataFrame, List[Dict]]) -> bool:
        """验证数据格式。

        Args:
            data: 输入数据

        Returns:
            bool: 数据是否有效
        """
        if isinstance(data, pd.DataFrame):
            return not data.empty
        elif isinstance(data, list):
            return len(data) > 0
        return False

    def process_batch(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """处理数据批次。

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 处理后的数据
        """
        if not self.validate_data(data):
            raise ValueError("无效的输入数据")
        return pd.DataFrame(data)

class AlpacaLoader:
    """Alpaca数据集加载器。"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)

    def validate_entry(self, entry: Dict) -> bool:
        """验证数据条目。

        Args:
            entry: 数据条目

        Returns:
            bool: 条目是否有效
        """
        required_fields = ['instruction', 'input', 'output']
        return all(field in entry for field in required_fields)

    def get_statistics(self) -> Dict:
        """获取数据集统计信息。

        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_entries': 0,
            'valid_entries': 0,
            'invalid_entries': 0
        }
        return stats

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
                    if tokens is None or (isinstance(tokens, list) and not tokens):
                        continue
                    if isinstance(tokens, list):
                        valid_tokens = [str(t) for t in tokens if t is not None and str(t).strip()]
                        all_tokens.extend(valid_tokens)
                        total_tokens += len(valid_tokens)
                    else:
                        token_str = str(tokens).strip()
                        if token_str:
                            all_tokens.append(token_str)
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
                except Exception as e:
                    self.logger.error(f"保存分布图时出错: {str(e)}")
                    raise ValueError(f"保存分布图时出错: {str(e)}")

            return distribution

        except MemoryError as e:
            self.logger.error(f"内存不足: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"无效的输入或路径: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"计算分布时出错: {str(e)}")
            raise RuntimeError(f"计算分布时出错: {str(e)}")
        
    def _save_distribution_plot(self, distribution: Dict[str, float], save_path: str) -> None:
        """保存分布图到指定路径。

        Args:
            distribution: token分布字典
            save_path: 保存路径

        Raises:
            ValueError: 当路径无效或没有写入权限时
        """
        if not save_path or not isinstance(save_path, str):
            raise ValueError("保存路径不能为空")

        # 规范化路径
        save_path = os.path.abspath(save_path)
        save_dir = os.path.dirname(save_path)
        filename = os.path.basename(save_path)

        # 检查文件扩展名
        if not filename.lower().endswith('.png'):
            raise ValueError("文件必须是PNG格式")

        # Windows特定检查
        if os.name == 'nt':
            # 检查Windows保留名称
            reserved_names = {'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
                            'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3',
                            'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'}
            name_without_ext = os.path.splitext(filename)[0].lower()
            if name_without_ext in reserved_names:
                raise ValueError("文件名不能使用Windows保留名称")

            # 检查Windows路径长度
            if len(save_path) > 260:
                raise ValueError("文件路径过长")

            # 检查Windows非法字符
            invalid_chars = r'[<>:"/\\|?*]'
            if re.search(invalid_chars, filename):
                raise ValueError("文件名包含非法字符")

        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                raise ValueError(f"无法创建目录: {str(e)}")

        # 检查写入权限
        if os.path.exists(save_dir):
            test_file = os.path.join(save_dir, '.test_write')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception:
                raise ValueError("没有写入权限")

        try:
            # 创建图表
            plt.figure(figsize=(12, 6))
            plt.clf()
            tokens = list(distribution.keys())
            frequencies = list(distribution.values())
            
            # 使用seaborn创建条形图
            sns.barplot(x=tokens, y=frequencies)
            plt.xticks(rotation=45, ha='right')
            plt.title('Token分布')
            plt.xlabel('Token')
            plt.ylabel('频率')
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"分布图已保存到: {save_path}")
        except Exception as e:
            plt.close()
            raise ValueError(f"保存分布图时出错: {str(e)}")

    def cleanup(self) -> None:
        """清理资源。

        清理所有打开的文件句柄和图表资源。
        """
        try:
            # 关闭所有matplotlib图表
            plt.close('all')
        except Exception as e:
            self.logger.warning(f"清理图表资源时出错: {str(e)}")
