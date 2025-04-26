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
            
    def compute_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, float]:
        """计算token分布。

        Args:
            df (pd.DataFrame): 包含token数据的DataFrame
            save_path (Optional[str], optional): 保存分布图的路径. Defaults to None.

        Returns:
            Dict[str, float]: token分布字典

        Raises:
            ValueError: 当输入数据无效或保存路径无效时
        """
        if df is None:
            raise ValueError("输入DataFrame不能为None")

        if df.empty:
            self.logger.warning("输入数据为空")
            return {}

        # 确定使用哪一列
        token_column = 'input_tokens' if 'input_tokens' in df.columns else 'token'
        if token_column not in df.columns:
            self.logger.warning(f"DataFrame中缺少所需的列: {token_column}")
            return {}

        # 展平所有token列表并计算频率分布
        all_tokens = []
        for tokens in df[token_column]:
            if isinstance(tokens, list):
                all_tokens.extend(map(str, tokens))
            else:
                all_tokens.append(str(tokens))

        if not all_tokens:
            self.logger.warning("没有找到有效的token")
            return {}

        # 计算分布
        total_tokens = len(all_tokens)
        distribution = {token: count/total_tokens for token, count in Counter(all_tokens).items()}

        # 如果提供了保存路径，则保存分布图
        if save_path:
            try:
                self._save_distribution_plot(distribution, save_path)
            except ValueError as e:
                self.logger.error(f"保存分布图时出错: {str(e)}")
                raise

        return distribution
        
    def _save_distribution_plot(self, distribution: Dict[str, float], save_path: str) -> None:
        """保存分布图到指定路径。

        Args:
            distribution (Dict[str, float]): token分布字典
            save_path (str): 保存路径

        Raises:
            ValueError: 当路径无效或没有写入权限时
        """
        if not save_path or save_path.isspace():
            raise ValueError("保存路径不能为空")

        # 标准化路径
        try:
            save_path = os.path.abspath(save_path)
        except Exception as e:
            raise ValueError(f"无效的路径格式: {str(e)}")

        # 基本路径验证
        if not save_path.lower().endswith('.png'):
            raise ValueError("必须是.png格式")

        # 检查文件名
        file_name = os.path.basename(save_path)
        dir_path = os.path.dirname(save_path)

        # Windows特定检查
        if os.name == 'nt':
            # 检查保留名称
            reserved_names = {'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
                            'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3',
                            'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'}
            if file_name.split('.')[0].lower() in reserved_names:
                raise ValueError("文件名不能使用Windows保留名称")

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

        # 检查系统目录
        system_dirs = [
            os.environ.get('SystemRoot', 'C:\\Windows'),
            os.environ.get('ProgramFiles', 'C:\\Program Files'),
            os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')
        ]
        if any(save_path.lower().startswith(d.lower()) for d in system_dirs if d):
            raise ValueError("不能保存到系统目录")

        # 检查目录是否存在
        if not os.path.exists(dir_path):
            raise ValueError(f"目录不存在: {dir_path}")

        # 权限检查
        try:
            # 检查目录权限
            if not os.access(dir_path, os.W_OK):
                raise ValueError("没有目录写入权限")

            # 如果文件已存在，检查文件权限
            if os.path.exists(save_path):
                if not os.access(save_path, os.W_OK):
                    raise ValueError("没有文件写入权限")
                try:
                    # 尝试打开文件进行写入测试
                    with open(save_path, 'a'):
                        pass
                except (IOError, OSError):
                    raise ValueError("文件被锁定或无法写入")
            else:
                # 测试是否可以在目录中创建新文件
                test_file = os.path.join(dir_path, f"test_{int(time.time())}.tmp")
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except (IOError, OSError):
                    raise ValueError("无法在目录中创建新文件")

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"权限检查失败: {str(e)}")

        # 创建和保存图表
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(distribution.keys(), distribution.values())
            plt.title('Token Distribution')
            plt.xlabel('Token')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(save_path)
            self.logger.info(f"分布图已保存到: {save_path}")

        except Exception as e:
            raise ValueError(f"保存图表失败: {str(e)}")

        finally:
            plt.close()
