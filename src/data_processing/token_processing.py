# hybrid-llm-inference/src/data_processing/token_processing.py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from toolbox.logger import get_logger
from model_zoo import get_model
import os
import numpy as np
from typing import Dict, List, Optional, Union, Any
from .token_processor import TokenProcessor
import logging
from collections import Counter
import seaborn as sns
import re
import time
import psutil
import tempfile
from .data_processor import DataProcessor
from .alpaca_loader import AlpacaLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__)

class TokenProcessing:
    """令牌处理类，用于处理模型推理过程中的令牌。"""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        """初始化令牌处理器。

        Args:
            model_name: 模型名称
            model_config: 模型配置
        """
        self.model_name = model_name
        self.model_config = model_config
        self.initialized = False
    
    def initialize(self) -> None:
        """初始化令牌处理器。"""
        if self.initialized:
            return
            
        # 初始化模型特定的令牌处理逻辑
        self._init_model_specific_processing()
        self.initialized = True
        logger.info(f"已初始化模型 {self.model_name} 的令牌处理器")
    
    def _init_model_specific_processing(self) -> None:
        """初始化模型特定的令牌处理逻辑。"""
        # 根据模型配置初始化特定的处理逻辑
        pass
    
    def process_tokens(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """处理令牌序列。

        Args:
            data: 输入数据列表，每个元素是一个字典

        Returns:
            pd.DataFrame: 包含以下列的DataFrame：
                - decoded_text: 解码后的文本
                - input_tokens: 输入令牌列表
        """
        if not self.initialized:
            raise RuntimeError("令牌处理器未初始化")
            
        # 确保输入是列表
        if not isinstance(data, list):
            data = [data]
            
        # 处理每个输入
        processed_data = []
        for item in data:
            # 确保item是字典
            if not isinstance(item, dict):
                item = {"text": str(item)}
                
            # 获取文本
            text = str(item.get("text", ""))
            
            # 处理令牌
            tokens = self._apply_model_specific_processing([ord(c) for c in text])
            
            # 创建结果字典
            result = {
                "decoded_text": text,
                "input_tokens": tokens
            }
            processed_data.append(result)
            
        # 创建DataFrame
        df = pd.DataFrame(processed_data)
        
        # 确保必需的列存在
        if "decoded_text" not in df.columns:
            df["decoded_text"] = ""
        if "input_tokens" not in df.columns:
            df["input_tokens"] = [[]] * len(df)
            
        return df
    
    def _apply_model_specific_processing(self, tokens: List[int]) -> List[int]:
        """应用模型特定的令牌处理逻辑。

        Args:
            tokens: 输入令牌序列

        Returns:
            处理后的令牌序列
        """
        # 默认实现：直接返回输入令牌序列
        return tokens

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
            logger.warning("输入DataFrame为空")
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
            logger.info(f"处理 {len(df)} 行数据用时: {process_time:.2f}秒")
            
            return output
            
        except Exception as e:
            logger.error(f"获取token数据时出错: {str(e)}")
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
            logger.warning("输入数据为空")
            return {}

        # 如果提供了保存路径，先进行验证
        if save_path:
            # 基本路径验证
            if not save_path or not isinstance(save_path, str) or save_path.isspace():
                raise ValueError("保存路径不能为空")

            # 规范化路径
            save_path = Path(save_path)
            if not save_path.is_absolute():
                raise ValueError("必须使用绝对路径")

            # 检查文件扩展名
            if save_path.suffix.lower() != '.png':
                raise ValueError("必须是.png格式")

            # 检查文件名
            filename = save_path.name
            if not filename or filename.isspace() or filename.startswith('.') or filename.endswith('.'):
                raise ValueError("无效的文件名")

            # Windows特定检查
            if os.name == 'nt':
                # 检查Windows保留名称
                reserved_names = {'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
                                'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3',
                                'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'}
                name_without_ext = save_path.stem.lower()
                if name_without_ext in reserved_names:
                    raise ValueError("文件名不能使用Windows保留名称")

                # 检查Windows路径长度
                if len(str(save_path)) > 260:
                    raise ValueError("路径长度超过系统限制")

                # 检查Windows非法字符
                invalid_chars = '<>:"|?*\0'
                if any(char in filename for char in invalid_chars):
                    raise ValueError("文件名包含无效字符")

                # 检查UNC路径
                if str(save_path).startswith('\\\\'):
                    raise ValueError("无法访问网络路径")

                # 检查系统目录
                system_dirs = ['C:\\Windows', 'C:\\Windows\\System32']
                if any(str(save_path).startswith(sys_dir) for sys_dir in system_dirs):
                    raise ValueError("不能保存到系统目录")

            # 检查目录是否存在
            save_dir = save_path.parent
            if not save_dir.exists():
                try:
                    save_dir.mkdir(parents=True)
                except (OSError, PermissionError):
                    raise ValueError("目录不存在且无法创建")

            # 检查写入权限
            try:
                if save_path.exists():
                    # 如果文件已存在，尝试打开它进行写入
                    with open(save_path, 'a'):
                        pass
                else:
                    # 如果文件不存在，尝试在目录中创建测试文件
                    test_file = save_dir / '.test_write'
                    test_file.touch()
                    test_file.unlink()
            except (OSError, PermissionError):
                raise ValueError("没有写入权限")

        try:
            # 检查内存使用
            process = psutil.Process()
            if process.memory_info().rss > 1024 * 1024 * 1024:  # 1GB限制
                raise MemoryError("内存使用过高，请减少输入数据量")

            # 确定使用哪一列
            token_column = 'input_tokens' if 'input_tokens' in df.columns else 'token'
            if token_column not in df.columns:
                logger.warning(f"DataFrame中缺少所需的列: {token_column}")
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
                    logger.info(f"已处理 {i}/{len(df)} 行数据")

            if not all_tokens:
                logger.warning("没有找到有效的token")
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
                    logger.error(f"保存分布图时出错: {str(e)}")
                    raise
                except Exception as e:
                    logger.error(f"保存分布图时出错: {str(e)}")
                    raise ValueError(f"保存分布图失败: {str(e)}")

            return distribution

        except MemoryError as e:
            logger.error(f"内存不足: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"无效的输入或路径: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"计算分布时出错: {str(e)}")
            raise RuntimeError(f"计算分布时出错: {str(e)}")
        
    def _save_distribution_plot(self, distribution: Dict[str, float], save_path: Union[str, Path]) -> None:
        """保存分布图到指定路径。

        Args:
            distribution: token分布字典
            save_path: 保存路径，可以是字符串或Path对象

        Raises:
            ValueError: 当路径无效或没有写入权限时
        """
        # 基本路径验证
        if save_path is None:
            raise ValueError("保存路径不能为None")

        # 规范化路径
        if isinstance(save_path, str):
            if not save_path.strip():
                raise ValueError("保存路径不能为空")
            save_path = Path(save_path)

        # 检查文件扩展名
        if save_path.suffix.lower() != '.png':
            raise ValueError("必须是.png格式")

        # 检查文件名
        filename = save_path.name
        if not filename or filename.isspace() or filename.startswith('.') or filename.endswith('.'):
            raise ValueError("无效的文件名")

        # Windows特定检查
        if os.name == 'nt':
            # 检查Windows保留名称
            reserved_names = {'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
                            'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3',
                            'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'}
            name_without_ext = save_path.stem.lower()
            if name_without_ext in reserved_names:
                raise ValueError("文件名不能使用Windows保留名称")

            # 检查Windows路径长度
            if len(str(save_path)) > 260:
                raise ValueError("路径长度超过系统限制")

            # 检查Windows非法字符
            invalid_chars = '<>:"|?*\0'
            if any(char in filename for char in invalid_chars):
                raise ValueError("文件名包含无效字符")

            # 检查UNC路径
            if str(save_path).startswith('\\\\'):
                raise ValueError("无法访问网络路径")

            # 检查系统目录
            system_dirs = ['C:\\Windows', 'C:\\Windows\\System32']
            if any(str(save_path).startswith(sys_dir) for sys_dir in system_dirs):
                raise ValueError("不能保存到系统目录")

        # 检查目录是否存在
        save_dir = save_path.parent
        if not save_dir.exists():
            try:
                save_dir.mkdir(parents=True)
            except (OSError, PermissionError):
                raise ValueError("目录不存在且无法创建")

        # 检查写入权限
        try:
            if save_path.exists():
                # 如果文件已存在，尝试打开它进行写入
                with open(save_path, 'a'):
                    pass
            else:
                # 如果文件不存在，尝试在目录中创建测试文件
                test_file = save_dir / '.test_write'
                test_file.touch()
                test_file.unlink()
        except (OSError, PermissionError):
            raise ValueError("没有写入权限")

        # 创建并保存图表
        plt.figure(figsize=(10, 6))
        plt.bar(distribution.keys(), distribution.values())
        plt.title('Token Distribution')
        plt.xlabel('Token')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)

        try:
            plt.savefig(str(save_path))
            plt.close()
            logging.info(f"分布图已保存到: {save_path}")
        except Exception as e:
            plt.close()
            if isinstance(e, PermissionError):
                raise ValueError("没有写入权限")
            raise ValueError(f"保存图表失败: {str(e)}")

    def cleanup(self) -> None:
        """清理资源。

        清理所有打开的文件句柄和图表资源。
        """
        try:
            # 关闭所有matplotlib图表
            plt.close('all')
        except Exception as e:
            logger.warning(f"清理图表资源时出错: {str(e)}")
