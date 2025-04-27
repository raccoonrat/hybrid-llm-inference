import logging
from typing import Union, List, Dict, Any, Generator
import pandas as pd

class DataProcessor:
    """数据处理基类。"""
    
    def __init__(self):
        """初始化数据处理器。"""
        self.logger = logging.getLogger(__name__)
        
    def validate_data(self, data: Union[pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """验证数据格式。

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 验证后的数据

        Raises:
            ValueError: 当输入数据为None或空时
            TypeError: 当输入数据类型不正确时
        """
        if data is None:
            raise ValueError("输入数据不能为None")
            
        if isinstance(data, list):
            if not data:
                raise ValueError("输入数据列表不能为空")
            df = pd.DataFrame(data)
            
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("输入DataFrame不能为空")
            df = data
            
        elif isinstance(data, dict):
            if not data:
                raise ValueError("输入字典不能为空")
            df = pd.DataFrame([data])
            
        else:
            raise TypeError("输入数据必须是DataFrame、字典或字典列表")

        # 检查必需列
        required_columns = {'text', 'tokens', 'length'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"缺少必需的列: {', '.join(missing_columns)}")

        return df

    def process_batch(self, data: Union[pd.DataFrame, List[Dict]], batch_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
        """处理数据批次。

        Args:
            data: 输入数据
            batch_size: 批处理大小，必须是正整数

        Returns:
            Generator[pd.DataFrame, None, None]: 处理后的数据批次生成器

        Raises:
            ValueError: 当输入数据无效或batch_size不是正整数时
        """
        if batch_size <= 0:
            raise ValueError("batch_size必须是正整数")
            
        # 验证并转换数据为DataFrame
        df = self.validate_data(data)
            
        # 按批次处理数据
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            yield self._process_single_batch(batch)
        
    def _process_single_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """处理单个数据批次。子类应该重写此方法。

        Args:
            batch: 输入数据批次

        Returns:
            pd.DataFrame: 处理后的数据批次
        """
        return batch

    def calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算数据统计信息。

        Args:
            data: 输入DataFrame

        Returns:
            统计信息字典

        Raises:
            ValueError: 数据无效
        """
        df = self.validate_data(data)
        
        try:
            stats = {
                "total_samples": len(df),
                "avg_length": df["length"].mean() if "length" in df.columns else 0,
                "max_length": df["length"].max() if "length" in df.columns else 0,
                "min_length": df["length"].min() if "length" in df.columns else 0,
                "std_length": df["length"].std() if "length" in df.columns else 0,
                "token_count": df["tokens"].apply(len).sum() if "tokens" in df.columns else 0
            }
            return stats
        except Exception as e:
            self.logger.error(f"计算统计信息失败: {str(e)}")
            raise RuntimeError(f"统计计算失败: {str(e)}") 