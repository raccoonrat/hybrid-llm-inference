import logging
from typing import Union, List, Dict, Any, Generator
import pandas as pd

class DataProcessor:
    """数据处理器类。"""
    
    def __init__(self):
        """初始化数据处理器。"""
        self.logger = logging.getLogger(__name__)
        
    def validate_data(self, data: Union[pd.DataFrame, List[Dict], Dict]) -> pd.DataFrame:
        """验证并标准化输入数据。

        Args:
            data: 输入数据，可以是DataFrame、字典列表或单个字典

        Returns:
            标准化后的DataFrame

        Raises:
            ValueError: 数据格式无效
            TypeError: 数据类型错误
        """
        if data is None:
            raise ValueError("输入数据不能为None")
            
        if isinstance(data, dict):
            data = [data]
            
        if isinstance(data, list):
            if not data:
                raise ValueError("输入数据列表不能为空")
            if not all(isinstance(d, dict) for d in data):
                raise TypeError("列表中所有元素必须是字典")
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("输入数据必须是DataFrame、字典或字典列表")
            
        if data.empty:
            raise ValueError("DataFrame不能为空")
            
        required_columns = ["text", "tokens", "length"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必需的列：{missing_columns}")
            
        return data
        
    def process_batch(self, data: pd.DataFrame, batch_size: int = 32) -> Generator[pd.DataFrame, None, None]:
        """批量处理数据。

        Args:
            data: 输入DataFrame
            batch_size: 批次大小

        Yields:
            处理后的数据批次

        Raises:
            ValueError: 参数无效
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size必须是正整数")
            
        data = self.validate_data(data)
        
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size].copy()
            try:
                # 在这里进行批次处理
                yield batch
            except Exception as e:
                self.logger.error(f"处理批次{i//batch_size}失败: {str(e)}")
                raise RuntimeError(f"批次处理失败: {str(e)}")

    def calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算数据统计信息。

        Args:
            data: 输入DataFrame

        Returns:
            统计信息字典

        Raises:
            ValueError: 数据无效
        """
        data = self.validate_data(data)
        
        try:
            stats = {
                "total_samples": len(data),
                "avg_length": data["length"].mean(),
                "max_length": data["length"].max(),
                "min_length": data["length"].min(),
                "std_length": data["length"].std(),
                "token_count": data["tokens"].apply(len).sum()
            }
            return stats
        except Exception as e:
            self.logger.error(f"计算统计信息失败: {str(e)}")
            raise RuntimeError(f"统计计算失败: {str(e)}") 