# hybrid-llm-inference/src/dataset_manager/alpaca_loader.py
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)

class AlpacaLoader:
    """Alpaca数据集加载器类。"""
    
    def __init__(self, dataset_path: str):
        """初始化加载器。
        
        Args:
            dataset_path (str): 数据集路径
        """
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        self.data = None
        
    def load(self) -> pd.DataFrame:
        """加载数据集。
        
        Returns:
            pd.DataFrame: 加载的数据
            
        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON解析错误
            ValueError: 数据集为空
        """
        if not self.dataset_path.exists():
            self.logger.error(f"数据集不存在：{self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                
            if not raw_data:
                self.logger.error("数据集为空")
                raise ValueError("Dataset is empty")
                
            df = pd.DataFrame(raw_data)
            required_columns = ["instruction", "input", "output"]
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"数据集缺少必需的列：{col}")
                    raise ValueError(f"Dataset missing required column: {col}")
                
            self.data = df
            self.logger.info(f"已加载并验证{len(self.data)}条数据，来自{self.dataset_path}")
            return self.data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误：{e}")
            raise json.JSONDecodeError(f"JSON parsing error: {e.msg}", e.doc, e.pos)
        except Exception as e:
            self.logger.error(f"加载数据时发生错误：{e}")
            raise
            
    def get_data(self) -> pd.DataFrame:
        """获取已加载的数据。
        
        Returns:
            pd.DataFrame: 已加载的数据，如果未加载则返回空DataFrame
        """
        if self.data is None:
            self.logger.warning("数据未加载，请先调用load()")
            return pd.DataFrame()
        return self.data
