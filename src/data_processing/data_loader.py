# hybrid-llm-inference/src/data_processing/data_loader.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

class DataLoader:
    """数据加载器类。"""
    
    def __init__(self):
        """初始化数据加载器。"""
        self.logger = logging.getLogger(__name__)
        
    def load(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """加载数据文件。
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            List[Dict[str, Any]]: 加载的数据列表
            
        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON解析错误
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在：{file_path}")
                
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                data = [data]
                
            # 验证数据格式
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"数据项必须是字典类型：{item}")
                    
            self.logger.info(f"成功加载{len(data)}条数据")
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误：{e}")
            raise
        except Exception as e:
            self.logger.error(f"加载数据时发生错误：{e}")
            raise

    def get_data(self):
        """Return loaded data."""
        if self.data is None:
            self.logger.warning("Data not loaded. Call load() first.")
            return pd.DataFrame()
        return self.data

