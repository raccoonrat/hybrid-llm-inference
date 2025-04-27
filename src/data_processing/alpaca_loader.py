import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd

class AlpacaLoader:
    """Alpaca数据集加载器。"""
    
    def __init__(self, data_path: Union[str, Path]):
        """初始化Alpaca数据加载器。

        Args:
            data_path: 数据文件路径

        Raises:
            ValueError: 路径无效
            FileNotFoundError: 文件不存在
        """
        self.logger = logging.getLogger(__name__)
        
        if not data_path:
            raise ValueError("数据路径不能为空")
            
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"找不到数据文件：{data_path}")
            
        if self.data_path.suffix.lower() not in ['.json', '.jsonl']:
            raise ValueError("数据文件必须是JSON或JSONL格式")
    
    def validate_entry(self, entry: Dict) -> bool:
        """验证数据条目的格式。

        Args:
            entry: 数据条目

        Returns:
            是否有效
        """
        required_fields = ["instruction", "input", "output"]
        
        # 检查必需字段
        if not all(field in entry for field in required_fields):
            self.logger.warning(f"数据条目缺少必需字段：{required_fields}")
            return False
            
        # 检查字段类型
        if not all(isinstance(entry[field], str) for field in required_fields):
            self.logger.warning("数据字段必须是字符串类型")
            return False
            
        # 检查内容是否为空
        if not entry["instruction"].strip():
            self.logger.warning("instruction字段不能为空")
            return False
            
        return True
        
    def load_data(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """加载数据集。

        Args:
            max_samples: 最大加载样本数

        Returns:
            包含数据的DataFrame

        Raises:
            RuntimeError: 加载失败
        """
        try:
            # 读取数据文件
            with open(self.data_path, 'r', encoding='utf-8') as f:
                if self.data_path.suffix.lower() == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
                    
            if not isinstance(data, list):
                raise ValueError("数据必须是列表格式")
                
            # 验证和过滤数据
            valid_data = []
            for entry in data:
                if self.validate_entry(entry):
                    valid_data.append(entry)
                    if max_samples and len(valid_data) >= max_samples:
                        break
                        
            if not valid_data:
                raise ValueError("没有有效的数据条目")
                
            # 转换为DataFrame
            df = pd.DataFrame(valid_data)
            
            # 添加组合文本字段
            df['text'] = df.apply(
                lambda x: f"Instruction: {x['instruction']}\nInput: {x['input']}\nOutput: {x['output']}", 
                axis=1
            )
            
            self.logger.info(f"成功加载{len(df)}个有效样本")
            return df
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise RuntimeError(f"数据加载失败: {str(e)}")
            
    def get_statistics(self) -> Dict:
        """获取数据集统计信息。

        Returns:
            统计信息字典
        """
        df = self.load_data()
        stats = {
            "total_samples": len(df),
            "avg_instruction_length": df["instruction"].str.len().mean(),
            "avg_input_length": df["input"].str.len().mean(),
            "avg_output_length": df["output"].str.len().mean(),
            "empty_input_ratio": (df["input"] == "").mean()
        }
        return stats 