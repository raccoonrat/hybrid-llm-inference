import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
import tempfile

class AlpacaLoader:
    """Alpaca数据集加载器。"""
    
    def __init__(self, data_path: str):
        """初始化Alpaca数据集加载器。

        Args:
            data_path: 数据集路径
        """
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)

    def validate_entry(self, entry: Dict) -> bool:
        """验证数据条目。

        Args:
            entry: 数据条目

        Returns:
            bool: 条目是否有效
        """
        if not isinstance(entry, dict):
            return False
            
        required_fields = ['instruction', 'input', 'output']
        if not all(field in entry for field in required_fields):
            return False
            
        if not all(isinstance(entry[field], str) for field in required_fields):
            return False
            
        if not entry['instruction'].strip():
            return False
            
        return True

    def get_statistics(self) -> Dict:
        """获取数据集统计信息。

        Returns:
            Dict: 统计信息，包含总条目数、有效条目数、无效条目数和其他统计指标
        """
        if not Path(self.data_path).exists():
            self.logger.error(f"数据文件不存在: {self.data_path}")
            return {
                'total_entries': 0,
                'valid_entries': 0,
                'invalid_entries': 0,
                'total_samples': 0,
                'avg_instruction_length': 0,
                'avg_input_length': 0,
                'avg_output_length': 0,
                'empty_input_ratio': 0,
                'error': '数据文件不存在'
            }

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            self.logger.error(f"无效的JSON文件: {self.data_path}")
            return {
                'total_entries': 0,
                'valid_entries': 0,
                'invalid_entries': 0,
                'total_samples': 0,
                'avg_instruction_length': 0,
                'avg_input_length': 0,
                'avg_output_length': 0,
                'empty_input_ratio': 0,
                'error': '无效的JSON文件'
            }
        except Exception as e:
            self.logger.error(f"读取文件时出错: {str(e)}")
            return {
                'total_entries': 0,
                'valid_entries': 0,
                'invalid_entries': 0,
                'total_samples': 0,
                'avg_instruction_length': 0,
                'avg_input_length': 0,
                'avg_output_length': 0,
                'empty_input_ratio': 0,
                'error': f'读取文件时出错: {str(e)}'
            }

        total_entries = len(data) if isinstance(data, list) else 0
        valid_entries = sum(1 for entry in data if self.validate_entry(entry)) if total_entries > 0 else 0
        invalid_entries = total_entries - valid_entries
        
        # 计算额外的统计信息
        instruction_lengths = []
        input_lengths = []
        output_lengths = []
        empty_inputs = 0
        
        for entry in data:
            if self.validate_entry(entry):
                instruction_lengths.append(len(entry['instruction']))
                input_lengths.append(len(entry['input']))
                output_lengths.append(len(entry['output']))
                if not entry['input'].strip():
                    empty_inputs += 1

        stats = {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'invalid_entries': invalid_entries,
            'total_samples': total_entries,
            'avg_instruction_length': sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
            'avg_input_length': sum(input_lengths) / len(input_lengths) if input_lengths else 0,
            'avg_output_length': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            'empty_input_ratio': empty_inputs / valid_entries if valid_entries > 0 else 0
        }
        return stats

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
                if self.data_path.endswith('.jsonl'):
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