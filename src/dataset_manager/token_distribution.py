# hybrid-llm-inference/src/dataset_manager/token_distribution.py
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)

class TokenDistribution:
    """令牌分布分析类。"""
    
    def __init__(self, data: pd.DataFrame, models: dict, output_dir: str = "data/processed"):
        """初始化Token分布分析器。
        
        Args:
            data: 包含'instruction'和'output'列的DataFrame
            models: 模型字典
            output_dir: 输出目录
        """
        self.data = data
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.distribution = None
        self.stats = None
        
    def analyze(self, model_name: str = "llama3", save_path: Optional[str] = None) -> Dict:
        """分析输入和输出token的分布。
        
        Args:
            model_name: 使用的模型名称
            save_path: 图表保存路径
            
        Returns:
            Dict: 分析结果，包含分布和统计信息
            
        Raises:
            ValueError: 模型不存在
        """
        if model_name not in self.models:
            self.logger.error(f"模型{model_name}不存在")
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        input_tokens = []
        output_tokens = []
        
        # 计算token数量
        for _, row in self.data.iterrows():
            instruction = row["instruction"]
            output = row["output"]
            
            input_count = model.get_token_count(instruction) if model else len(instruction)
            output_count = model.get_token_count(output) if model and output else 0
            
            # 根据论文范围过滤
            if 8 <= input_count <= 2048 and (0 <= output_count <= 4096 or output_count == 0):
                input_tokens.append(input_count)
                output_tokens.append(output_count)
            else:
                self.logger.debug(f"跳过范围外的token：input={input_count}, output={output_count}")
                
        # 计算分布
        input_series = pd.Series(input_tokens)
        output_series = pd.Series(output_tokens)
        
        self.distribution = {
            "input_distribution": input_series.value_counts().sort_index().to_dict(),
            "output_distribution": output_series.value_counts().sort_index().to_dict()
        }
        
        # 计算统计信息
        self.stats = {
            "input": {
                "mean": input_series.mean(),
                "median": input_series.median(),
                "std": input_series.std(),
                "min": input_series.min(),
                "max": input_series.max()
            },
            "output": {
                "mean": output_series.mean(),
                "median": output_series.median(),
                "std": output_series.std(),
                "min": output_series.min(),
                "max": output_series.max()
            }
        }
        
        # 可视化分布
        self._visualize_distribution(save_path)
        
        # 返回分布和统计信息
        return self.distribution

    def save_distribution(self, save_path: str) -> None:
        """保存分布结果到文件。
        
        Args:
            save_path: 保存路径
            
        Raises:
            ValueError: 当分布未计算、文件已存在或保存失败时
        """
        if self.distribution is None or self.stats is None:
            raise ValueError("分布未计算，请先调用analyze()")
            
        save_path = Path(save_path)
        if save_path.exists():
            raise ValueError("文件已存在")
            
        try:
            data = {
                "distribution": self.distribution,
                "stats": self.stats
            }
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
            self.logger.info(f"分布数据已保存到：{save_path}")
        except Exception as e:
            self.logger.error(f"保存分布数据时出错：{e}")
            raise ValueError(f"保存分布数据失败：{str(e)}")
            
    def load_distribution(self, load_path: str) -> Dict:
        """从文件加载分布结果。
        
        Args:
            load_path: 加载路径
            
        Returns:
            Dict: 加载的分布数据
            
        Raises:
            ValueError: 当文件不存在或加载失败时
        """
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.distribution = data["distribution"]
            self.stats = data["stats"]
            self.logger.info(f"已从{load_path}加载分布数据")
            return self.distribution
        except FileNotFoundError:
            raise ValueError(f"文件不存在：{load_path}")
        except Exception as e:
            self.logger.error(f"加载分布数据时出错：{e}")
            raise ValueError(f"加载分布数据失败：{str(e)}")
        
    def _visualize_distribution(self, save_path: Optional[str] = None):
        """Generate and save token distribution plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Input tokens
        input_dist = self.distribution["input_distribution"]
        ax1.bar(list(input_dist.keys()), list(input_dist.values()), width=10)
        ax1.set_title("Input Token Distribution")
        ax1.set_xlabel("Token Count")
        ax1.set_ylabel("Frequency")
        ax1.set_yscale("log")  # Use log scale for better visualization
        ax1.grid(True, which="both", ls="--")
        ax1.set_xlim(0, 2048)
        
        # Output tokens
        output_dist = self.distribution["output_distribution"]
        ax2.bar(list(output_dist.keys()), list(output_dist.values()), width=10)
        ax2.set_title("Output Token Distribution")
        ax2.set_xlabel("Token Count")
        ax2.set_ylabel("Frequency")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", ls="--")
        ax2.set_xlim(0, 4096)
        
        plt.tight_layout()
        plot_path = self.output_dir / "token_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Distribution plot saved to {plot_path}")
        
        if save_path:
            try:
                plt.savefig(save_path)
                self.logger.info(f"Plot saved to: {save_path}")
            except Exception as e:
                self.logger.error(f"Error saving plot: {e}")
                raise
        
    def get_distribution(self):
        """返回token分布和统计信息。
        
        Returns:
            tuple: (分布字典, 统计字典)
        """
        if self.distribution is None or self.stats is None:
            self.logger.warning("分布未分析，请先调用analyze()")
            return {}, {}
        return self.distribution, self.stats

