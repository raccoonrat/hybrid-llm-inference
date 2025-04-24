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
            Dict: 分析结果
            
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
        
        # 保存分布和统计信息
        dist_path = self.output_dir / "token_distribution.pkl"
        with open(dist_path, "wb") as f:
            pickle.dump({"distribution": self.distribution, "stats": self.stats}, f)
        self.logger.info(f"已保存token分布和统计信息到{dist_path}")
        
        # 可视化分布
        self._visualize_distribution(save_path)
        return self.stats
        
    def _visualize_distribution(self, save_path: Optional[str] = None):
        """生成并保存token分布图。"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 输入token
        input_dist = self.distribution["input_distribution"]
        ax1.bar(list(input_dist.keys()), list(input_dist.values()), width=10)
        ax1.set_title("输入Token分布")
        ax1.set_xlabel("Token数量")
        ax1.set_ylabel("频率")
        ax1.set_yscale("log")  # 使用对数尺度以便更好地可视化
        ax1.grid(True, which="both", ls="--")
        ax1.set_xlim(0, 2048)
        
        # 输出token
        output_dist = self.distribution["output_distribution"]
        ax2.bar(list(output_dist.keys()), list(output_dist.values()), width=10)
        ax2.set_title("输出Token分布")
        ax2.set_xlabel("Token数量")
        ax2.set_ylabel("频率")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", ls="--")
        ax2.set_xlim(0, 4096)
        
        plt.tight_layout()
        plot_path = self.output_dir / "token_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"已保存分布图到{plot_path}")
        
        if save_path:
            try:
                plt.savefig(save_path)
                self.logger.info(f"图表已保存到：{save_path}")
            except Exception as e:
                self.logger.error(f"保存图表时出错：{e}")
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

