# hybrid-llm-inference/src/benchmarking/report_generator.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import numpy as np
from toolbox.logger import get_logger
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """报告生成器类。"""
    
    def __init__(self, output_dir: str) -> None:
        """初始化报告生成器。"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("报告生成器初始化完成")
    
    def _validate_tradeoff_results(self, tradeoff_results: Dict[str, Any]) -> None:
        """验证权衡结果的格式和内容。

        Args:
            tradeoff_results: 包含权衡分析结果的字典

        Raises:
            ValueError: 当结果格式不正确或值无效时抛出
        """
        # 检查必需字段
        if not isinstance(tradeoff_results, dict):
            raise ValueError("权衡结果必须是字典类型")
            
        required_fields = ["weights", "values"]
        for field in required_fields:
            if field not in tradeoff_results:
                raise ValueError(f"权衡结果缺少必需字段: {field}")
        
        # 验证权重
        weights = tradeoff_results["weights"]
        if not isinstance(weights, list):
            raise ValueError("weights必须是列表类型")
        if not weights:
            raise ValueError("weights不能为空")
        if not all(isinstance(w, (int, float)) for w in weights):
            raise ValueError("weights必须全部为数值类型")
        if not all(0 <= w <= 1 for w in weights):
            raise ValueError("weights的值必须在0到1之间")
            
        # 验证值列表
        values = tradeoff_results["values"]
        if not isinstance(values, list):
            raise ValueError("values必须是列表类型")
        if not values:
            raise ValueError("values不能为空")
            
        # 验证每个值字典
        required_metrics = {"energy", "runtime", "throughput"}
        for i, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"values中的第{i+1}个元素必须是字典类型")
            
            # 检查必需的指标
            missing_metrics = required_metrics - set(value.keys())
            if missing_metrics:
                raise ValueError(f"values中的第{i+1}个元素缺少以下指标: {missing_metrics}")
            
            # 验证指标值
            for metric, val in value.items():
                if not isinstance(val, (int, float)):
                    raise ValueError(f"values中的第{i+1}个元素的{metric}必须是数值类型")
                if val < 0:
                    raise ValueError(f"values中的第{i+1}个元素的{metric}不能为负数")
    
    def generate_report(self, benchmark_results: dict, tradeoff_results: dict) -> None:
        """生成报告。"""
        try:
            # 首先验证权衡结果
            self._validate_tradeoff_results(tradeoff_results)
            self._generate_plots(benchmark_results, tradeoff_results)
            self._generate_html_report(benchmark_results, tradeoff_results)
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            raise
    
    def _generate_plots(self, benchmark_results: dict, tradeoff_results: dict) -> None:
        """生成图表。"""
        try:
            self._plot_energy_per_token(benchmark_results)
            self._plot_runtime(benchmark_results)
            self._plot_tradeoff_curve(tradeoff_results)
        except Exception as e:
            logger.error(f"生成图表失败: {str(e)}")
            raise
    
    def _plot_energy_per_token(self, results: dict) -> None:
        """绘制每令牌能量图。"""
        plt.figure(figsize=(10, 6))
        models = list(results.keys())
        energy_per_token = [results[model]["summary"]["avg_energy_per_token"] for model in models]
        
        plt.bar(models, energy_per_token)
        plt.xlabel("Model")
        plt.ylabel("Energy per Token (J/token)")
        plt.title("Energy per Token Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "energy_per_token.png"))
        plt.close()
    
    def _plot_runtime(self, results: dict) -> None:
        """绘制运行时间图。"""
        plt.figure(figsize=(10, 6))
        models = list(results.keys())
        runtime = [results[model]["summary"]["avg_runtime"] for model in models]
        
        plt.bar(models, runtime)
        plt.xlabel("Model")
        plt.ylabel("Runtime (s)")
        plt.title("Runtime Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "runtime.png"))
        plt.close()
    
    def _plot_tradeoff_curve(self, results: dict) -> None:
        """绘制权衡曲线。"""
        plt.figure(figsize=(10, 6))
        weights = results["weights"]
        values = results["values"]
        
        # 确保每个指标的值列表长度与权重列表长度相同
        energy_values = []
        runtime_values = []
        throughput_values = []
        
        # 如果只有一个值，则将其重复扩展到与权重列表相同的长度
        if len(values) == 1:
            value = values[0]
            energy_values = [value["energy"]] * len(weights)
            runtime_values = [value["runtime"]] * len(weights)
            throughput_values = [value["throughput"]] * len(weights)
        else:
            energy_values = [v["energy"] for v in values]
            runtime_values = [v["runtime"] for v in values]
            throughput_values = [v["throughput"] for v in values]
        
        plt.plot(weights, energy_values, label="Energy")
        plt.plot(weights, runtime_values, label="Runtime")
        plt.plot(weights, throughput_values, label="Throughput")
        
        plt.xlabel("Weight")
        plt.ylabel("Value")
        plt.title("Energy-Runtime Tradeoff Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "tradeoff_curve.png"))
        plt.close()
    
    def _generate_html_report(self, benchmark_results: dict, tradeoff_results: dict) -> None:
        """生成HTML报告。"""
        html_content = """
        <html>
        <head>
            <title>Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                img { max-width: 100%; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Benchmark Report</h1>
            <h2>Performance Metrics</h2>
            <img src="energy_per_token.png" alt="Energy per Token">
            <img src="runtime.png" alt="Runtime">
            <img src="tradeoff_curve.png" alt="Tradeoff Curve">
        </body>
        </html>
        """
        
        with open(os.path.join(self.output_dir, "report.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
