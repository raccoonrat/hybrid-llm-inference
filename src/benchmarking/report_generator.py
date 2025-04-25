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
    
    def __init__(self, output_dir: str, style_config: Dict[str, Any] = None) -> None:
        """初始化报告生成器。

        Args:
            output_dir: 输出目录
            style_config: 图表样式配置
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        if style_config:
            plt.rcParams.update(style_config)
        
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
            
        # 如果是空字典，提供默认值
        if not tradeoff_results:
            tradeoff_results.update({
                "weights": [0.33, 0.33, 0.34],
                "values": []
            })
            return
            
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
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("weights的和必须等于1")
            
        # 验证值列表
        values = tradeoff_results["values"]
        if not isinstance(values, list):
            raise ValueError("values必须是列表类型")
            
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
                if metric == "throughput" and val == 0:
                    raise ValueError(f"values中的第{i+1}个元素的throughput不能为0")
    
    def _validate_metric_value(self, value, key):
        """验证指标值的格式。
        
        Args:
            value: 要验证的指标值
            key: 指标的键名
            
        Returns:
            bool: 验证是否通过
            
        Raises:
            ValueError: 当验证失败时抛出，包含具体错误信息
        """
        # 特殊处理 summary 字段，允许非数值数据
        if key == "summary":
            return True
        
        if isinstance(value, (int, float)):
            return True
        elif isinstance(value, list):
            # 递归验证列表中的每个值
            for item in value:
                if not isinstance(item, (int, float)):
                    raise ValueError(f"列表中的指标值必须是数值类型: {key}")
            return True
        elif isinstance(value, dict):
            # 递归验证字典中的每个值
            for k, v in value.items():
                if k != "summary":  # 跳过 summary 字段的验证
                    self._validate_metric_value(v, k)
            return True
        else:
            raise ValueError(f"指标值必须是数值类型、数值列表或字典: {key}")
    
    def validate_metrics(self, metrics):
        """验证指标数据的格式。
        
        Args:
            metrics (dict): 要验证的指标数据
            
        Returns:
            bool: 验证是否通过
            
        Raises:
            ValueError: 当验证失败时抛出，包含具体错误信息
        """
        if not metrics:
            raise ValueError("指标数据不能为空")
            
        if not isinstance(metrics, dict):
            raise ValueError("指标数据必须是字典格式")
        
        required_fields = ["throughput", "latency", "energy", "runtime"]
        for model, data in metrics.items():
            if not isinstance(data, dict):
                raise ValueError(f"模型 {model} 的指标数据必须是字典类型")
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"模型 {model} 缺少以下必需指标字段: {', '.join(missing_fields)}")
            
            for key, value in data.items():
                if key != "summary":
                    self._validate_metric_value(value, key)
                else:
                    # 验证 summary 字段
                    if not isinstance(value, dict):
                        raise ValueError(f"模型 {model} 的 summary 必须是字典类型")
                    required_summary_fields = ["avg_throughput", "avg_latency", "avg_energy_per_token", "avg_runtime"]
                    missing_summary_fields = [field for field in required_summary_fields if field not in value]
                    if missing_summary_fields:
                        raise ValueError(f"模型 {model} 的 summary 缺少以下字段: {', '.join(missing_summary_fields)}")
                    for field, val in value.items():
                        if not isinstance(val, (int, float)):
                            raise ValueError(f"模型 {model} 的 summary.{field} 必须是数值类型")
                        if val < 0:
                            raise ValueError(f"模型 {model} 的 summary.{field} 不能为负数")
                        if field == "avg_throughput" and val == 0:
                            raise ValueError(f"模型 {model} 的 summary.avg_throughput 不能为0")
    
        return True
    
    def plot_time_series(self, metrics: Dict[str, Any], metric_name: str) -> None:
        """绘制时间序列图。

        Args:
            metrics: 指标数据
            metric_name: 指标名称
        """
        plt.figure(figsize=(10, 6))
        for model, data in metrics.items():
            if isinstance(data[metric_name], list):
                plt.plot(data[metric_name], label=model)
        
        plt.xlabel("Time")
        plt.ylabel(metric_name.capitalize())
        plt.title(f"{metric_name.capitalize()} Time Series")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f"{metric_name}_time_series.png"))
        plt.close()
    
    def plot_boxplot(self, metrics: Dict[str, Any], metric_name: str) -> None:
        """绘制箱线图。

        Args:
            metrics: 指标数据
            metric_name: 指标名称
        """
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        for model, model_data in metrics.items():
            if isinstance(model_data[metric_name], list):
                data.append(model_data[metric_name])
                labels.append(model)
        
        plt.boxplot(data, labels=labels)
        plt.ylabel(metric_name.capitalize())
        plt.title(f"{metric_name.capitalize()} Distribution")
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f"{metric_name}_boxplot.png"))
        plt.close()
    
    def plot_scatter(self, metrics: Dict[str, Any], x_metric: str, y_metric: str) -> None:
        """绘制散点图。

        Args:
            metrics: 指标数据
            x_metric: X轴指标名称
            y_metric: Y轴指标名称
        """
        plt.figure(figsize=(10, 6))
        for model, data in metrics.items():
            if isinstance(data[x_metric], list) and isinstance(data[y_metric], list):
                plt.scatter(data[x_metric], data[y_metric], label=model)
        
        plt.xlabel(x_metric.capitalize())
        plt.ylabel(y_metric.capitalize())
        plt.title(f"{x_metric.capitalize()} vs {y_metric.capitalize()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f"{x_metric}_vs_{y_metric}_scatter.png"))
        plt.close()
    
    def plot_heatmap(self, metrics: Dict[str, Any]) -> None:
        """绘制热力图。

        Args:
            metrics: 指标数据
        """
        plt.figure(figsize=(10, 8))
        models = list(metrics.keys())
        metric_names = ["throughput", "latency", "energy", "runtime"]
        data = np.zeros((len(models), len(metric_names)))
        
        for i, model in enumerate(models):
            for j, metric in enumerate(metric_names):
                if isinstance(metrics[model][metric], list):
                    data[i, j] = np.mean(metrics[model][metric])
                else:
                    data[i, j] = metrics[model][metric]
        
        plt.imshow(data, aspect='auto', cmap='YlOrRd')
        plt.colorbar()
        plt.xticks(range(len(metric_names)), metric_names, rotation=45)
        plt.yticks(range(len(models)), models)
        plt.title("Performance Metrics Heatmap")
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "metrics_heatmap.png"))
        plt.close()
    
    def generate_report(self, metrics: Dict[str, Any], tradeoff_results: Dict[str, Any],
                       output_format: str = "html", template: str = None) -> None:
        """生成报告。

        Args:
            metrics: 性能指标数据
            tradeoff_results: 权衡分析结果
            output_format: 输出格式，支持 "html"、"pdf" 和 "csv"
            template: 自定义HTML模板

        Raises:
            ValueError: 当指标数据为空或格式不正确时抛出
        """
        try:
            # 首先验证指标数据是否为空
            if not metrics:
                raise ValueError("指标数据不能为空")

            # 验证数据
            self.validate_metrics(metrics)
            self._validate_tradeoff_results(tradeoff_results)
            
            # 生成图表
            self._generate_plots(metrics, tradeoff_results)
            
            # 根据输出格式生成报告
            if output_format == "html":
                self._generate_html_report(metrics, tradeoff_results, template)
            elif output_format == "pdf":
                self._generate_pdf_report(metrics, tradeoff_results)
            elif output_format == "csv":
                self._generate_csv_report(metrics)
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")
            
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
    
    def _generate_html_report(self, benchmark_results: dict, tradeoff_results: dict, template: str = None) -> None:
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
