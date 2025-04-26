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

logger = get_logger(__name__)

class ReportGenerator:
    """报告生成器类。"""
    
    def __init__(self, output_dir: str, style_config: Dict[str, Any] = None) -> None:
        """初始化报告生成器。

        Args:
            output_dir: 输出目录
            style_config: 图表样式配置
        """
        self.output_dir = output_dir
        self.logger = get_logger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        if style_config:
            plt.rcParams.update(style_config)
        
        # 配置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        self.logger.info("报告生成器初始化完成")
    
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
    
    def _validate_data(self, data):
        """验证数据格式。

        Args:
            data (dict): 要验证的数据

        Raises:
            ValueError: 如果数据格式无效
        """
        if not data:
            raise ValueError("指标数据不能为空")
            
        # 检查是否是系统基准测试数据格式
        if all(field in data for field in ["metrics", "parallel_metrics", "scheduling_metrics"]):
            if not isinstance(data["metrics"], dict):
                raise ValueError("metrics必须是字典类型")
            if not isinstance(data["parallel_metrics"], list):
                raise ValueError("parallel_metrics必须是列表类型")
            if not isinstance(data["scheduling_metrics"], dict):
                raise ValueError("scheduling_metrics必须是字典类型")
            return
            
        # 检查是否是模型基准测试数据格式
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for model, metrics in data.items():
                if not all(k in metrics for k in ["throughput", "latency", "energy", "runtime"]):
                    raise ValueError(f"模型 {model} 缺少必要的性能指标")
                if "summary" not in metrics:
                    raise ValueError(f"模型 {model} 缺少summary字段")
            return
            
        raise ValueError("无效的数据格式")

    def _plot_tradeoff_curve(self, data, output_path):
        """绘制权衡曲线。

        Args:
            data (dict): 基准测试数据
            output_path (str): 输出文件路径
        """
        try:
            # 提取数据
            throughputs = [m["throughput"] for m in data["parallel_metrics"]]
            latencies = [m["latency"] for m in data["parallel_metrics"]]
            
            if not throughputs or not latencies:
                raise ValueError("没有足够的数据来绘制权衡曲线")
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.scatter(throughputs, latencies, alpha=0.6)
            
            # 添加趋势线
            z = np.polyfit(throughputs, latencies, 1)
            p = np.poly1d(z)
            plt.plot(throughputs, p(throughputs), "r--", alpha=0.5)
            
            # 设置图表属性
            plt.title("吞吐量-延迟权衡曲线")
            plt.xlabel("吞吐量 (requests/second)")
            plt.ylabel("延迟 (seconds)")
            plt.grid(True)
            
            # 保存图表
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制权衡曲线时发生错误: {str(e)}")
            raise

    def generate_report(self, data: Dict[str, Any], tradeoff_results: Optional[Dict[str, Any]] = None, output_format: str = "markdown") -> str:
        """生成基准测试报告。

        Args:
            data (Dict[str, Any]): 基准测试数据
            tradeoff_results (Optional[Dict[str, Any]]): 权衡分析结果
            output_format (str): 输出格式，支持 "markdown" 或 "pdf"

        Returns:
            str: 生成的报告文件路径
        """
        try:
            # 验证数据
            self._validate_data(data)
            
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 根据数据类型生成不同的报告
            if all(field in data for field in ["metrics", "parallel_metrics", "scheduling_metrics"]):
                return self._generate_system_report(data, output_format)
            else:
                return self._generate_model_report(data, output_format)
        except Exception as e:
            self.logger.error(f"生成报告时发生错误: {str(e)}")
            raise

    def _generate_system_report(self, data, output_format):
        """生成系统基准测试报告。"""
        report_path = os.path.join(self.output_dir, f"benchmark_report.{output_format}")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 基准测试报告\n\n")
            
            # 写入基本指标
            f.write("## 基本指标\n\n")
            for key, value in data["metrics"].items():
                f.write(f"- {key}: {value:.2f}\n")
                
            # 写入调度指标
            f.write("\n## 调度指标\n\n")
            for key, value in data["scheduling_metrics"].items():
                f.write(f"- {key}: {value}\n")
                
            # 生成权衡曲线
            curve_path = os.path.join(self.output_dir, "tradeoff_curve.png")
            self._plot_tradeoff_curve(data, curve_path)
            f.write(f"\n## 权衡曲线\n\n![权衡曲线]({os.path.basename(curve_path)})\n")
            
        return report_path

    def _generate_model_report(self, data, output_format):
        """生成模型基准测试报告。"""
        report_path = os.path.join(self.output_dir, f"model_benchmark_report.{output_format}")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 模型基准测试报告\n\n")
            
            for model, metrics in data.items():
                f.write(f"## {model}\n\n")
                
                # 写入基本指标
                f.write("### 性能指标\n\n")
                for key in ["throughput", "latency", "energy", "runtime"]:
                    f.write(f"- {key}: {metrics[key]:.2f}\n")
                
                # 写入汇总指标
                f.write("\n### 汇总指标\n\n")
                for key, value in metrics["summary"].items():
                    f.write(f"- {key}: {value:.2f}\n")
                
                f.write("\n")
                
            # 生成性能对比图
            self._plot_model_comparison(data)
            
        return report_path

    def _plot_model_comparison(self, data):
        """绘制模型性能对比图。"""
        metrics = ["throughput", "latency", "energy", "runtime"]
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            models = list(data.keys())
            values = [data[model][metric] for model in models]
            
            plt.bar(models, values)
            plt.title(f"{metric.capitalize()} 对比")
            plt.xlabel("模型")
            plt.ylabel(metric.capitalize())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, f"{metric}_comparison.png"))
            plt.close()
