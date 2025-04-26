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
import pandas as pd
from datetime import datetime

logger = get_logger(__name__)

class ReportGenerator:
    """报告生成器类。"""
    
    def __init__(self, output_dir: str, style_config: Dict[str, Any] = None, output_format: str = "json") -> None:
        """初始化报告生成器。

        Args:
            output_dir: 输出目录
            style_config: 图表样式配置
            output_format: 输出格式，支持 "json", "csv", "txt", "markdown"
        """
        self.output_dir = output_dir
        self.logger = get_logger(__name__)
        self.output_format = output_format
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
        
        # 获取指标值
        metric_value = metrics.get(metric_name)
        
        # 处理不同类型的指标值
        if isinstance(metric_value, (int, float)):
            # 单个值的情况
            plt.plot([0], [metric_value], marker='o', label=metric_name)
        elif isinstance(metric_value, list):
            # 列表值的情况
            x = range(len(metric_value))
            plt.plot(x, metric_value, label=metric_name)
        elif isinstance(metric_value, dict):
            # 字典值的情况，绘制每个子指标
            for key, values in metric_value.items():
                if isinstance(values, list):
                    x = range(len(values))
                    plt.plot(x, values, label=f"{metric_name}_{key}")
                elif isinstance(values, (int, float)):
                    plt.plot([0], [values], marker='o', label=f"{metric_name}_{key}")
        
        plt.title(f"{metric_name} 时间序列")
        plt.xlabel("时间点")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, f"{metric_name}_time_series.png")
        plt.savefig(output_path)
        plt.close()  # 关闭图表以释放内存
    
    def plot_boxplot(self, metrics: Dict[str, Any], metric_name: str) -> None:
        """绘制箱线图。

        Args:
            metrics: 指标数据
            metric_name: 指标名称
        """
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        
        # 处理单个值的情况
        if isinstance(metrics, dict):
            if metric_name in metrics:
                if isinstance(metrics[metric_name], list):
                    data.append(metrics[metric_name])
                    labels.append(metric_name)
                else:
                    data.append([metrics[metric_name]])
                    labels.append(metric_name)
            else:
                for model, model_data in metrics.items():
                    if isinstance(model_data, dict) and metric_name in model_data:
                        if isinstance(model_data[metric_name], list):
                            data.append(model_data[metric_name])
                        else:
                            data.append([model_data[metric_name]])
                        labels.append(model)
        
        if data:  # 只在有数据时绘图
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
        
        # 处理单个值的情况
        if isinstance(metrics, dict):
            if x_metric in metrics and y_metric in metrics:
                if isinstance(metrics[x_metric], (int, float)) and isinstance(metrics[y_metric], (int, float)):
                    plt.scatter([metrics[x_metric]], [metrics[y_metric]], label="single point")
                elif isinstance(metrics[x_metric], list) and isinstance(metrics[y_metric], list):
                    plt.scatter(metrics[x_metric], metrics[y_metric], label="time series")
            else:
                for model, data in metrics.items():
                    if isinstance(data, dict) and x_metric in data and y_metric in data:
                        if isinstance(data[x_metric], (int, float)) and isinstance(data[y_metric], (int, float)):
                            plt.scatter([data[x_metric]], [data[y_metric]], label=model)
                        elif isinstance(data[x_metric], list) and isinstance(data[y_metric], list):
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
        
        # 如果metrics是单层字典，将其转换为双层字典
        if all(isinstance(v, (int, float)) for v in metrics.values()):
            metrics = {"default": metrics}
        
        models = list(metrics.keys())
        metric_names = list(set().union(*[set(m.keys()) for m in metrics.values()]))
        data = np.zeros((len(models), len(metric_names)))
        
        for i, model in enumerate(models):
            for j, metric in enumerate(metric_names):
                if metric in metrics[model]:
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
            raise ValueError("基准测试数据不能为空")

        # 检查是否是系统基准测试数据格式
        if "metrics" in data:
            if not isinstance(data["metrics"], dict):
                raise ValueError("metrics必须是字典类型")
            return

        # 检查是否是模型基准测试数据格式
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for model, metrics in data.items():
                if not all(k in metrics for k in ["throughput", "latency", "energy", "runtime"]):
                    raise ValueError(f"模型 {model} 缺少必要的性能指标")
            return

        raise ValueError("无效的基准测试数据格式")

    def _plot_tradeoff_curve(self, data, output_path):
        """绘制权衡曲线。

        Args:
            data: 基准测试数据
            output_path: 输出文件路径
        """
        try:
            # 提取数据
            if "parallel_metrics" in data:
                throughputs = [m["throughput"] for m in data["parallel_metrics"]]
                latencies = [m["latency"] for m in data["parallel_metrics"]]
            else:
                # 如果没有并行指标，使用单个点
                throughputs = [data["metrics"]["throughput"]]
                latencies = [data["metrics"]["latency"]]
            
            if not throughputs or not latencies:
                raise ValueError("没有足够的数据来绘制权衡曲线")
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.scatter(throughputs, latencies, alpha=0.6)
            
            # 添加趋势线（如果有多个点）
            if len(throughputs) > 1:
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

    def generate_report(self, data: dict, format: str = None, include_visualizations: bool = False, template: str = None) -> str:
        """生成基准测试报告。

        Args:
            data: 基准测试数据
            format: 输出格式，支持 "json", "csv", "txt", "markdown"
            include_visualizations: 是否包含可视化图表
            template: 报告模板，可选

        Returns:
            生成的报告文件路径
        """
        if not data:
            raise ValueError("基准测试结果不能为空")
            
        # 使用指定的格式或默认格式
        output_format = format or self.output_format
        
        # 验证数据格式
        self._validate_data(data)
        
        # 生成报告文件路径
        report_path = os.path.join(self.output_dir, f"benchmark_report.{output_format}")
        
        # 生成权衡曲线（如果有并行指标）
        if "parallel_metrics" in data:
            curve_path = os.path.join(self.output_dir, "tradeoff_curve.png")
            self._plot_tradeoff_curve(data, curve_path)
        
        # 生成可视化图表（如果需要）
        if include_visualizations:
            self._generate_visualizations(data)
        
        # 根据格式生成报告
        if output_format == "json":
            self._generate_json_report(data, report_path)
        elif output_format == "csv":
            self._generate_csv_report(data, report_path)
        elif output_format == "txt":
            self._generate_text_report(data, report_path)
        elif output_format == "markdown":
            self._generate_markdown_report(data, report_path, include_visualizations)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
            
        return report_path
        
    def export_raw_data(self, data: dict, format: str = "json") -> str:
        """导出原始数据。

        Args:
            data: 要导出的数据
            format: 输出格式，支持 "json", "csv"

        Returns:
            导出的数据文件路径
        """
        if not data:
            raise ValueError("导出数据不能为空")
            
        # 生成导出文件路径
        export_path = os.path.join(self.output_dir, f"raw_data.{format}")
        
        # 根据格式导出数据
        if format == "json":
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "csv":
            # 将数据转换为DataFrame格式
            df = pd.DataFrame(data)
            df.to_csv(export_path, index=False, encoding="utf-8")
        else:
            raise ValueError(f"不支持的导出格式: {format}")
            
        return export_path
        
    def _generate_text_report(self, data: dict, report_path: str) -> None:
        """生成文本格式报告。

        Args:
            data: 基准测试数据
            report_path: 报告文件路径
        """
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("基准测试报告\n")
            f.write("============\n\n")
            
            # 写入系统信息
            if "system_info" in data:
                f.write("系统信息\n")
                f.write("--------\n")
                for key, value in data["system_info"].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
            # 写入模型信息
            if "model_info" in data:
                f.write("模型信息\n")
                f.write("--------\n")
                for key, value in data["model_info"].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
            # 写入指标数据
            if "metrics" in data:
                f.write("性能指标\n")
                f.write("--------\n")
                for metric, value in data["metrics"].items():
                    f.write(f"{metric}: {value}\n")
                f.write("\n")
                
            # 写入调度统计
            if "scheduler_stats" in data:
                f.write("调度统计\n")
                f.write("--------\n")
                for stat, value in data["scheduler_stats"].items():
                    f.write(f"{stat}: {value}\n")
                    
    def _generate_visualizations(self, data: dict) -> None:
        """生成可视化图表。

        Args:
            data: 基准测试数据
        """
        if "metrics" in data:
            metrics = data["metrics"]
            
            # 生成时间序列图
            for metric in ["throughput", "latency", "memory_usage", "power_usage"]:
                if metric in metrics:
                    self.plot_time_series(metrics, metric)
                    
            # 生成箱线图
            for metric in ["throughput", "latency", "memory_usage", "power_usage"]:
                if metric in metrics:
                    self.plot_boxplot(metrics, metric)
                    
            # 生成散点图（如果有多个指标）
            if all(m in metrics for m in ["latency", "throughput"]):
                self.plot_scatter(metrics, "latency", "throughput")
            if all(m in metrics for m in ["power_usage", "throughput"]):
                self.plot_scatter(metrics, "power_usage", "throughput")
            
            # 生成热力图（如果有足够的指标）
            if len(metrics) >= 2:
                self.plot_heatmap(metrics)

    def _generate_markdown_report(self, data, report_path, include_visualizations):
        """生成Markdown格式的基准测试报告。"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 基准测试报告\n\n")
            
            # 写入基本指标
            f.write("## 基本指标\n\n")
            for key, value in data["metrics"].items():
                f.write(f"- {key}: {value:.2f}\n")
                
            # 写入调度指标（如果存在）
            if "scheduling_metrics" in data:
                f.write("\n## 调度指标\n\n")
                for key, value in data["scheduling_metrics"].items():
                    f.write(f"- {key}: {value}\n")
                
            # 生成权衡曲线（如果有并行指标）
            if "parallel_metrics" in data:
                curve_path = os.path.join(os.path.dirname(report_path), "tradeoff_curve.png")
                self._plot_tradeoff_curve(data, curve_path)
                f.write(f"\n## 权衡曲线\n\n![权衡曲线]({os.path.basename(curve_path)})\n")
            
            # 生成可视化图表（如果需要）
            if include_visualizations:
                f.write("\n## 可视化图表\n\n")
                metrics = data["metrics"]
                for metric in ["throughput", "latency", "memory_usage", "power_usage"]:
                    if metric in metrics:
                        chart_path = f"{metric}_time_series.png"
                        f.write(f"\n### {metric.capitalize()} 时间序列\n\n")
                        f.write(f"![{metric} 时间序列]({chart_path})\n")

    def _generate_json_report(self, data, report_path):
        """生成JSON格式的基准测试报告。"""
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _generate_csv_report(self, data, report_path):
        """生成CSV格式的基准测试报告。"""
        # 将数据转换为适合 CSV 的格式
        df = pd.DataFrame(data)
        df.to_csv(report_path, index=False)

    def _generate_system_report(self, data, report_path, output_format):
        """生成系统基准测试报告。"""
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
            curve_path = os.path.join(os.path.dirname(report_path), "tradeoff_curve.png")
            self._plot_tradeoff_curve(data, curve_path)
            f.write(f"\n## 权衡曲线\n\n![权衡曲线]({os.path.basename(curve_path)})\n")

    def _generate_model_report(self, data, report_path, output_format):
        """生成模型基准测试报告。"""
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
            self._plot_model_comparison(data, os.path.dirname(report_path))

    def _plot_model_comparison(self, data, output_dir):
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
            
            plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"))
            plt.close()

    def export_summary(self, data: dict, output_path: str) -> str:
        """导出基准测试数据的摘要。

        Args:
            data: 基准测试数据
            output_path: 输出文件路径

        Returns:
            导出的文件路径
        """
        self._validate_data(data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 根据文件扩展名选择输出格式
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif ext == ".csv":
            # 将数据转换为DataFrame格式
            if "metrics" in data:
                df = pd.DataFrame([data["metrics"]])
            else:
                df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"不支持的输出格式: {ext}")
        
        return output_path
