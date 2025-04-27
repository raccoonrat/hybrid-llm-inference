# hybrid-llm-inference/src/benchmarking/report_generator.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import numpy as np
from toolbox.logger import get_logger
import logging
import pandas as pd
from datetime import datetime
import csv
import seaborn as sns

logger = get_logger(__name__)

class ReportGenerator:
    """基准测试报告生成器。"""

    def __init__(self, output_dir: str, style_config: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None):
        """初始化报告生成器。

        Args:
            output_dir: 输出目录路径
            style_config: 可选的样式配置
            options: 可选的其他配置选项
        """
        self.output_dir = output_dir
        self.style_config = style_config or {}
        self.options = options or {}
        self.logger = logging.getLogger(__name__)
        self.output_format = "json"  # 默认输出格式
        os.makedirs(output_dir, exist_ok=True)
        self.visualization_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.visualization_dir, exist_ok=True)

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """验证基准测试数据。

        Args:
            data: 基准测试数据

        Raises:
            ValueError: 当数据无效时抛出
            TypeError: 当数据类型错误时抛出
        """
        # 验证数据类型
        if not isinstance(data, dict):
            raise TypeError("基准测试数据必须是字典类型")
            
        if not data:
            raise ValueError("基准测试结果不能为空")
        
        # 如果数据已经包含metrics字段，直接使用
        if "metrics" in data:
            metrics = data["metrics"]
        else:
            # 否则，将整个数据作为metrics
            metrics = data
            data = {"metrics": metrics}
        
        # 验证metrics字段
        if not isinstance(metrics, dict):
            raise ValueError("metrics 必须是字典类型")
        
        # 检查是否有任何指标数据
        if not metrics:
            raise ValueError("metrics 不能为空")
        
        # 添加时间戳（如果不存在）
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        def validate_metric_value(value):
            """递归验证指标值。"""
            if isinstance(value, (int, float)):
                return True
            elif isinstance(value, list):
                if not value:
                    return False
                return all(isinstance(x, (int, float)) for x in value)
            elif isinstance(value, dict):
                # 跳过 summary 字段的验证
                if "summary" in value:
                    return True
                if not value:
                    return False
                return all(validate_metric_value(v) for v in value.values())
            return False

        # 验证指标值类型
        invalid_metrics = []
        for key, value in metrics.items():
            if not validate_metric_value(value):
                invalid_metrics.append(key)
        
        if invalid_metrics:
            raise ValueError(f"以下指标的值类型无效: {', '.join(invalid_metrics)}")
        
        # 验证必需的指标
        required_metrics = ['latency', 'throughput']  # 要求 latency 和 throughput 都是必需的
        missing_metrics = [metric for metric in required_metrics if metric not in metrics]
        if missing_metrics:
            raise ValueError(f"缺少必需的指标: {', '.join(missing_metrics)}")
            
        # 确保 throughput 是正数
        throughput = metrics['throughput']
        if isinstance(throughput, (int, float)) and throughput <= 0:
            raise ValueError("throughput 必须是正数")
        elif isinstance(throughput, list) and any(x <= 0 for x in throughput):
            raise ValueError("throughput 列表中的所有值必须是正数")

    def _generate_visualizations(self, data: Dict[str, Any], include_tradeoff: bool = True) -> List[str]:
        """生成基准测试结果的可视化图表。

        Args:
            data: 基准测试数据
            include_tradeoff: 是否包含权衡分析图表

        Returns:
            生成的图表文件路径列表
        """
        chart_files = []
        metrics = data['metrics']
        
        # 使用默认样式
        plt.style.use('default')
        
        # 创建输出目录
        output_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        plt.figure(figsize=self.style_config.get('figsize', (10, 6)))
        
        # 生成延迟分布图（如果存在）
        if 'latency' in metrics:
            plt.clf()
            latency_data = metrics['latency']
            if isinstance(latency_data, dict) and "distribution" in latency_data:
                sns.histplot(latency_data["distribution"], kde=True)
            elif isinstance(latency_data, list):
                sns.histplot(latency_data, kde=True)
            elif isinstance(latency_data, (int, float)):
                plt.bar(['latency'], [latency_data])
            plt.title('Latency Distribution')
            plt.xlabel('Latency (ms)')
            plt.ylabel('Frequency')
            latency_plot = os.path.join(output_dir, 'latency_distribution.png')
            plt.savefig(latency_plot)
            chart_files.append(latency_plot)
        
        # 生成吞吐量分布图（如果存在）
        if 'throughput' in metrics:
            plt.clf()
            throughput_data = metrics['throughput']
            if isinstance(throughput_data, dict) and "distribution" in throughput_data:
                sns.histplot(throughput_data["distribution"], kde=True)
            elif isinstance(throughput_data, list):
                sns.histplot(throughput_data, kde=True)
            elif isinstance(throughput_data, (int, float)):
                plt.bar(['throughput'], [throughput_data])
            plt.title('Throughput Distribution')
            plt.xlabel('Throughput (requests/second)')
            plt.ylabel('Frequency')
            throughput_plot = os.path.join(output_dir, 'throughput_distribution.png')
            plt.savefig(throughput_plot)
            chart_files.append(throughput_plot)
        
        # 生成能耗分布图（如果存在）
        if 'energy' in metrics:
            plt.clf()
            energy_data = metrics['energy']
            if isinstance(energy_data, dict) and "distribution" in energy_data:
                sns.histplot(energy_data["distribution"], kde=True)
            elif isinstance(energy_data, list):
                sns.histplot(energy_data, kde=True)
            elif isinstance(energy_data, (int, float)):
                plt.bar(['energy'], [energy_data])
            plt.title('Energy Distribution')
            plt.xlabel('Energy (J)')
            plt.ylabel('Frequency')
            energy_plot = os.path.join(output_dir, 'energy_distribution.png')
            plt.savefig(energy_plot)
            chart_files.append(energy_plot)
        
        # 生成其他指标的分布图
        for key, value in metrics.items():
            if key not in ['latency', 'energy', 'throughput']:
                plt.clf()
                if isinstance(value, dict):
                    if "distribution" in value:
                        sns.histplot(value["distribution"], kde=True)
                    elif "value" in value:
                        plt.bar([key], [value["value"]])
                elif isinstance(value, list):
                    sns.histplot(value, kde=True)
                elif isinstance(value, (int, float)):
                    plt.bar([key], [value])
                plt.title(f'{key.capitalize()} Distribution')
                plt.xlabel(key.capitalize())
                plt.ylabel('Frequency')
                plot_path = os.path.join(output_dir, f'{key}_distribution.png')
                plt.savefig(plot_path)
                chart_files.append(plot_path)
        
        # 生成权衡图（如果存在相关指标）
        if include_tradeoff:
            # 延迟-吞吐量权衡
            if 'latency' in metrics and 'throughput' in metrics:
                plt.clf()
                latency_data = metrics['latency']
                throughput_data = metrics['throughput']
                
                if isinstance(latency_data, dict) and isinstance(throughput_data, dict):
                    if "distribution" in latency_data and "distribution" in throughput_data:
                        plt.scatter(latency_data["distribution"], throughput_data["distribution"])
                    elif "value" in latency_data and "value" in throughput_data:
                        plt.scatter([latency_data["value"]], [throughput_data["value"]])
                elif isinstance(latency_data, list) and isinstance(throughput_data, list):
                    plt.scatter(latency_data, throughput_data)
                elif isinstance(latency_data, (int, float)) and isinstance(throughput_data, (int, float)):
                    plt.scatter([latency_data], [throughput_data])
                
                plt.title('Latency-Throughput Tradeoff')
                plt.xlabel('Latency (ms)')
                plt.ylabel('Throughput (requests/second)')
                tradeoff_plot = os.path.join(output_dir, 'latency_throughput_tradeoff.png')
                plt.savefig(tradeoff_plot)
                chart_files.append(tradeoff_plot)
            
            # 延迟-能耗权衡
            if 'latency' in metrics and 'energy' in metrics:
                plt.clf()
                latency_data = metrics['latency']
                energy_data = metrics['energy']
                
                if isinstance(latency_data, dict) and isinstance(energy_data, dict):
                    if "distribution" in latency_data and "distribution" in energy_data:
                        plt.scatter(latency_data["distribution"], energy_data["distribution"])
                    elif "value" in latency_data and "value" in energy_data:
                        plt.scatter([latency_data["value"]], [energy_data["value"]])
                elif isinstance(latency_data, list) and isinstance(energy_data, list):
                    plt.scatter(latency_data, energy_data)
                elif isinstance(latency_data, (int, float)) and isinstance(energy_data, (int, float)):
                    plt.scatter([latency_data], [energy_data])
                
                plt.title('Latency-Energy Tradeoff')
                plt.xlabel('Latency (ms)')
                plt.ylabel('Energy (J)')
                tradeoff_plot = os.path.join(output_dir, 'latency_energy_tradeoff.png')
                plt.savefig(tradeoff_plot)
                chart_files.append(tradeoff_plot)
        
        plt.close('all')
        return chart_files

    def generate_report(self, data: Dict[str, Any], output_format: str = "json", include_visualizations: bool = True) -> str:
        """生成基准测试报告。

        Args:
            data: 基准测试数据
            output_format: 输出格式，支持 "json"、"html"、"csv" 和 "markdown"
            include_visualizations: 是否包含可视化图表

        Returns:
            报告文件路径

        Raises:
            ValueError: 当输入数据无效或输出格式不支持时
        """
        # 验证输入数据
        self._validate_data(data)
        
        # 生成可视化图表
        chart_files = []
        if include_visualizations:
            try:
                chart_files = self._generate_visualizations(data)
                # 生成权衡曲线
                curve_path = os.path.join(self.output_dir, "tradeoff_curve.png")
                self._plot_tradeoff_curve(data, curve_path)
            except Exception as e:
                logger.warning(f"生成可视化图表时出错: {str(e)}")

        # 根据输出格式生成报告
        if output_format == "json":
            report_path = os.path.join(self.output_dir, "report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                # 添加图表路径到数据中
                if chart_files:
                    data["visualizations"] = [os.path.basename(f) for f in chart_files]
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif output_format == "html":
            report_path = os.path.join(self.output_dir, "report.html")
            html_content = self._generate_html_report(data)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        elif output_format == "csv":
            report_path = os.path.join(self.output_dir, "report.csv")
            # 将数据扁平化为CSV格式
            df = pd.DataFrame()
            if "metrics" in data:
                metrics = data["metrics"]
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        df[key] = [value]
                    elif isinstance(value, list):
                        df[key] = [",".join(map(str, value))]
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            df[f"{key}_{sub_key}"] = [sub_value]
            df.to_csv(report_path, index=False)
        elif output_format == "markdown":
            report_path = os.path.join(self.output_dir, "report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(self._generate_markdown(data))
        else:
            raise ValueError("不支持的报告格式")
        
        return report_path
    
    def generate_tradeoff_curve(self, data: Dict[str, Any]) -> str:
        """生成权衡曲线图。

        Args:
            data: 基准测试数据

        Returns:
            图片文件路径
        """
        plot_path = os.path.join(self.output_dir, "tradeoff_curve.png")
        # 生成权衡曲线图
        plt.figure(figsize=(10, 6))
        plt.plot(data["latency"], data["energy"], "o-")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Energy (J)")
        plt.title("Latency-Energy Tradeoff Curve")
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

    def _generate_markdown(self, results: Dict[str, Any]) -> str:
        """生成 Markdown 格式的报告。"""
        md_content = ["# Benchmark Test Report\n"]
        
        # 添加时间戳
        md_content.append(f"Generated Time: {results['timestamp']}\n")
        
        # 添加指标摘要
        if "metrics" in results:
            md_content.append("## Performance Metrics\n")
            metrics = results["metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    md_content.append(f"- {key}: {value}\n")
                elif isinstance(value, list):
                    avg_value = sum(value) / len(value) if value else 0
                    md_content.append(f"- {key} (Average): {avg_value:.2f}\n")
                elif isinstance(value, dict):
                    md_content.append(f"\n### {key}\n")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            md_content.append(f"- {sub_key}: {sub_value}\n")

        # 添加图表引用
        if results.get("charts_generated"):
            md_content.append("\n## Visualizations\n")
            if 'latency' in results['metrics']:
                md_content.append("![Latency Distribution](latency_distribution.png)\n")
            if 'energy' in results['metrics']:
                md_content.append("![Energy Distribution](energy_distribution.png)\n")
            if 'latency' in results['metrics'] and 'energy' in results['metrics']:
                md_content.append("![Latency-Energy Tradeoff](latency_energy_tradeoff.png)\n")

        return "\n".join(md_content)

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
    
    def plot_time_series(self, data: List[float], title: str, xlabel: str, ylabel: str, output_path: str):
        """绘制时间序列图。

        Args:
            data: 时间序列数据
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            output_path: 输出文件路径
        """
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
    
    def plot_boxplot(self, data: Dict[str, List[float]], title: str, xlabel: str, ylabel: str, output_path: str):
        """绘制箱线图。

        Args:
            data: 箱线图数据，键为类别，值为数值列表
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            output_path: 输出文件路径
        """
        plt.figure(figsize=(10, 6))
        plt.boxplot(data.values(), labels=data.keys())
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(output_path)
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
                    value = metrics[model][metric]
                    if isinstance(value, dict):
                        # 跳过字典类型的值
                        continue
                    elif isinstance(value, list):
                        data[i, j] = np.mean(value)
                    else:
                        data[i, j] = float(value)
        
        plt.imshow(data, aspect='auto', cmap='YlOrRd')
        plt.colorbar()
        plt.xticks(range(len(metric_names)), metric_names, rotation=45)
        plt.yticks(range(len(models)), models)
        plt.title("Performance Metrics Heatmap")
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "metrics_heatmap.png"))
        plt.close()
    
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
            plt.title("Throughput-Latency Tradeoff Curve")
            plt.xlabel("Throughput (requests/second)")
            plt.ylabel("Latency (seconds)")
            plt.grid(True)
            
            # 保存图表
            plt.savefig(output_path)
            plt.close()
        except Exception as e:
            self.logger.error(f"绘制权衡曲线时发生错误: {str(e)}")
            raise

    def _generate_time_series_plot(self, metric_name: str, metric_values: List[float], output_path: str) -> None:
        """生成时间序列图。

        Args:
            metric_name: 指标名称
            metric_values: 指标值列表
            output_path: 输出文件路径
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(metric_values)), metric_values, marker='o')
        plt.title(f"{metric_name} Time Series")
        plt.xlabel("Time Point")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

    def _generate_csv_report(self, data: Dict[str, Any], output_path: str) -> str:
        """生成 CSV 格式的报告。

        Args:
            data: 要导出的数据
            output_path: 输出文件路径

        Returns:
            str: CSV 文件的路径
        """
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding="utf-8")
        return output_path

    def _generate_system_report(self, data, report_path, output_format):
        """生成系统基准测试报告。"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Test Report\n\n")
            
            # 写入基本指标
            f.write("## Basic Metrics\n\n")
            for key, value in data["metrics"].items():
                f.write(f"- {key}: {value:.2f}\n")
                
            # 写入调度指标
            f.write("\n## Scheduling Metrics\n\n")
            for key, value in data["scheduling_metrics"].items():
                f.write(f"- {key}: {value}\n")
                
            # 生成权衡曲线
            curve_path = os.path.join(os.path.dirname(report_path), "tradeoff_curve.png")
            self._plot_tradeoff_curve(data, curve_path)
            f.write(f"\n## Tradeoff Curve\n\n![Tradeoff Curve]({os.path.basename(curve_path)})\n")

    def _generate_model_report(self, data, report_path, output_format):
        """生成模型基准测试报告。"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Model Benchmark Test Report\n\n")
            
            for model, metrics in data.items():
                f.write(f"## {model}\n\n")
                
                # 写入基本指标
                f.write("### Performance Metrics\n\n")
                for key in ["throughput", "latency", "energy", "runtime"]:
                    f.write(f"- {key}: {metrics[key]:.2f}\n")
                
                # 写入汇总指标
                f.write("\n### Summary Metrics\n\n")
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
            plt.title(f"{metric.capitalize()} Comparison")
            plt.xlabel("Model")
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

    def _get_report_path(self, report_type: str) -> str:
        """获取报告文件路径。

        Args:
            report_type: 报告类型

        Returns:
            str: 报告文件路径
        """
        return os.path.join(self.output_dir, f"{report_type}_report.json")

    def cleanup(self) -> None:
        """清理报告生成器的资源。"""
        # 清理所有生成的图表文件
        for file in os.listdir(self.output_dir):
            if file.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                os.remove(os.path.join(self.output_dir, file))
