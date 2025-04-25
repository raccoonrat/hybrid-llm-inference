# hybrid-llm-inference/src/benchmarking/report_generator.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from toolbox.logger import get_logger

logger = get_logger(__name__)

class ReportGenerator:
    """报告生成器类。"""
    
    def __init__(self, output_dir: Path) -> None:
        """
        初始化报告生成器。
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("报告生成器初始化完成")
    
    def generate_report(self, benchmark_results: Dict[str, Any], tradeoff_results: Optional[Dict[str, Any]] = None) -> None:
        """
        生成基准测试报告。
        
        Args:
            benchmark_results: 基准测试结果
            tradeoff_results: 权衡分析结果
            
        Raises:
            ValueError: 当基准测试结果为空或权衡结果无效时抛出
        """
        if not benchmark_results:
            raise ValueError("基准测试结果不能为空")
        
        if tradeoff_results:
            self._validate_tradeoff_results(tradeoff_results)
        
        try:
            self._generate_summary(benchmark_results)
            self._generate_plots(benchmark_results, tradeoff_results)
            logger.info("报告生成完成")
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            raise
    
    def _validate_tradeoff_results(self, tradeoff_results: Dict[str, Any]) -> None:
        """
        验证权衡结果。
        
        Args:
            tradeoff_results: 权衡分析结果
            
        Raises:
            ValueError: 当权衡结果无效时抛出
        """
        for weight, metrics in tradeoff_results.items():
            if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
                raise ValueError("权重必须在0到1之间")
            
            if not isinstance(metrics, dict):
                raise ValueError("指标必须是字典类型")
            
            if "energy" not in metrics or "runtime" not in metrics:
                raise ValueError("指标必须包含energy和runtime")
            
            if not isinstance(metrics["energy"], (int, float)) or metrics["energy"] < 0:
                raise ValueError("能量必须是非负数")
            
            if not isinstance(metrics["runtime"], (int, float)) or metrics["runtime"] < 0:
                raise ValueError("运行时间必须是非负数")
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """
        生成摘要报告。
        
        Args:
            results: 基准测试结果
        """
        try:
            summary = {}
            for model, data in results.items():
                if "summary" in data:
                    summary[model] = data["summary"]
                else:
                    # 如果没有摘要，从指标中计算
                    metrics = data.get("metrics", [])
                    if metrics:
                        summary[model] = {
                            "avg_energy": np.mean([m["energy"] for m in metrics]),
                            "avg_runtime": np.mean([m["runtime"] for m in metrics]),
                            "avg_throughput": np.mean([m["throughput"] for m in metrics]),
                            "avg_energy_per_token": np.mean([m["energy_per_token"] for m in metrics])
                        }
            
            with open(self.output_dir / "benchmark_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.error(f"生成摘要失败: {str(e)}")
            raise
    
    def _generate_plots(self, results: Dict[str, Any], tradeoff_results: Optional[Dict[str, Any]] = None) -> None:
        """
        生成图表。
        
        Args:
            results: 基准测试结果
            tradeoff_results: 权衡分析结果
        """
        try:
            # 生成能量每令牌图
            self._plot_energy_per_token(results)
            
            # 生成运行时间图
            self._plot_runtime(results)
            
            # 如果有权衡分析结果，生成权衡曲线
            if tradeoff_results:
                self._plot_tradeoff_curve(tradeoff_results)
        except Exception as e:
            logger.error(f"生成图表失败: {str(e)}")
            raise
    
    def _plot_energy_per_token(self, results: Dict[str, Any]) -> None:
        """
        生成能量每令牌图。
        
        Args:
            results: 基准测试结果
        """
        models = list(results.keys())
        energy_per_token = [results[model]["summary"]["avg_energy_per_token"] for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, energy_per_token)
        plt.xlabel("Model")
        plt.ylabel("Energy per Token (J/token)")
        plt.title("Energy per Token Comparison")
        plt.savefig(self.output_dir / "energy_per_token.png")
        plt.close()
    
    def _plot_runtime(self, results: Dict[str, Any]) -> None:
        """
        生成运行时间图。
        
        Args:
            results: 基准测试结果
        """
        models = list(results.keys())
        runtime = [results[model]["summary"]["avg_runtime"] for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, runtime)
        plt.xlabel("Model")
        plt.ylabel("Runtime (s)")
        plt.title("Runtime Comparison")
        plt.savefig(self.output_dir / "runtime.png")
        plt.close()
    
    def _plot_tradeoff_curve(self, tradeoff_results: Dict[str, Any]) -> None:
        """
        生成权衡曲线。
        
        Args:
            tradeoff_results: 权衡分析结果
        """
        weights = list(tradeoff_results.keys())
        energy = [tradeoff_results[w]["energy"] for w in weights]
        runtime = [tradeoff_results[w]["runtime"] for w in weights]
        
        plt.figure(figsize=(10, 6))
        plt.plot(weights, energy, label="Energy")
        plt.plot(weights, runtime, label="Runtime")
        plt.xlabel("Weight")
        plt.ylabel("Value")
        plt.title("Energy-Runtime Tradeoff Curve")
        plt.legend()
        plt.savefig(self.output_dir / "tradeoff_curve.png")
        plt.close()
