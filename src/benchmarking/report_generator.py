# hybrid-llm-inference/src/benchmarking/report_generator.py
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from toolbox.logger import get_logger

class ReportGenerator:
    def __init__(self, output_dir="data/benchmarks"):
        """
        Initialize ReportGenerator for creating benchmark reports and visualizations.
        
        Args:
            output_dir (str): Directory to save reports and plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
    
    def generate_report(self, benchmark_results, tradeoff_results=None):
        """
        Generate benchmark report and visualizations.
        
        Args:
            benchmark_results (dict): Results from SystemBenchmarking {strategy: {"metrics": list, "summary": dict}}.
            tradeoff_results (dict): Optional tradeoff results from TradeoffAnalyzer {lambda: {"energy": float, "runtime": float}}.
            
        Raises:
            ValueError: 如果基准测试结果为空或权衡结果无效。
        """
        # 验证基准测试结果
        if not benchmark_results:
            self.logger.error("基准测试结果为空")
            raise ValueError("Benchmark results are empty")
        
        # 验证权衡结果
        if tradeoff_results:
            for lambda_val, metrics in tradeoff_results.items():
                if metrics["energy"] < 0 or metrics["runtime"] < 0:
                    self.logger.error(f"无效的权衡结果: lambda={lambda_val}, metrics={metrics}")
                    raise ValueError("Invalid tradeoff results")
        
        # Save summary report
        summary = {strategy: res["summary"] for strategy, res in benchmark_results.items()}
        report_path = self.output_dir / "benchmark_summary.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Saved benchmark summary to {report_path}")
        
        # Generate visualizations
        self._plot_energy_per_token(benchmark_results)
        self._plot_runtime(benchmark_results)
        if tradeoff_results:
            self._plot_tradeoff(tradeoff_results)
    
    def _plot_energy_per_token(self, benchmark_results):
        """Generate energy per token plot (similar to Figure 1)."""
        plt.figure(figsize=(8, 6))
        for strategy, res in benchmark_results.items():
            metrics = pd.DataFrame(res["metrics"])
            plt.hist(metrics["energy_per_token"], bins=50, alpha=0.5, label=strategy, density=True)
        
        plt.xlabel("Energy per Token (Joules/token)")
        plt.ylabel("Density")
        plt.title("Energy per Token Distribution")
        plt.legend()
        plt.grid(True)
        
        plot_path = self.output_dir / "energy_per_token.png"
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved energy per token plot to {plot_path}")
    
    def _plot_runtime(self, benchmark_results):
        """Generate runtime plot (similar to Figure 2)."""
        plt.figure(figsize=(8, 6))
        for strategy, res in benchmark_results.items():
            metrics = pd.DataFrame(res["metrics"])
            plt.hist(metrics["runtime"], bins=50, alpha=0.5, label=strategy, density=True)
        
        plt.xlabel("Runtime (seconds)")
        plt.ylabel("Density")
        plt.title("Runtime Distribution")
        plt.legend()
        plt.grid(True)
        
        plot_path = self.output_dir / "runtime.png"
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved runtime plot to {plot_path}")
    
    def _plot_tradeoff(self, tradeoff_results):
        """Generate energy-runtime tradeoff plot (similar to Figure 4)."""
        lambdas = list(tradeoff_results.keys())
        energies = [tradeoff_results[l]["energy"] for l in lambdas]
        runtimes = [tradeoff_results[l]["runtime"] for l in lambdas]
        
        plt.figure(figsize=(8, 6))
        plt.plot(runtimes, energies, marker='o')
        for i, l in enumerate(lambdas):
            plt.annotate(f"λ={l:.1f}", (runtimes[i], energies[i]))
        plt.xlabel("Average Runtime (seconds)")
        plt.ylabel("Average Energy (Joules)")
        plt.title("Energy-Runtime Tradeoff")
        plt.grid(True)
        
        plot_path = self.output_dir / "tradeoff_curve.png"
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved tradeoff curve to {plot_path}")
