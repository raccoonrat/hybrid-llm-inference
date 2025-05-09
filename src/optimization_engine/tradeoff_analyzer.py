# hybrid-llm-inference/src/optimization_engine/tradeoff_analyzer.py
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.toolbox.logger import get_logger
from src.optimization_engine.cost_function import CostFunction
from src.model_zoo import get_model
import os

class TradeoffAnalyzer:
    def __init__(self, token_distribution_path, hardware_config, model_config, output_dir="data/processed"):
        """
        Initialize TradeoffAnalyzer for analyzing energy-runtime tradeoffs.
        
        Args:
            token_distribution_path (str): Path to token_distribution.pkl.
            hardware_config (dict): Hardware configuration.
            model_config (dict): Model configuration.
            output_dir (str): Directory to save tradeoff results.
        """
        self.token_distribution_path = Path(token_distribution_path)
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self.models = {name: get_model(name, cfg) 
                      for name, cfg in model_config["models"].items()}

    def analyze(self, model_name="tinyllama"):
        """
        Analyze energy-runtime tradeoffs for different λ values.
        
        Args:
            model_name (str): Model to use for inference.
        
        Returns:
            dict: Tradeoff results {lambda: {"energy": float, "runtime": float}}.
        """
        if not self.token_distribution_path.exists():
            self.logger.warning(f"Token distribution not found at {self.token_distribution_path}, using default distribution")
            # 使用更真实的默认分布数据
            distribution = {
                "distribution": {
                    "input_distribution": {
                        128: 0.2,    # 短对话（约50个汉字）
                        256: 0.3,    # 中等对话（约100个汉字）
                        512: 0.3,    # 长对话（约200个汉字）
                        1024: 0.15,  # 很长对话（约400个汉字）
                        2048: 0.05   # 超长对话（约800个汉字）
                    },
                    "output_distribution": {
                        256: 0.3,    # 简短回复（约100个汉字）
                        512: 0.4,    # 标准回复（约200个汉字）
                        1024: 0.2,   # 详细回复（约400个汉字）
                        2048: 0.1    # 长篇回复（约800个汉字）
                    }
                }
            }
        else:
            try:
                with open(self.token_distribution_path, 'rb') as f:
                    distribution = pickle.load(f)
                    
                # 验证分布数据结构
                if not isinstance(distribution, dict):
                    raise ValueError("分布数据必须是字典类型")
                    
                # 尝试不同的数据结构路径
                if "distribution" in distribution:
                    input_dist = distribution["distribution"].get("input_distribution")
                    output_dist = distribution["distribution"].get("output_distribution")
                else:
                    input_dist = distribution.get("input_distribution")
                    output_dist = distribution.get("output_distribution")
                    
                if not input_dist or not output_dist:
                    self.logger.warning("分布数据为空，使用默认分布 input: {32: 1.0}, output: {64: 1.0}")
                    input_dist = {32: 1.0}
                    output_dist = {64: 1.0}
                # 更新分布数据结构
                distribution = {
                    "distribution": {
                        "input_distribution": input_dist,
                        "output_distribution": output_dist
                    }
                }
                
            except Exception as e:
                self.logger.error(f"加载分布数据失败: {str(e)}")
                raise
                
        self.logger.info(f"使用的分布数据: {distribution}")

        model = self.models.get(model_name)
        if not model:
            self.logger.error(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")

        # 生成更真实的测试输入模板
        test_inputs = [
            # 短对话场景
            "请问今天的天气怎么样？我想知道是否需要带伞。",
            
            # 中等对话场景
            "我最近在学习人工智能和机器学习，想请教一下应该如何规划学习路线？我有Python基础，但对深度学习完全不了解。",
            
            # 长对话场景
            "我正在开发一个在线教育平台，需要实现以下功能：用户认证、课程管理、视频直播、作业提交和批改、在线讨论等。请问这个项目应该如何架构？需要使用哪些技术栈？",
            
            # 很长对话场景
            "我们公司准备开发一个企业级的混合云解决方案，需要支持多云管理、容器编排、微服务架构、DevOps流程自动化、安全合规等特性。同时还需要考虑高可用性、灾备、性能监控等问题。请详细分析一下技术选型和架构设计方案。" * 2,
            
            # 超长对话场景
            "请帮我详细分析一下大型语言模型在推理阶段的性能优化策略。包括但不限于：量化技术、模型压缩、分布式推理、异构计算、缓存优化等方面。同时也请考虑能耗优化、成本控制、延迟要求等实际部署问题。对于不同的应用场景，如何选择最适合的优化策略组合？" * 4
        ]

        # lambda_values = np.arange(0.0, 1.1, 0.1)
        lambda_values = np.arange(0.0, 1.1, 0.5)  # 只测3个点
        results = {}
        # 新增：采样点详细记录
        sample_points = []
        for lambda_param in lambda_values:
            self.logger.info(f"\n开始测试 λ={lambda_param} 的性能")
            cost_function = CostFunction(lambda_param, self.hardware_config)
            total_energy = 0
            total_runtime = 0
            total_tasks = 0
            try:
                for input_tokens, input_freq in distribution["distribution"]["input_distribution"].items():
                    for output_tokens, output_freq in distribution["distribution"]["output_distribution"].items():
                        input_level = min(len(test_inputs)-1, input_tokens // 256)
                        test_input = test_inputs[input_level]
                        def inference_task():
                            return model.infer(
                                test_input,
                                max_tokens=output_tokens,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.1
                            )
                        self.logger.info(f"测试配置: lambda={lambda_param}, input_tokens={input_tokens}, output_tokens={output_tokens}")
                        self.logger.debug(f"测试输入: {test_input[:200]}...")
                        try:
                            metrics = cost_function.calculate(
                                input_tokens,
                                output_tokens,
                                task=inference_task,
                                return_metrics=True
                            )
                            self.logger.info(f"测量结果: lambda={lambda_param}, input_tokens={input_tokens}, output_tokens={output_tokens}, metrics={metrics}")
                            # 新增：采样点详细记录
                            sample_points.append({
                                "lambda": float(lambda_param),
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "metrics": metrics
                            })
                            if not metrics:
                                raise ValueError("性能指标为空")
                            energy = metrics.get("energy", 0)
                            runtime = metrics.get("runtime", 0)
                            if energy <= 0:
                                self.logger.warning(f"无效的能耗值: {energy}")
                                continue
                            if runtime <= 0:
                                self.logger.warning(f"无效的运行时间: {runtime}")
                                continue
                            total_energy += energy * input_freq * output_freq
                            total_runtime += runtime * input_freq * output_freq
                            total_tasks += input_freq * output_freq
                        except Exception as e:
                            self.logger.error(f"测试失败: {str(e)}")
                            continue
            except Exception as e:
                self.logger.error(f"处理分布数据时出错: {str(e)}")
                continue
            if total_tasks > 0:
                avg_energy = total_energy / total_tasks
                avg_runtime = total_runtime / total_tasks
                results[str(lambda_param)] = {
                    "energy": avg_energy,
                    "runtime": avg_runtime
                }
                self.logger.info(f"λ={lambda_param} 的平均结果: Energy={float(avg_energy):.3f}J, Runtime={float(avg_runtime):.3f}s")
            else:
                self.logger.warning(f"λ={lambda_param} 没有有效的测试结果")
        if not results:
            raise ValueError("没有生成任何有效的权衡结果")
        # 保存采样点详细数据
        sample_path = self.output_dir / 'tradeoff_samples.json'
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_points, f, indent=2, ensure_ascii=False)
        self.logger.info(f"采样点详细数据已保存到 {sample_path}")
        # 保存结果
        result_path = self.output_dir / 'tradeoff_results.json'
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved tradeoff results to {result_path}")

        # 可视化 tradeoff 曲线
        self._visualize_tradeoff(results)
        return results

    def _visualize_tradeoff(self, results):
        """Generate and save energy-runtime tradeoff curve (similar to Figure 4)."""
        lambdas = list(results.keys())
        energies = [results[l]["energy"] for l in lambdas]
        runtimes = [results[l]["runtime"] for l in lambdas]

        plt.figure(figsize=(8, 6))
        plt.plot(runtimes, energies, marker='o')
        for i, l in enumerate(lambdas):
            plt.annotate(f"λ={float(l):.1f}", (runtimes[i], energies[i]))
        plt.xlabel('Average Runtime (seconds)')
        plt.ylabel('Average Energy (Joules)')
        plt.title('Energy-Runtime Tradeoff')
        plt.grid(True)

        plot_path = self.output_dir / 'tradeoff_curve.png'
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved tradeoff curve to {plot_path}")

    def _visualize_tradeoff_mock(self, tradeoff_results):
        """使用 mock 数据生成论文风格的能耗-时延权衡曲线。"""
        runtimes = [v["runtime"] for v in tradeoff_results["values"]]
        energies = [v["energy"] for v in tradeoff_results["values"]]
        weights = tradeoff_results["weights"]

        plt.figure(figsize=(8, 6))
        plt.plot(runtimes, energies, marker='o')
        for i, l in enumerate(weights):
            plt.annotate(f"λ={l:.2f}", (runtimes[i], energies[i]))
        plt.xlabel('Average Runtime (seconds)')
        plt.ylabel('Average Energy (Joules)')
        plt.title('Energy-Runtime Tradeoff (Mock)')
        plt.grid(True)
        plot_path = self.output_dir / 'tradeoff_curve.png'
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved tradeoff curve to {plot_path}")

