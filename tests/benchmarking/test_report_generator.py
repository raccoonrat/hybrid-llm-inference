import os
import json
import unittest
from pathlib import Path
from src.benchmarking.report_generator import ReportGenerator
import pytest
import numpy as np
from datetime import datetime
import tempfile

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        """设置测试环境。"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建报告生成器
        self.report_generator = ReportGenerator(self.output_dir)
        
        # 创建测试数据
        self.test_data = {
            "metrics": {
                "latency": 0.1,
                "memory_usage": 2048.0,
                "power_usage": 50.2,
                "throughput": 100.5,
                "energy": 10.0
            },
            "parallel_metrics": [
                {
                    "latency": 0.2,
                    "memory_usage": 1024.0,
                    "power_usage": 25.1,
                    "throughput": 50.2,
                    "energy": 5.0
                },
                {
                    "latency": 0.3,
                    "memory_usage": 3072.0,
                    "power_usage": 75.3,
                    "throughput": 150.8,
                    "energy": 15.0
                }
            ],
            "scheduling_metrics": {
                "avg_queue_length": 2.5,
                "num_workers": 4,
                "strategy": "token_based"
            },
            "timestamp": "2025-04-27T10:02:00.986751"
        }

    def test_data_validation(self):
        """测试数据验证功能。"""
        # 测试有效数据
        valid_data = {
            "metrics": {
                "latency": [0.1, 0.2],
                "energy": [50.0, 51.0],
                "throughput": 100.5
            },
            "timestamp": datetime.now().isoformat()
        }
        self.report_generator._validate_data(valid_data)
        
        # 测试缺少必要字段
        invalid_data = valid_data.copy()
        del invalid_data["metrics"]
        with self.assertRaises(ValueError):
            self.report_generator._validate_data(invalid_data)
            
        # 测试字段类型错误
        invalid_data = valid_data.copy()
        invalid_data["metrics"] = "not a dict"
        with self.assertRaises(ValueError):
            self.report_generator._validate_data(invalid_data)

    def test_tradeoff_curve(self):
        """测试权衡曲线生成。"""
        curve_path = os.path.join(self.output_dir, "test_curve.png")
        self.report_generator._plot_tradeoff_curve(self.test_data, curve_path)
        
        # 验证图表文件是否生成
        self.assertTrue(os.path.exists(curve_path))
        
        # 测试数据不足的情况
        invalid_data = self.test_data.copy()
        invalid_data["parallel_metrics"] = []
        with self.assertRaises(ValueError):
            self.report_generator._plot_tradeoff_curve(invalid_data, curve_path)

    def test_report_generation(self):
        """测试报告生成。"""
        report_path = self.report_generator.generate_report(self.test_data, output_format="json")
        assert os.path.exists(report_path)
        assert report_path.endswith(".json")
        
        # 验证报告内容
        with open(report_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            self.assertIn("metrics", content)
            self.assertIn("parallel_metrics", content)
            self.assertIn("scheduling_metrics", content)
            
        # 验证权衡曲线图片是否生成
        curve_path = os.path.join(self.output_dir, "tradeoff_curve.png")
        self.assertTrue(os.path.exists(curve_path))

    def tearDown(self):
        """清理测试环境。"""
        import shutil
        shutil.rmtree(self.temp_dir)

@pytest.fixture
def sample_benchmark_results():
    """生成用于测试的示例基准测试结果。"""
    return {
        "metrics": {
            "latency": [0.1, 0.2, 0.3, 0.4, 0.5],
            "energy": [50.0, 51.0, 49.0, 50.5, 50.2],
            "throughput": 100.5,
            "memory_usage": {
                "max": 2048,
                "mean": 1024,
                "min": 512
            }
        },
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "name": "test_model",
            "size": "10M",
            "type": "mock"
        },
        "test_config": {
            "batch_size": 32,
            "num_threads": 4
        },
        "task_distribution": [
            {"task_id": 1, "duration": 0.5},
            {"task_id": 2, "duration": 0.3}
        ]
    }

@pytest.fixture
def report_generator(tmp_path):
    """创建报告生成器实例。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    return ReportGenerator(str(output_dir))

def test_report_generator_visualization(tmp_path):
    """测试报告生成器的可视化功能。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))

    # 准备测试数据
    metrics = {
        "metrics": {
            "latency": [0.1, 0.12, 0.09],
            "energy": [75.0, 76.0, 74.0],
            "throughput": [100.0, 110.0, 90.0],
            "memory_usage": [1024.0, 1024.5, 1023.5]
        },
        "timestamp": datetime.now().isoformat()
    }

    # 生成报告和图表
    report_path = generator.generate_report(metrics, include_visualizations=True)
    
    # 验证图表文件是否生成
    vis_dir = os.path.join(os.path.dirname(report_path), "visualizations")
    assert os.path.exists(vis_dir)
    assert os.path.exists(os.path.join(vis_dir, "latency_distribution.png"))
    assert os.path.exists(os.path.join(vis_dir, "energy_distribution.png"))
    assert os.path.exists(os.path.join(vis_dir, "latency_energy_tradeoff.png"))

def test_report_generator_custom_options(tmp_path):
    """测试报告生成器的自定义配置选项。"""
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)

    # 测试自定义图表样式
    custom_style = {
        "figure.figsize": (12, 8),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 2,
        "lines.markersize": 8
    }
    generator = ReportGenerator(str(output_dir), style_config=custom_style)

    # 准备测试数据
    metrics = {
        "metrics": {
            "latency": [0.1, 0.2, 0.3],
            "energy": [50.0, 51.0, 49.0],
            "throughput": 100.0,
            "runtime": 1.0
        },
        "timestamp": datetime.now().isoformat()
    }

    # 生成报告
    report_path = generator.generate_report(metrics, output_format="json")
    assert os.path.exists(report_path)
    assert report_path.endswith(".json")

def test_generate_json_report(report_generator, sample_benchmark_results):
    """测试生成 JSON 格式报告"""
    report_path = report_generator.generate_report(
        sample_benchmark_results,
        output_format="json",
        include_visualizations=True
    )

    assert os.path.exists(report_path)
    assert report_path.endswith(".json")

    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert "timestamp" in report_data
    assert "metrics" in report_data
    assert "model_info" in report_data
    assert "test_config" in report_data
    assert "task_distribution" in report_data

    # 验证指标数据
    metrics = report_data["metrics"]
    assert isinstance(metrics["latency"], list)
    assert isinstance(metrics["energy"], list)
    assert isinstance(metrics["throughput"], (int, float))
    assert isinstance(metrics["memory_usage"], dict)

    # 验证可视化文件
    vis_dir = os.path.join(os.path.dirname(report_path), "visualizations")
    assert os.path.exists(vis_dir)
    assert os.path.exists(os.path.join(vis_dir, "latency_distribution.png"))
    assert os.path.exists(os.path.join(vis_dir, "energy_distribution.png"))
    assert os.path.exists(os.path.join(vis_dir, "latency_energy_tradeoff.png"))

def test_generate_csv_report(report_generator, sample_benchmark_results):
    """测试生成 CSV 格式报告"""
    # 准备扁平化的数据结构
    metrics = {
        "metrics": {
            "latency_mean": np.mean(sample_benchmark_results["metrics"]["latency"]),
            "energy_mean": np.mean(sample_benchmark_results["metrics"]["energy"]),
            "memory_max": sample_benchmark_results["metrics"]["memory_usage"]["max"],
            "memory_mean": sample_benchmark_results["metrics"]["memory_usage"]["mean"],
            "throughput": sample_benchmark_results["metrics"]["throughput"],
            "batch_size": sample_benchmark_results["test_config"]["batch_size"],
            "num_threads": sample_benchmark_results["test_config"]["num_threads"],
            "energy": sample_benchmark_results["metrics"]["energy"],
            "latency": sample_benchmark_results["metrics"]["latency"]
        },
        "timestamp": datetime.now().isoformat()
    }

    report_path = report_generator.generate_report(metrics, output_format="csv")
    assert os.path.exists(report_path)
    assert report_path.endswith(".csv")

def test_generate_markdown_report(report_generator, sample_benchmark_results):
    """测试生成 Markdown 格式报告"""
    report_path = report_generator.generate_report(
        sample_benchmark_results,
        output_format="markdown",
        include_visualizations=True
    )

    assert os.path.exists(report_path)
    assert report_path.endswith(".md")

    # 验证可视化文件
    vis_dir = os.path.join(os.path.dirname(report_path), "visualizations")
    assert os.path.exists(vis_dir)
    assert os.path.exists(os.path.join(vis_dir, "latency_distribution.png"))
    assert os.path.exists(os.path.join(vis_dir, "energy_distribution.png"))
    assert os.path.exists(os.path.join(vis_dir, "latency_energy_tradeoff.png"))

def test_invalid_input(report_generator):
    """测试无效输入处理"""
    with pytest.raises(ValueError, match="基准测试结果不能为空"):
        report_generator.generate_report({})

    with pytest.raises(ValueError, match="不支持的报告格式"):
        report_generator.generate_report(
            {
                "metrics": {
                    "latency": [0.1],
                    "energy": [50.0],
                    "throughput": 100
                },
                "timestamp": datetime.now().isoformat()
            },
            output_format="invalid"
        )

def test_visualization_generation(report_generator, sample_benchmark_results):
    """测试可视化生成"""
    report_path = report_generator.generate_report(
        sample_benchmark_results,
        include_visualizations=True
    )

    # 验证可视化文件
    vis_dir = os.path.join(os.path.dirname(report_path), "visualizations")
    assert os.path.exists(vis_dir)
    assert os.path.exists(os.path.join(vis_dir, "latency_distribution.png"))
    assert os.path.exists(os.path.join(vis_dir, "energy_distribution.png"))
    assert os.path.exists(os.path.join(vis_dir, "latency_energy_tradeoff.png"))

def test_report_directory_creation(tmp_path):
    """测试报告目录创建"""
    output_dir = os.path.join(tmp_path, "reports")
    report_generator = ReportGenerator(output_dir)

    metrics = {
        "metrics": {
            "latency": [0.1, 0.2, 0.3],
            "energy": [50.0, 51.0, 49.0],
            "throughput": 100.0
        },
        "timestamp": datetime.now().isoformat()
    }

    report_path = report_generator.generate_report(
        metrics,
        include_visualizations=False
    )

    assert os.path.exists(output_dir)
    assert os.path.exists(report_path)

if __name__ == "__main__":
    unittest.main() 