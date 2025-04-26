import os
import json
import unittest
from pathlib import Path
from src.benchmarking.report_generator import ReportGenerator
import pytest

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        """设置测试环境。"""
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # 创建测试数据
        self.test_data = {
            "metrics": {
                "throughput": 100.5,
                "latency": 0.1,
                "power_usage": 50.2,
                "memory_usage": 2048.0
            },
            "parallel_metrics": [
                {"throughput": 90.0, "latency": 0.12},
                {"throughput": 95.0, "latency": 0.11},
                {"throughput": 105.0, "latency": 0.09},
                {"throughput": 110.0, "latency": 0.08}
            ],
            "scheduling_metrics": {
                "strategy": "token_based",
                "num_workers": 4,
                "avg_queue_length": 2.5
            }
        }
        
        # 初始化报告生成器
        self.report_generator = ReportGenerator(str(self.test_dir))

    def test_data_validation(self):
        """测试数据验证功能。"""
        # 测试有效数据
        self.report_generator._validate_data(self.test_data)
        
        # 测试缺少必要字段
        invalid_data = self.test_data.copy()
        del invalid_data["metrics"]
        with self.assertRaises(ValueError):
            self.report_generator._validate_data(invalid_data)
            
        # 测试字段类型错误
        invalid_data = self.test_data.copy()
        invalid_data["metrics"] = "not a dict"
        with self.assertRaises(ValueError):
            self.report_generator._validate_data(invalid_data)

    def test_tradeoff_curve(self):
        """测试权衡曲线生成。"""
        curve_path = self.test_dir / "test_curve.png"
        self.report_generator._plot_tradeoff_curve(self.test_data, str(curve_path))
        
        # 验证图表文件是否生成
        self.assertTrue(curve_path.exists())
        
        # 测试数据不足的情况
        invalid_data = self.test_data.copy()
        invalid_data["parallel_metrics"] = []
        with self.assertRaises(ValueError):
            self.report_generator._plot_tradeoff_curve(invalid_data, str(curve_path))

    def test_report_generation(self):
        """测试报告生成。"""
        report_path = self.report_generator.generate_report(self.test_data, format="json")
        
        # 验证报告文件是否生成
        self.assertTrue(Path(report_path).exists())
        
        # 验证报告内容
        with open(report_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            self.assertIn("metrics", content)
            self.assertIn("parallel_metrics", content)
            self.assertIn("scheduling_metrics", content)
            
        # 验证权衡曲线图片是否生成
        curve_path = self.test_dir / "tradeoff_curve.png"
        self.assertTrue(curve_path.exists())

    def tearDown(self):
        """清理测试环境。"""
        import shutil
        shutil.rmtree(self.test_dir)

@pytest.fixture
def sample_benchmark_results():
    """创建示例基准测试结果。"""
    return {
        "metrics": {
            "throughput": 100.5,
            "latency": 50.2,
            "memory_usage": 1024.0,
            "power_usage": 75.3
        },
        "model_info": {
            "name": "test_model",
            "parameters": 1000000,
            "device": "cpu"
        },
        "test_config": {
            "batch_size": 32,
            "num_iterations": 100,
            "warmup_iterations": 10
        },
        "scheduling_stats": {
            "strategy": "round_robin",
            "num_workers": 4,
            "task_distribution": [25, 25, 25, 25]
        }
    }

@pytest.fixture
def report_generator(tmp_path):
    """创建报告生成器实例。"""
    return ReportGenerator(str(tmp_path))

def test_report_generator_valid_output(report_generator, sample_benchmark_results, tmp_path):
    """测试报告生成器的有效输出。"""
    # 生成报告
    report_path = report_generator.generate_report(sample_benchmark_results)
    
    # 验证报告文件存在
    assert os.path.exists(report_path)
    
    # 验证报告内容
    with open(report_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        assert "metrics" in content
        assert "model_info" in content
        assert "test_config" in content
        assert "scheduling_stats" in content
        
        # 验证指标值
        assert content["metrics"]["throughput"] == 100.5
        assert content["metrics"]["latency"] == 50.2
        
        # 验证模型信息
        assert content["model_info"]["name"] == "test_model"
        assert content["model_info"]["parameters"] == 1000000
        
        # 验证调度信息
        assert content["scheduling_stats"]["strategy"] == "round_robin"
        assert content["scheduling_stats"]["num_workers"] == 4

def test_report_generator_invalid_data_handling(report_generator):
    """测试报告生成器处理无效数据的情况。"""
    # 测试空数据
    with pytest.raises(ValueError, match="基准测试结果不能为空"):
        report_generator.generate_report({})

def test_report_generator_custom_options(report_generator, sample_benchmark_results, tmp_path):
    """测试报告生成器的自定义选项。"""
    # 测试不同的输出格式
    formats = ["json", "csv", "markdown"]
    for fmt in formats:
        report_path = report_generator.generate_report(
            sample_benchmark_results,
            format=fmt
        )
        assert os.path.exists(report_path)

def test_report_generator_visualization(report_generator, sample_benchmark_results, tmp_path):
    """测试报告生成器的可视化功能。"""
    # 生成报告和图表
    report_path = report_generator.generate_report(
        sample_benchmark_results,
        include_visualizations=True
    )
    assert os.path.exists(report_path)
    
    # 验证图表文件是否存在
    for metric in ["throughput", "latency", "memory_usage", "power_usage"]:
        chart_path = os.path.join(os.path.dirname(report_path), f"{metric}_time_series.png")
        assert os.path.exists(chart_path)

def test_report_generator_data_export(report_generator, sample_benchmark_results, tmp_path):
    """测试报告生成器的数据导出功能。"""
    # 测试原始数据导出
    raw_data_path = report_generator.export_raw_data(
        sample_benchmark_results,
        format="json"
    )
    assert os.path.exists(raw_data_path)
    
    # 测试摘要数据导出
    summary_path = os.path.join(tmp_path, "summary.csv")
    report_generator.export_summary(sample_benchmark_results, summary_path)
    assert os.path.exists(summary_path)

def test_report_generator_error_handling(report_generator):
    """测试报告生成器的错误处理。"""
    # 测试无效的输出格式
    with pytest.raises(ValueError, match="不支持的输出格式"):
        report_generator.generate_report(
            {"metrics": {}},
            format="invalid"
        )
    
    # 测试无效的模板
    with pytest.raises(ValueError, match="不支持的输出格式"):
        report_generator.generate_report(
            {"metrics": {}},
            format="invalid",
            template="invalid"
        )

if __name__ == "__main__":
    unittest.main() 