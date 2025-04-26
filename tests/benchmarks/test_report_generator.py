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
        report_path = self.report_generator.generate_report(self.test_data, str(self.test_dir))
        
        # 验证报告文件是否生成
        self.assertTrue(Path(report_path).exists())
        
        # 验证报告内容
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("基准测试报告", content)
            self.assertIn("基本指标", content)
            self.assertIn("调度指标", content)
            self.assertIn("权衡曲线", content)
            
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
    
    # 测试缺少必要字段
    invalid_data = {"metrics": {}}
    with pytest.raises(ValueError, match="缺少必要的指标数据"):
        report_generator.generate_report(invalid_data)
    
    # 测试无效的指标值
    invalid_metrics = {
        "metrics": {
            "throughput": "invalid",
            "latency": 50.2
        }
    }
    with pytest.raises(TypeError, match="指标值必须是数字类型"):
        report_generator.generate_report(invalid_metrics)
    
    # 测试无效的模型信息
    invalid_model_info = {
        "metrics": {
            "throughput": 100.5,
            "latency": 50.2
        },
        "model_info": "invalid"
    }
    with pytest.raises(TypeError, match="模型信息必须是字典类型"):
        report_generator.generate_report(invalid_model_info)

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
        assert report_path.endswith(f".{fmt}")
    
    # 测试自定义报告模板
    custom_template = {
        "title": "性能测试报告",
        "sections": ["模型信息", "性能指标", "调度统计"],
        "charts": ["throughput_vs_latency", "memory_usage_timeline"]
    }
    report_path = report_generator.generate_report(
        sample_benchmark_results,
        template=custom_template
    )
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        assert "title" in content
        assert content["title"] == "性能测试报告"
        assert all(section in content for section in custom_template["sections"])

def test_report_generator_visualization(report_generator, sample_benchmark_results, tmp_path):
    """测试报告生成器的可视化功能。"""
    # 生成报告和图表
    report_generator.generate_report(
        sample_benchmark_results,
        include_visualizations=True
    )
    
    # 验证图表文件是否生成
    expected_charts = [
        "throughput_latency_tradeoff.png",
        "memory_usage_timeline.png",
        "task_distribution.png"
    ]
    
    for chart in expected_charts:
        chart_path = tmp_path / chart
        assert os.path.exists(chart_path)

def test_report_generator_data_export(report_generator, sample_benchmark_results, tmp_path):
    """测试报告生成器的数据导出功能。"""
    # 测试原始数据导出
    raw_data_path = report_generator.export_raw_data(
        sample_benchmark_results,
        format="json"
    )
    assert os.path.exists(raw_data_path)
    
    # 测试聚合数据导出
    summary_path = report_generator.export_summary(
        sample_benchmark_results,
        format="csv"
    )
    assert os.path.exists(summary_path)
    
    # 验证导出数据的完整性
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        assert raw_data == sample_benchmark_results

def test_report_generator_error_handling(report_generator):
    """测试报告生成器的错误处理。"""
    # 测试无效的输出格式
    with pytest.raises(ValueError, match="不支持的输出格式"):
        report_generator.generate_report(
            {"metrics": {}},
            format="invalid"
        )
    
    # 测试无效的模板
    with pytest.raises(ValueError, match="无效的报告模板"):
        report_generator.generate_report(
            {"metrics": {}},
            template="invalid"
        )
    
    # 测试文件系统错误
    report_generator.output_dir = "/invalid/path"
    with pytest.raises(OSError):
        report_generator.generate_report({"metrics": {}})

if __name__ == "__main__":
    unittest.main() 