import os
import json
import unittest
from pathlib import Path
from src.benchmarking.report_generator import ReportGenerator

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

if __name__ == "__main__":
    unittest.main() 