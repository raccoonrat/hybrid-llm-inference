"""报告生成器测试模块。"""

import os
import json
import pytest
from pathlib import Path
from src.benchmarking.report_generator import ReportGenerator
import time
import threading
from unittest.mock import patch
import psutil

@pytest.fixture
def temp_dir():
    """创建临时目录。"""
    temp_dir = Path("test_output")
    temp_dir.mkdir(exist_ok=True)
    yield temp_dir
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)

@pytest.fixture
def report_generator(temp_dir):
    """创建报告生成器实例。"""
    return ReportGenerator(str(temp_dir))

@pytest.fixture
def test_data():
    """创建测试数据。"""
    return {
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

def test_data_validation(report_generator, test_data):
    """测试数据验证功能。"""
    # 测试有效数据
    report_generator._validate_data(test_data)
    
    # 测试缺少必要字段
    invalid_data = test_data.copy()
    del invalid_data["metrics"]
    with pytest.raises(ValueError):
        report_generator._validate_data(invalid_data)
        
    # 测试字段类型错误
    invalid_data = test_data.copy()
    invalid_data["metrics"] = "not a dict"
    with pytest.raises(ValueError):
        report_generator._validate_data(invalid_data)

def test_tradeoff_curve(report_generator, test_data, temp_dir):
    """测试权衡曲线生成。"""
    curve_path = temp_dir / "test_curve.png"
    report_generator._plot_tradeoff_curve(test_data, str(curve_path))
    
    # 验证图表文件是否生成
    assert curve_path.exists()
    
    # 测试数据不足的情况
    invalid_data = test_data.copy()
    invalid_data["parallel_metrics"] = []
    with pytest.raises(ValueError):
        report_generator._plot_tradeoff_curve(invalid_data, str(curve_path))

def test_report_generation(report_generator, test_data, temp_dir):
    """测试报告生成。"""
    report_path = report_generator.generate_report(test_data, str(temp_dir))
    
    # 验证报告文件是否生成
    assert Path(report_path).exists()
    
    # 验证报告内容
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "基准测试报告" in content
        assert "基本指标" in content
        assert "调度指标" in content
        assert "权衡曲线" in content
        
    # 验证权衡曲线图片是否生成
    curve_path = temp_dir / "tradeoff_curve.png"
    assert curve_path.exists()

def test_report_generator_valid_output(tmp_path):
    """测试报告生成器生成有效输出"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {"throughput": 100, "latency": 0.1},
        "parallel_metrics": {"speedup": 2.0},
        "scheduling_metrics": {"efficiency": 0.8}
    }
    report_path = generator.generate_report(data)
    assert os.path.exists(report_path)
    assert report_path.endswith(".md")

def test_report_generator_invalid_data_handling(tmp_path):
    """测试报告生成器处理无效数据"""
    generator = ReportGenerator(str(tmp_path))
    data = {"invalid": "data"}
    with pytest.raises(ValueError, match="数据缺少必要的字段"):
        generator.generate_report(data)

def test_report_generator_custom_options(tmp_path):
    """测试报告生成器的自定义选项"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {"throughput": 100, "latency": 0.1},
        "parallel_metrics": {"speedup": 2.0},
        "scheduling_metrics": {"efficiency": 0.8}
    }
    report_path = generator.generate_report(data, output_format="pdf")
    assert os.path.exists(report_path)
    assert report_path.endswith(".pdf")

def test_report_generator_invalid_tradeoff_results(temp_dir):
    """测试报告生成器的无效权衡结果。"""
    output_dir = temp_dir / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))
    metrics = {
        "throughput": 100.0,
        "latency": 0.1,
        "energy": 50.0,
        "runtime": 1.0,
        "summary": {
            "average_throughput": 100.0,
            "average_latency": 0.1,
            "average_energy": 50.0
        }
    }
    tradeoff_results = {
        "weights": [0.2, 0.3, 0.5],
        "values": "invalid"
    }
    with pytest.raises(ValueError):
        generator.generate_report(metrics, tradeoff_results)

def test_report_generator_custom_plots(temp_dir):
    """测试报告生成器的自定义图表生成。"""
    output_dir = temp_dir / "output"
    os.makedirs(output_dir)
    generator = ReportGenerator(str(output_dir))
    
    # 准备测试数据
    metrics = {
        "model1": {
            "throughput": [100.0, 110.0, 90.0],
            "latency": [0.1, 0.12, 0.09],
            "energy": [50.0, 55.0, 45.0],
            "runtime": [1.0, 1.1, 0.9],
            "summary": {
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
                "avg_energy_per_token": 0.5,
                "avg_runtime": 1.0
            }
        },
        "model2": {
            "throughput": [200.0, 220.0, 180.0],
            "latency": [0.2, 0.22, 0.18],
            "energy": [100.0, 110.0, 90.0],
            "runtime": [2.0, 2.2, 1.8],
            "summary": {
                "avg_throughput": 200.0,
                "avg_latency": 0.2,
                "avg_energy_per_token": 1.0,
                "avg_runtime": 2.0
            }
        }
    }
    
    # 测试时间序列图
    generator.plot_time_series(metrics, "throughput")
    assert os.path.exists(output_dir / "throughput_time_series.png")
    
    # 测试箱线图
    generator.plot_boxplot(metrics, "latency")
    assert os.path.exists(output_dir / "latency_boxplot.png")
    
    # 测试散点图
    generator.plot_scatter(metrics, "energy", "runtime")
    assert os.path.exists(output_dir / "energy_vs_runtime_scatter.png")
    
    # 测试热力图
    generator.plot_heatmap(metrics)
    assert os.path.exists(output_dir / "metrics_heatmap.png")

def test_report_generator_custom_options(temp_dir):
    """测试报告生成器的自定义配置选项。"""
    output_dir = temp_dir / "output"
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
    
    # 测试自定义输出格式
    metrics = {
        "model1": {
            "throughput": 100.0,
            "latency": 0.1,
            "energy": 50.0,
            "runtime": 1.0,
            "summary": {
                "avg_throughput": 100.0,
                "avg_latency": 0.1,
                "avg_energy_per_token": 0.5,
                "avg_runtime": 1.0
            }
        }
    }
    
    # 生成PDF报告
    generator.generate_report(metrics, {}, output_format="pdf")
    assert os.path.exists(output_dir / "report.pdf")
    
    # 生成CSV报告
    generator.generate_report(metrics, {}, output_format="csv")
    assert os.path.exists(output_dir / "metrics.csv")
    
    # 测试自定义模板
    custom_template = """
    <html>
    <head><title>Custom Report</title></head>
    <body>
        <h1>Custom Benchmark Report</h1>
        <div class="metrics">{{metrics_table}}</div>
        <div class="plots">{{plots}}</div>
    </body>
    </html>
    """
    generator.generate_report(metrics, {}, template=custom_template)
    
    # 验证生成的报告包含自定义模板内容
    with open(output_dir / "report.html", "r", encoding="utf-8") as f:
        content = f.read()
        assert "Custom Benchmark Report" in content
        assert '<div class="metrics">' in content
        assert '<div class="plots">' in content

def test_report_generator_empty_metrics(tmp_path):
    """测试报告生成器处理空指标数据"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {},
        "parallel_metrics": {},
        "scheduling_metrics": {}
    }
    report_path = generator.generate_report(data)
    assert os.path.exists(report_path)
    assert report_path.endswith(".md")

def test_report_generator_extreme_values(tmp_path):
    """测试报告生成器处理极端值"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {
            "throughput": float('inf'),
            "latency": 0.0
        },
        "parallel_metrics": {
            "speedup": float('nan')
        },
        "scheduling_metrics": {
            "efficiency": -1.0
        }
    }
    report_path = generator.generate_report(data)
    assert os.path.exists(report_path)
    assert report_path.endswith(".md")

def test_report_generator_invalid_output_format(tmp_path):
    """测试报告生成器处理无效的输出格式"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {"throughput": 100, "latency": 0.1},
        "parallel_metrics": {"speedup": 2.0},
        "scheduling_metrics": {"efficiency": 0.8}
    }
    with pytest.raises(ValueError, match="不支持的输出格式"):
        generator.generate_report(data, output_format="invalid_format")

def test_report_generator_large_data(tmp_path):
    """测试报告生成器处理大量数据"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {f"metric_{i}": i for i in range(1000)},
        "parallel_metrics": {f"parallel_{i}": i * 2 for i in range(1000)},
        "scheduling_metrics": {f"scheduling_{i}": i * 0.5 for i in range(1000)}
    }
    start_time = time.time()
    report_path = generator.generate_report(data)
    end_time = time.time()
    assert os.path.exists(report_path)
    assert report_path.endswith(".md")
    assert end_time - start_time < 5.0  # 确保生成报告的时间不超过5秒

def test_report_generator_concurrent_access(tmp_path):
    """测试报告生成器的并发访问"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {"throughput": 100, "latency": 0.1},
        "parallel_metrics": {"speedup": 2.0},
        "scheduling_metrics": {"efficiency": 0.8}
    }
    
    def generate_report():
        return generator.generate_report(data)
    
    # 创建10个线程同时生成报告
    threads = [threading.Thread(target=generate_report) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # 验证所有报告文件都已生成
    report_files = list(tmp_path.glob("*.md"))
    assert len(report_files) == 10

def test_report_generator_error_recovery(tmp_path):
    """测试报告生成器的错误恢复能力"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {"throughput": 100, "latency": 0.1},
        "parallel_metrics": {"speedup": 2.0},
        "scheduling_metrics": {"efficiency": 0.8}
    }
    
    # 模拟文件系统错误
    with patch("builtins.open", side_effect=IOError("模拟文件系统错误")):
        with pytest.raises(IOError):
            generator.generate_report(data)
    
    # 验证错误后仍然可以正常生成报告
    report_path = generator.generate_report(data)
    assert os.path.exists(report_path)
    assert report_path.endswith(".md")

def test_report_generator_memory_cleanup(tmp_path):
    """测试报告生成器的内存清理"""
    generator = ReportGenerator(str(tmp_path))
    data = {
        "metrics": {"throughput": 100, "latency": 0.1},
        "parallel_metrics": {"speedup": 2.0},
        "scheduling_metrics": {"efficiency": 0.8}
    }
    
    # 生成大量报告
    for _ in range(100):
        generator.generate_report(data)
    
    # 验证内存使用情况
    process = psutil.Process()
    memory_info = process.memory_info()
    assert memory_info.rss < 100 * 1024 * 1024  # 确保内存使用不超过100MB 