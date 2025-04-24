"""系统管道模块。"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..dataset_manager.alpaca_loader import AlpacaLoader
from ..data_processing.token_processing import TokenProcessing
from ..optimization_engine.threshold_optimizer import ThresholdOptimizer
from ..optimization_engine.tradeoff_analyzer import TradeoffAnalyzer
from ..scheduling.token_based_scheduler import TokenBasedScheduler
from ..scheduling.task_allocator import TaskAllocator
from ..benchmarking.system_benchmarking import SystemBenchmarking
from ..benchmarking.report_generator import ReportGenerator
from ..toolbox.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemPipeline:
    """系统管道类"""
    
    def __init__(
        self,
        data_path: Path,
        distribution_path: Path,
        output_dir: Path,
        model_name: str,
        model_path: str,
        mode: str = "local"
    ):
        """
        初始化系统管道。

        Args:
            data_path: 数据文件路径
            distribution_path: 分布文件路径
            output_dir: 输出目录
            model_name: 模型名称
            model_path: 模型路径
            mode: 运行模式（local/remote）
        """
        self.data_path = data_path
        self.distribution_path = distribution_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.model_path = model_path
        self.mode = mode
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.config_manager = ConfigManager()
        self.token_processor = TokenProcessing()
        self.scheduler = TokenBasedScheduler()
        self.allocator = TaskAllocator()
        self.benchmarking = SystemBenchmarking()
        self.report_generator = ReportGenerator()
        
    def run(self) -> Dict[str, Any]:
        """
        运行系统管道。

        Returns:
            Dict[str, Any]: 运行结果
        """
        try:
            # 加载数据
            data_loader = AlpacaLoader(self.data_path)
            data = data_loader.load()
            
            # 处理数据
            processed_data = self.token_processor.process(data)
            
            # 优化阈值
            optimizer = ThresholdOptimizer()
            thresholds = optimizer.optimize(processed_data)
            
            # 分析权衡
            analyzer = TradeoffAnalyzer()
            tradeoffs = analyzer.analyze(processed_data, thresholds)
            
            # 调度任务
            scheduled_tasks = self.scheduler.schedule(processed_data, tradeoffs)
            
            # 分配任务
            allocated_tasks = self.allocator.allocate(scheduled_tasks)
            
            # 基准测试
            metrics = self.benchmarking.benchmark(allocated_tasks)
            
            # 生成报告
            report = self.report_generator.generate(metrics)
            
            return {
                "energy": metrics.get("energy", 0.0),
                "runtime": metrics.get("runtime", 0.0),
                "metrics": metrics,
                "config": self.config_manager.get_config(),
                "distribution": self.distribution_path
            }
            
        except Exception as e:
            logger.error(f"运行系统管道时出错: {e}")
            raise 