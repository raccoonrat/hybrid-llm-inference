# hybrid-llm-inference/src/main.py
import yaml
from pathlib import Path
from src.toolbox.config_manager import ConfigManager
from src.toolbox.logger import get_logger
from src.data_processing.alpaca_loader import AlpacaLoader
from src.data_processing.token_processing import TokenProcessing, analyze_token_distribution
from src.dataset_manager.token_distribution import TokenDistribution
from src.optimization_engine.threshold_optimizer import ThresholdOptimizer
from src.optimization_engine.tradeoff_analyzer import TradeoffAnalyzer
from src.scheduling.token_based_scheduler import TokenBasedScheduler
from src.scheduling.task_allocator import TaskAllocator
from src.benchmarking.system_benchmarking import SystemBenchmarking
from src.benchmarking.model_benchmarking import ModelBenchmarking
from src.benchmarking.report_generator import ReportGenerator
import pandas as pd
import pickle
import torch
import copy
import os
from src.model_zoo import get_model

def main():
    logger = get_logger(__name__)
    logger.info("Starting hybrid LLM inference system")
    
    try:
        # Load configurations
        config_dir = "configs"
        config_manager = ConfigManager(config_dir)
        hardware_config = config_manager.load_config("hardware_config.yaml")
        # 只对设备做 search_range 处理，避免 hardware_config 顶层出现 search_range

        if "devices" in hardware_config:
            for dev in hardware_config["devices"].values():
                if isinstance(dev, dict):
                    ensure_tuple_search_range(dev)
        # 设置主用 device 字段，供 ModelBenchmarking 使用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hardware_config["device"] = device
        

        
        for k, v in hardware_config["devices"].items():
            print("DEBUG device key:", k, "value type:", type(v))

        # 加载数据集
        dataset_path = "data/alpaca_data.json"
        df = AlpacaLoader(dataset_path).load_data()
        df = df.head(3)
        
        # 加载模型配置
        model_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "model_config.yaml")
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config_all = yaml.safe_load(f)
        
        # 加载调度器配置
        scheduler_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "scheduler_config.yaml")
        with open(scheduler_config_path, "r", encoding="utf-8") as f:
            scheduler_config = yaml.safe_load(f)
        
        # 获取模型配置
        model_type = os.getenv("MODEL_TYPE", "TinyLlama-1.1B-Chat-v1.0")
        model_config = model_config_all["models"].get(model_type, {})
        if not model_config:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        # 自动分析 token 分布（此时 model_config 已定义）
        analyze_token_distribution(
            data_path=dataset_path,
            model_path=model_config.get("model_path", "models/TinyLlama-1.1B-Chat-v1.0"),
            output_json="data/token_distribution.json",
            input_hist_png="data/input_token_hist.png",
            output_hist_png="data/output_token_hist.png"
        )
        
        # 确保 search_range 是元组
        if "search_range" in model_config:
            if isinstance(model_config["search_range"], list):
                model_config["search_range"] = tuple(model_config["search_range"])
            elif not isinstance(model_config["search_range"], tuple):
                raise ValueError("search_range 必须是列表或元组")
        else:
            model_config["search_range"] = (16, 100)  # 默认值
        
        # SystemBenchmarking 需要 config 字典
        system_benchmarking_config = {
            "model_name": model_type,
            "batch_size": model_config.get("batch_size", 1),
            "dataset_path": dataset_path,
            "model_config": model_config_all,  # 使用完整的模型配置
            "hardware_config": copy.deepcopy(hardware_config),
            "output_dir": "data/benchmarking",
            "scheduler_config": scheduler_config,
            "model_path": model_config.get("model_path", ""),
            "device": device,
            "dtype": model_config.get("dtype", "float32")
        }
        
        # 初始化TokenProcessing
        token_processor = TokenProcessing(model_name=model_type, model_config=model_config)
        tokenized_tasks = token_processor.process_tokens([{"input": row["text"]} for _, row in df.iterrows()])
        logger.info(f"Loaded and processed {len(tokenized_tasks)} tasks")
        
        # Compute token distribution (if not exists)
        distribution_path = "data/token_distribution.pkl"
        if not Path(distribution_path).exists():
            # 先转为DataFrame
            token_df = token_processor.get_token_data(pd.DataFrame(tokenized_tasks), format="dataframe")
            distribution = token_processor.compute_distribution(token_df)
            # 保存分布数据
            with open(distribution_path, "wb") as f:
                pickle.dump(distribution, f)
            logger.info(f"Saved token distribution to {distribution_path}")
        
        # Run model-specific benchmarks
        model_benchmarking_config = {
            "model_name": model_type,
            "batch_size": model_config.get("batch_size", 1),
            "dataset_path": dataset_path,
            "model_config": model_config,
            "hardware_config": hardware_config,
            "output_dir": "data/benchmarking"
        }
        model_benchmarker = ModelBenchmarking(model_benchmarking_config)
        model_benchmark_results = model_benchmarker.run_benchmarks()
        logger.info("Completed model-specific benchmarks")
        # 删除 device 字段，避免影响 TaskAllocator
        del hardware_config["device"]
        
        # 初始化模型实例
        model = get_model(model_type, model_config)

        # Optimize thresholds
        optimizer = ThresholdOptimizer(
            search_range=model_config.get("search_range", (0, 100)),
            num_points=3,  # 只采样3个点
            device_id=device,
            hardware_config=hardware_config,
            model=model
        )
        # 这里的optimize参数需根据实际实现调整，暂用空字典
        thresholds = optimizer.optimize({})
        logger.info(f"Optimized thresholds: {thresholds}")
        
        # Schedule and allocate tasks
        scheduler_config["token_threshold"] = thresholds
        scheduler_config["hardware_config"] = hardware_config
        scheduler_config["model_config"] = model_config_all
        scheduler = TokenBasedScheduler(scheduler_config)
        allocations = scheduler.schedule(tokenized_tasks)
        logger.info(f"Scheduled {len(allocations)} tasks")
        
        allocator = TaskAllocator(hardware_config, model_config_all)
        # allocations 需转换为 TaskAllocator 期望的格式
        allocations_for_allocator = []
        for task in allocations:
            input_tokens = task.get("input_tokens_count", 0)
            output_tokens = task.get("output_tokens_count", 0)
            try:
                input_tokens = int(input_tokens)
            except Exception:
                input_tokens = 0
            try:
                output_tokens = int(output_tokens)
            except Exception:
                output_tokens = 0
            # 跳过总令牌数为0的任务
            if input_tokens + output_tokens <= 0:
                continue
            
            # 获取硬件映射
            hardware_name = task.get("hardware", "rtx4050")  # 默认使用 rtx4050
            hardware_map = scheduler_config.get("hardware_map", {})
            mapped_hardware = hardware_map.get(hardware_name)
            
            # 如果映射不存在，使用原始硬件名称
            if mapped_hardware is None:
                mapped_hardware = hardware_name
            
            # 确保硬件名称存在于硬件配置中
            if mapped_hardware not in hardware_config["devices"]:
                logger.warning(f"硬件 {mapped_hardware} 不在配置中，使用默认硬件 rtx4050")
                mapped_hardware = "rtx4050"
            
            allocations_for_allocator.append({
                "query": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "prompt": task.get("input", "")
                },
                "model": task.get("model", model_type),
                "hardware": mapped_hardware
            })

        results = allocator.allocate(allocations_for_allocator, model_name=model_type)
        logger.info(f"Allocated and executed {len(results)} tasks")
        
        benchmarker = SystemBenchmarking(system_benchmarking_config)
        benchmark_results = benchmarker.run_benchmarks()
        logger.info("Completed system benchmarking")
        
        # 创建权衡分析器
        # 确保每个模型配置都包含必需字段
        for model_cfg in model_config_all["models"].values():
            model_cfg["device"] = device
            model_cfg["dtype"] = model_cfg.get("dtype", "float32")  # 使用默认值 float32
        analyzer = TradeoffAnalyzer(distribution_path, hardware_config, model_config_all, output_dir="data/benchmarking")
        tradeoff_results = analyzer.analyze(model_name=model_type)
        logger.info("Completed tradeoff analysis")
        
        # Generate report
        generator = ReportGenerator(output_dir="data/benchmarking")
        generator.generate_report(benchmark_results, tradeoff_results)
        logger.info("Generated benchmark report and visualizations")

        # 额外生成 markdown 格式的完整报告
        generator.generate_report(benchmark_results, tradeoff_results, output_format="markdown")
        logger.info("Generated markdown benchmark report")
        

        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    
    logger.info("Pipeline completed successfully")

def ensure_tuple_search_range(cfg):
    if "search_range" not in cfg:
        cfg["search_range"] = (16, 100)
    elif isinstance(cfg["search_range"], list):
        cfg["search_range"] = tuple(cfg["search_range"])
    elif not (isinstance(cfg["search_range"], tuple) and len(cfg["search_range"]) == 2):
        cfg["search_range"] = (16, 100)

if __name__ == "__main__":
    main()
