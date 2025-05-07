# hybrid-llm-inference/src/main.py
import yaml
from pathlib import Path
from src.toolbox.config_manager import ConfigManager
from src.toolbox.logger import get_logger
from src.data_processing.alpaca_loader import AlpacaLoader
from src.data_processing.token_processing import TokenProcessing
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
        

        print("DEBUG devices:", hardware_config["devices"])
        for k, v in hardware_config["devices"].items():
            print("DEBUG device key:", k, "value type:", type(v))

        # 加载数据集
        dataset_path = "data/alpaca_data.json"
        df = AlpacaLoader(dataset_path).load_data()
        
        # 初始化TokenProcessing
        model_config_all = config_manager.load_config("model_config.yaml")
        ensure_tuple_search_range(model_config_all)
        model_type = "tinyllama"
        model_config = model_config_all["models"][model_type]
        ensure_tuple_search_range(model_config)
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
        
        # 兼容ThresholdOptimizer的measure_fn签名
        def dummy_measure_fn(input_tokens, output_tokens, device_id):
            # 让成本随 input_tokens 和 output_tokens 增大
            return {"latency": input_tokens * 0.1 + 1, "energy": output_tokens * 0.05 + 1}
        
        # Optimize thresholds
        optimizer = ThresholdOptimizer(
            search_range=model_config.get("search_range", (0, 100)),
            num_points=10,
            device_id=device,
            measure_fn=dummy_measure_fn
        )
        # 这里的optimize参数需根据实际实现调整，暂用空字典
        thresholds = optimizer.optimize({})
        logger.info(f"Optimized thresholds: {thresholds}")
        
        # Schedule and allocate tasks
        scheduler_config = config_manager.load_config("scheduler_config.yaml")
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
            allocations_for_allocator.append({
                "query": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "prompt": task.get("input", "")
                },
                "model": task.get("model", model_type),
                "hardware": task.get("hardware", "nvidia_rtx4050")
            })

        results = allocator.allocate(allocations_for_allocator, model_name=model_type)
        logger.info(f"Allocated and executed {len(results)} tasks")
        
        # SystemBenchmarking 需要 config 字典
        system_benchmarking_config = {
            "model_name": model_type,
            "batch_size": model_config.get("batch_size", 1),
            "dataset_path": dataset_path,
            "model_config": model_config_all,
            "hardware_config": copy.deepcopy(hardware_config),
            "output_dir": "data/benchmarking",
            "scheduler_config": scheduler_config,
            "model_path": model_config.get("model_path", ""),
            "device": device,
            "dtype": model_config.get("dtype", "float32")
        }
        benchmarker = SystemBenchmarking(system_benchmarking_config)
        benchmark_results = benchmarker.run_benchmarks(thresholds, model_name=model_type, sample_size=1000)
        logger.info("Completed system benchmarking")
        
        # Analyze tradeoffs
        analyzer = TradeoffAnalyzer(distribution_path, hardware_config, model_config, output_dir="data/benchmarking")
        tradeoff_results = analyzer.analyze(model_name=model_type)
        logger.info("Completed tradeoff analysis")
        
        # Generate report
        generator = ReportGenerator(output_dir="data/benchmarking")
        generator.generate_report(benchmark_results, tradeoff_results)
        logger.info("Generated benchmark report and visualizations")
        

        
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
