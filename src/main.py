# hybrid-llm-inference/src/main.py
import yaml
from pathlib import Path
from toolbox.config_manager import ConfigManager
from toolbox.logger import get_logger
from dataset_manager.alpaca_loader import AlpacaLoader
from dataset_manager.data_processing import DataProcessing
from dataset_manager.token_distribution import TokenDistribution
from optimization_engine.threshold_optimizer import ThresholdOptimizer
from optimization_engine.tradeoff_analyzer import TradeoffAnalyzer
from scheduling.token_based_scheduler import TokenBasedScheduler
from scheduling.task_allocator import TaskAllocator
from benchmarking.system_benchmarking import SystemBenchmarking
from benchmarking.model_benchmarking import ModelBenchmarking
from benchmarking.report_generator import ReportGenerator

def main():
    logger = get_logger(__name__)
    logger.info("Starting hybrid LLM inference system")
    
    try:
        # Load configurations
        config_dir = "configs"
        config_manager = ConfigManager(config_dir)
        hardware_config = config_manager.load_config("hardware_config.yaml")
        model_config = config_manager.load_config("model_config.yaml")
        scheduler_config = config_manager.load_config("scheduler_config.yaml")
        
        # Load and process dataset
        dataset_path = "data/alpaca_prompts.json"
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        loader = AlpacaLoader(dataset_path)
        processor = DataProcessing(loader, model_config["models"]["llama3"])
        token_data = processor.get_token_data()
        logger.info(f"Loaded and processed {len(token_data)} tasks")
        
        # Compute token distribution (if not exists)
        distribution_path = "data/token_distribution.pkl"
        if not Path(distribution_path).exists():
            distribution = TokenDistribution(processor)
            distribution.compute_distribution()
            distribution.save_distribution(distribution_path)
            logger.info(f"Saved token distribution to {distribution_path}")
        
        # Run model-specific benchmarks
        model_benchmarker = ModelBenchmarking(dataset_path, hardware_config, model_config, output_dir="data/benchmarking")
        model_benchmark_results = model_benchmarker.run_benchmarks(sample_size=100)
        logger.info("Completed model-specific benchmarks")
        
        # Optimize thresholds
        optimizer = ThresholdOptimizer(distribution_path, hardware_config, model_config)
        thresholds = optimizer.optimize(lambda_param=0.5, model_name="llama3")
        logger.info(f"Optimized thresholds: {thresholds}")
        
        # Schedule and allocate tasks
        scheduler = TokenBasedScheduler(thresholds, scheduler_config)
        allocations = scheduler.schedule(token_data)
        logger.info(f"Scheduled {len(allocations)} tasks")
        
        allocator = TaskAllocator(hardware_config, model_config)
        results = allocator.allocate(allocations, model_name="llama3")
        logger.info(f"Allocated and executed {len(results)} tasks")
        
        # Run system benchmarks
        benchmarker = SystemBenchmarking(dataset_path, hardware_config, model_config, scheduler_config, 
                                       output_dir="data/benchmarking")
        benchmark_results = benchmarker.run_benchmarks(thresholds, model_name="llama3", sample_size=1000)
        logger.info("Completed system benchmarking")
        
        # Analyze tradeoffs
        analyzer = TradeoffAnalyzer(distribution_path, hardware_config, model_config, output_dir="data/benchmarking")
        tradeoff_results = analyzer.analyze(model_name="llama3")
        logger.info("Completed tradeoff analysis")
        
        # Generate report
        generator = ReportGenerator(output_dir="data/benchmarking")
        generator.generate_report(benchmark_results, tradeoff_results)
        logger.info("Generated benchmark report and visualizations")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()
