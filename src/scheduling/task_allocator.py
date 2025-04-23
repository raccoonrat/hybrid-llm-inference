# hybrid-llm-inference/src/scheduling/task_allocator.py
from toolbox.logger import get_logger
from hardware_profiling import get_profiler
from model_zoo import get_model

class TaskAllocator:
    def __init__(self, hardware_config, model_config):
        """
        Initialize TaskAllocator for executing tasks on assigned hardware.
        
        Args:
            hardware_config (dict): Hardware configuration.
            model_config (dict): Model configuration.
        """
        self.logger = get_logger(__name__)
        if not hardware_config or not model_config:
            self.logger.error("Hardware or model configuration is empty")
            raise ValueError("Hardware and model configurations cannot be empty")
        
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.profilers = {}
        for key, cfg in hardware_config.items():
            try:
                self.profilers[key] = get_profiler(key, cfg)
            except Exception as e:
                self.logger.error(f"Failed to initialize profiler for {key}: {e}")
                raise
        self.models = {}
        for name, cfg in model_config["models"].items():
            try:
                self.models[name] = get_model(name, cfg.get("mode", "local"), cfg)
            except Exception as e:
                self.logger.error(f"Failed to initialize model {name}: {e}")
                raise
        self.logger.info("TaskAllocator initialized")

    def allocate(self, allocations, model_name="llama3"):
        """
        Execute tasks on assigned hardware and collect metrics.
        
        Args:
            allocations (list): List of allocations [{"query": dict, "hardware": str}].
            model_name (str): Model to use for inference.
        
        Returns:
            list: Results [{"query": dict, "hardware": str, "metrics": dict}].
        """
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")
        
        if not allocations:
            self.logger.warning("No allocations provided, returning empty results")
            return []
        
        model = self.models[model_name]
        results = []

        for allocation in allocations:
            query = allocation.get("query", {})
            hardware = allocation.get("hardware")
            if not query or not hardware:
                self.logger.warning(f"Skipping invalid allocation: {allocation}")
                continue
            
            prompt = query.get("prompt", "")
            input_tokens = query.get("input_tokens", 0)
            output_tokens = query.get("output_tokens", 0)
            
            if hardware not in self.profilers:
                self.logger.warning(f"Hardware {hardware} not supported, skipping")
                continue
            
            profiler = self.profilers[hardware]
            task = lambda: model.infer(prompt)
            
            try:
                metrics = profiler.measure(task, input_tokens, output_tokens)
                result = {
                    "query": query,
                    "hardware": hardware,
                    "metrics": metrics
                }
                results.append(result)
                self.logger.debug(f"Executed task on {hardware}: {metrics}")
            except Exception as e:
                self.logger.error(f"Failed to execute task on {hardware}: {e}")
                continue
        
        self.logger.info(f"Allocated and executed {len(results)} tasks")
        return results
