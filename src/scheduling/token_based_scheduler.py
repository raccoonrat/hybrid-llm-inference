# hybrid-llm-inference/src/scheduling/token_based_scheduler.py
from toolbox.logger import get_logger

class TokenBasedScheduler:
    def __init__(self, thresholds, config):
        """
        Initialize TokenBasedScheduler for token-based task scheduling.
        
        Args:
            thresholds (dict): Scheduling thresholds {"T_in": int, "T_out": int}.
            config (dict): Scheduler configuration with hardware mapping.
        """
        self.logger = get_logger(__name__)
        if not thresholds or "T_in" not in thresholds or "T_out" not in thresholds:
            self.logger.error("Invalid thresholds provided")
            raise ValueError("Thresholds must include T_in and T_out")
        if thresholds["T_in"] <= 0 or thresholds["T_out"] <= 0:
            self.logger.error("Thresholds must be positive")
            raise ValueError("Thresholds must be positive")
        
        self.input_threshold = thresholds.get("T_in", 32)
        self.output_threshold = thresholds.get("T_out", 32)
        self.hardware_map = config.get("hardware_map", {})
        if not self.hardware_map:
            self.logger.error("Hardware map is empty")
            raise ValueError("Hardware map cannot be empty")
        
        self.logger.info(f"Initialized scheduler with T_in={self.input_threshold}, T_out={self.output_threshold}")

    def schedule(self, token_data):
        """
        Schedule tasks based on token counts.
        
        Args:
            token_data (list): List of queries with token counts 
                              [{"prompt": str, "response": str, "input_tokens": int, "output_tokens": int}, ...].
        
        Returns:
            list: Allocations [{"query": dict, "hardware": str}].
        """
        if not token_data:
            self.logger.warning("No token data provided, returning empty allocations")
            return []
        
        allocations = []
        for query in token_data:
            if "input_tokens" not in query or "output_tokens" not in query:
                self.logger.warning(f"Skipping invalid query: {query}")
                continue
            
            input_tokens = query.get("input_tokens", 0)
            output_tokens = query.get("output_tokens", 0)
            
            # Assign hardware based on thresholds
            if input_tokens <= self.input_threshold and output_tokens <= self.output_threshold:
                # Prefer low-power hardware for small tasks
                hardware = self.hardware_map.get("m1_pro", "m1_pro")
                if input_tokens <= 16 and output_tokens <= 16:
                    hardware = self.hardware_map.get("rtx4050", hardware)  # RTX 4050 for very small tasks
            else:
                # Prefer high-performance hardware for large tasks
                hardware = self.hardware_map.get("a800", self.hardware_map.get("a100", "a100"))
                
            allocation = {"query": query, "hardware": hardware}
            allocations.append(allocation)
            self.logger.debug(f"Allocated query (input={input_tokens}, output={output_tokens}) to {hardware}")
        
        self.logger.info(f"Scheduled {len(allocations)} tasks")
        return allocations
