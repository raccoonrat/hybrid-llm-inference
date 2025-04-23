# hybrid-llm-inference/src/toolbox/accelerate_wrapper.py
from accelerate import Accelerator
from toolbox.logger import get_logger

class AccelerateWrapper:
    def __init__(self, model, config):
        """
        Initialize AccelerateWrapper for optimized model inference.
        
        Args:
            model: PyTorch model instance.
            config (dict): Configuration with acceleration settings.
        """
        self.model = model
        self.config = config
        self.logger = get_logger(__name__)
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "fp16"),
            device_placement=config.get("device_placement", True)
        )
        self.model = self.accelerator.prepare(self.model)
        self.logger.info("Model prepared with Accelerate")
    
    def get_model(self):
        """Return the accelerated model."""
        return self.model
