# hybrid-llm-inference/src/model_zoo/base_model.py
from abc import ABC, abstractmethod
from toolbox.logger import get_logger

class BaseModel(ABC):
    def __init__(self, model_name, config):
        """
        Initialize BaseModel.
        
        Args:
            model_name (str): Name of the model.
            config (dict): Model configuration.
        """
        self.model_name = model_name
        self.config = config
        self.logger = get_logger(__name__)
    
    @abstractmethod
    def infer(self, input_text):
        """Perform inference on input text."""
        pass
    
    @abstractmethod
    def get_token_count(self, text):
        """Return token count for text."""
        pass
