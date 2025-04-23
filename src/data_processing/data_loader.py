# hybrid-llm-inference/src/data_processing/data_loader.py
import json
import pandas as pd
from pathlib import Path
from toolbox.logger import get_logger

class DataLoader:
    def __init__(self, dataset_path, models=None):
        """
        Initialize DataLoader for Alpaca dataset.
        
        Args:
            dataset_path (str): Path to Alpaca JSON file.
            models (dict): Dictionary of model instances from ModelZoo for tokenization.
        """
        self.dataset_path = Path(dataset_path)
        self.models = models or {}
        self.logger = get_logger(__name__)
        self.data = None

    def load(self):
        """Load and clean Alpaca dataset."""
        if not self.dataset_path.exists():
            self.logger.error(f"Dataset not found at {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            raise

        # Clean and format data
        cleaned_data = []
        for item in raw_data:
            prompt = item.get('prompt', '').strip()
            response = item.get('response', '').strip()
            if not prompt or len(prompt) > 10000:  # Skip empty or overly long prompts
                self.logger.warning(f"Skipping invalid prompt: {prompt[:50]}...")
                continue
            cleaned_data.append({'prompt': prompt, 'response': response})

        self.data = pd.DataFrame(cleaned_data)
        self.logger.info(f"Loaded {len(self.data)} prompts from {self.dataset_path}")
        return self.data

    def get_data(self):
        """Return loaded data."""
        if self.data is None:
            self.logger.warning("Data not loaded. Call load() first.")
            return pd.DataFrame()
        return self.data

