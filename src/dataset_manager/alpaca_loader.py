# hybrid-llm-inference/src/dataset_manager/alpaca_loader.py
import json
import pandas as pd
from pathlib import Path
from toolbox.logger import get_logger

class AlpacaLoader:
    def __init__(self, dataset_path):
        """
        Initialize AlpacaLoader for loading Alpaca dataset.
        
        Args:
            dataset_path (str): Path to Alpaca JSON file.
        """
        self.dataset_path = Path(dataset_path)
        self.logger = get_logger(__name__)
        self.data = None

    def load(self):
        """Load and validate Alpaca dataset."""
        if not self.dataset_path.exists():
            self.logger.error(f"Dataset not found at {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            raise

        # Validate and format data
        validated_data = []
        for item in raw_data:
            prompt = item.get('prompt', '').strip()
            response = item.get('response', '').strip()
            if not prompt:
                self.logger.warning("Skipping empty prompt")
                continue
            if len(prompt) > 10000 or len(response) > 10000:  # Prevent overly long texts
                self.logger.warning(f"Skipping overly long item: prompt={prompt[:50]}...")
                continue
            validated_data.append({'prompt': prompt, 'response': response})

        self.data = pd.DataFrame(validated_data)
        self.logger.info(f"Loaded and validated {len(self.data)} prompts from {self.dataset_path}")
        return self.data

    def get_data(self):
        """Return loaded data."""
        if self.data is None:
            self.logger.warning("Data not loaded. Call load() first.")
            return pd.DataFrame()
        return self.data
