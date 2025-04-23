# hybrid-llm-inference/src/dataset_manager/token_distribution.py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from toolbox.logger import get_logger
from model_zoo import get_model
import numpy as np

class TokenDistribution:
    def __init__(self, data, models, output_dir="data/processed"):
        """
        Initialize TokenDistribution for analyzing token distributions.
        
        Args:
            data (pd.DataFrame): DataFrame with 'prompt' and 'response' columns.
            models (dict): Dictionary of model instances from ModelZoo.
            output_dir (str): Directory to save distribution data and plots.
        """
        self.data = data
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self.distribution = None
        self.stats = None

    def analyze(self, model_name="llama3"):
        """Analyze token distribution for input and output tokens."""
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found in ModelZoo")
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        input_tokens = []
        output_tokens = []

        # Compute token counts
        for _, row in self.data.iterrows():
            prompt = row['prompt']
            response = row['response']
            
            input_count = model.get_token_count(prompt)
            output_count = model.get_token_count(response) if response else 0

            # Filter based on paper's range
            if 8 <= input_count <= 2048 and (0 <= output_count <= 4096 or output_count == 0):
                input_tokens.append(input_count)
                output_tokens.append(output_count)
            else:
                self.logger.debug(f"Skipping out-of-range tokens: input={input_count}, output={output_count}")

        # Compute distributions
        input_series = pd.Series(input_tokens)
        output_series = pd.Series(output_tokens)
        
        self.distribution = {
            'input_distribution': input_series.value_counts().sort_index().to_dict(),
            'output_distribution': output_series.value_counts().sort_index().to_dict()
        }

        # Compute statistics
        self.stats = {
            'input': {
                'mean': input_series.mean(),
                'median': input_series.median(),
                'std': input_series.std(),
                'min': input_series.min(),
                'max': input_series.max()
            },
            'output': {
                'mean': output_series.mean(),
                'median': output_series.median(),
                'std': output_series.std(),
                'min': output_series.min(),
                'max': output_series.max()
            }
        }

        # Save distribution and stats
        dist_path = self.output_dir / 'token_distribution.pkl'
        with open(dist_path, 'wb') as f:
            pickle.dump({'distribution': self.distribution, 'stats': self.stats}, f)
        self.logger.info(f"Saved token distribution and stats to {dist_path}")

        # Visualize distribution
        self._visualize_distribution()
        return self.distribution, self.stats

    def _visualize_distribution(self):
        """Generate and save token distribution plots, replicating Figure 3 in paper."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Input tokens
        input_dist = self.distribution['input_distribution']
        ax1.bar(list(input_dist.keys()), list(input_dist.values()), width=10)
        ax1.set_title('Input Token Distribution')
        ax1.set_xlabel('Token Count')
        ax1.set_ylabel('Frequency')
        ax1.set_yscale('log')  # Log scale for better visualization
        ax1.grid(True, which="both", ls="--")
        ax1.set_xlim(0, 2048)

        # Output tokens
        output_dist = self.distribution['output_distribution']
        ax2.bar(list(output_dist.keys()), list(output_dist.values()), width=10)
        ax2.set_title('Output Token Distribution')
        ax2.set_xlabel('Token Count')
        ax2.set_ylabel('Frequency')
        ax2.set_yscale('log')
        ax2.grid(True, which="both", ls="--")
        ax2.set_xlim(0, 4096)

        plt.tight_layout()
        plot_path = self.output_dir / 'token_distribution.png'
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved distribution plot to {plot_path}")

    def get_distribution(self):
        """Return token distribution and stats."""
        if self.distribution is None or self.stats is None:
            self.logger.warning("Distribution not analyzed. Call analyze() first.")
            return {}, {}
        return self.distribution, self.stats

