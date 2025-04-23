# hybrid-llm-inference/src/data_processing/token_processing.py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from toolbox.logger import get_logger
from model_zoo import get_model

class TokenProcessing:
    def __init__(self, data, models, output_dir="data/processed"):
        """
        Initialize TokenProcessing for computing token counts and distributions.
        
        Args:
            data (pd.DataFrame): DataFrame with 'prompt' and 'response' columns.
            models (dict): Dictionary of model instances from ModelZoo.
            output_dir (str): Directory to save token distribution.
        """
        self.data = data
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        self.token_data = None
        self.distribution = None

    def process_tokens(self, model_name="llama3"):
        """Compute input and output token counts for each prompt."""
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found in ModelZoo")
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        token_data = []

        for _, row in self.data.iterrows():
            prompt = row['prompt']
            response = row['response']
            
            # Compute token counts
            input_tokens = model.get_token_count(prompt)
            output_tokens = model.get_token_count(response) if response else 0
            
            # Filter tokens based on paper's range
            if not (8 <= input_tokens <= 2048 and (0 <= output_tokens <= 4096 or output_tokens == 0)):
                self.logger.warning(f"Skipping out-of-range tokens: input={input_tokens}, output={output_tokens}")
                continue

            token_data.append({
                'prompt': prompt,
                'response': response,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            })

        self.token_data = pd.DataFrame(token_data)
        self.logger.info(f"Processed {len(self.token_data)} prompts with token counts")
        return self.token_data

    def compute_distribution(self):
        """Compute and save token distribution."""
        if self.token_data is None:
            self.logger.error("Token data not processed. Call process_tokens() first.")
            raise ValueError("Token data not processed")

        # Compute frequency distributions
        input_dist = self.token_data['input_tokens'].value_counts().sort_index()
        output_dist = self.token_data['output_tokens'].value_counts().sort_index()

        self.distribution = {
            'input_distribution': input_dist.to_dict(),
            'output_distribution': output_dist.to_dict()
        }

        # Save distribution
        dist_path = self.output_dir / 'token_distribution.pkl'
        with open(dist_path, 'wb') as f:
            pickle.dump(self.distribution, f)
        self.logger.info(f"Saved token distribution to {dist_path}")

        # Visualize distribution (replicate Figure 3 in paper)
        self._visualize_distribution()
        return self.distribution

    def _visualize_distribution(self):
        """Generate and save token distribution plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Input tokens
        input_dist = self.distribution['input_distribution']
        ax1.bar(input_dist.keys(), input_dist.values())
        ax1.set_title('Input Token Distribution')
        ax1.set_xlabel('Token Count')
        ax1.set_ylabel('Frequency')
        ax1.grid(True)

        # Output tokens
        output_dist = self.distribution['output_distribution']
        ax2.bar(output_dist.keys(), output_dist.values())
        ax2.set_title('Output Token Distribution')
        ax2.set_xlabel('Token Count')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)

        plt.tight_layout()
        plot_path = self.output_dir / 'token_distribution.png'
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved distribution plot to {plot_path}")

    def get_token_data(self):
        """Return processed token data."""
        if self.token_data is None:
            self.logger.warning("Token data not processed. Call process_tokens() first.")
            return pd.DataFrame()
        return self.token_data
