# hybrid-llm-inference/tests/unit/test_data_processing.py
import pytest
import pandas as pd
from pathlib import Path
from data_processing.data_loader import DataLoader
from data_processing.token_processing import TokenProcessing
from model_zoo import get_model

@pytest.fixture
def sample_data(tmp_path):
    """Create a sample Alpaca dataset."""
    data = [
        {"prompt": "Write a short story", "response": "Once upon a time..."},
        {"prompt": "Explain AI", "response": "AI is the simulation of human intelligence..."}
    ]
    data_path = tmp_path / "alpaca_prompts.json"
    with open(data_path, 'w') as f:
        json.dump(data, f)
    return data_path

@pytest.fixture
def model():
    """Create a mock model for testing."""
    config = {"model_name": "meta-llama/Llama-3-8B", "mode": "local", "max_length": 512}
    return get_model("llama3", "local", config)

def test_data_loader(sample_data):
    loader = DataLoader(sample_data)
    data = loader.load()
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert list(data.columns) == ['prompt', 'response']

def test_token_processing(sample_data, model, tmp_path):
    # Load data
    loader = DataLoader(sample_data)
    data = loader.load()

    # Process tokens
    processor = TokenProcessing(data, {"llama3": model}, output_dir=tmp_path)
    token_data = processor.process_tokens(model_name="llama3")
    
    assert isinstance(token_data, pd.DataFrame)
    assert len(token_data) == 2
    assert set(token_data.columns) == {'prompt', 'response', 'input_tokens', 'output_tokens'}
    assert all(token_data['input_tokens'] >= 8)
    assert all(token_data['output_tokens'] >= 8)

    # Compute distribution
    distribution = processor.compute_distribution()
    assert 'input_distribution' in distribution
    assert 'output_distribution' in distribution
    assert (tmp_path / 'token_distribution.pkl').exists()
    assert (tmp_path / 'token_distribution.png').exists()
