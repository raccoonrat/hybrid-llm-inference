# hybrid-llm-inference/tests/unit/test_dataset_manager.py
import pytest
import pandas as pd
from pathlib import Path
from dataset_manager.alpaca_loader import AlpacaLoader
from data_processing.token_processing import TokenProcessing
from model_zoo import get_model

@pytest.fixture
def mock_dataset(tmp_path):
    data = pd.DataFrame([
        {"prompt": "Write a story", "response": "Once upon a time"},
        {"prompt": "Explain AI", "response": "AI is..."}
    ])
    dataset_path = tmp_path / "alpaca_prompts.json"
    data.to_json(dataset_path, orient="records")
    return dataset_path

@pytest.fixture
def model_config():
    return {
        "model_name": "meta-llama/Llama-3-8B",
        "mode": "local",
        "max_length": 512
    }

def test_alpaca_loader(mock_dataset):
    loader = AlpacaLoader(mock_dataset)
    data = loader.load()
    
    assert len(data) == 2
    assert list(data.columns) == ["prompt", "response"]
    assert data.iloc[0]["prompt"] == "Write a story"

def test_alpaca_loader_empty_file(tmp_path):
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]")
    loader = AlpacaLoader(empty_file)
    
    with pytest.raises(ValueError, match="Dataset is empty"):
        loader.load()

def test_data_processing(mock_dataset, model_config, monkeypatch):
    def mock_get_token_count(text): return 10
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", mock_get_token_count)
    
    loader = AlpacaLoader(mock_dataset)
    data = loader.load()
    model = get_model(model_config["model_name"], model_config["mode"], model_config)
    processor = TokenProcessing(data, {"llama3": model})
    token_data = processor.get_token_data()
    
    assert len(token_data) == 2
    assert token_data[0]["input_tokens"] == 10
    assert token_data[0]["output_tokens"] == 10
    assert token_data[0]["prompt"] == "Write a story"

def test_data_processing_no_response(mock_dataset, model_config, monkeypatch):
    def mock_get_token_count(text): return 10 if text else 0
    monkeypatch.setattr("model_zoo.base_model.BaseModel.get_token_count", mock_get_token_count)
    
    data = pd.DataFrame([{"prompt": "Write a story", "response": ""}])
    dataset_path = mock_dataset.parent / "no_response.json"
    data.to_json(dataset_path, orient="records")
    
    loader = AlpacaLoader(dataset_path)
    data = loader.load()
    model = get_model(model_config["model_name"], model_config["mode"], model_config)
    processor = TokenProcessing(data, {"llama3": model})
    token_data = processor.get_token_data()
    
    assert token_data[0]["output_tokens"] == 0
