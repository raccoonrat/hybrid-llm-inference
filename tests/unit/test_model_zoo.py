# hybrid-llm-inference/tests/unit/test_model_zoo.py
import pytest
from model_zoo import get_model

@pytest.fixture
def model_config():
    return {
        "model_name": "meta-llama/Llama-3-8B",
        "mode": "local",
        "max_length": 512,
        "api_key": "dummy_key"
    }

def test_local_llama3(model_config, monkeypatch):
    def mock_from_pretrained(name): return None
    def mock_generate(self, **kwargs): return [0]
    def mock_decode(self, tokens, **kwargs): return "Mock output"
    def mock_encode(self, text, **kwargs): return [[1, 2, 3]]
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", mock_from_pretrained)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.generate", mock_generate)
    monkeypatch.setattr("transformers.AutoTokenizer.decode", mock_decode)
    monkeypatch.setattr("transformers.AutoTokenizer.encode", mock_encode)
    
    model = get_model("llama3", "local", model_config)
    output = model.infer("Test input")
    token_count = model.get_token_count("Test input")
    
    assert output == "Mock output"
    assert token_count == 3

def test_api_llama3(model_config, monkeypatch):
    class MockResponse:
        def json(self): return [{"generated_text": "Mock output"}]
        def raise_for_status(self): pass
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: MockResponse())
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda name: None)
    monkeypatch.setattr("transformers.AutoTokenizer.encode", lambda self, text, **kwargs: [[1, 2, 3]])
    
    model = get_model("llama3", "api", model_config)
    output = model.infer("Test input")
    token_count = model.get_token_count("Test input")
    
    assert output == "Mock output"
    assert token_count == 3

def test_invalid_model(model_config):
    with pytest.raises(ValueError, match="Unsupported model invalid or mode local"):
        get_model("invalid", "local", model_config)
