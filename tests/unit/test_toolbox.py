# hybrid-llm-inference/tests/unit/test_toolbox.py
import pytest
import yaml
from pathlib import Path
from toolbox.logger import get_logger
from toolbox.config_manager import ConfigManager
from toolbox.accelerate_wrapper import AccelerateWrapper

@pytest.fixture
def tmp_log_dir(tmp_path):
    return tmp_path / "logs"

def test_logger(tmp_log_dir):
    logger = get_logger("test_logger", log_dir=tmp_log_dir)
    logger.info("Test message")
    
    log_file = tmp_log_dir / "test_logger.log"
    assert log_file.exists()
    with open(log_file, 'r') as f:
        content = f.read()
    assert "Test message" in content

def test_config_manager(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "test_config.yaml"
    config_content = {"key": "value"}
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    
    config_manager = ConfigManager(config_dir)
    config = config_manager.load_config("test_config.yaml")
    
    assert config == config_content
    assert config_manager.get_config("test_config.yaml") == config_content

def test_config_manager_invalid_yaml(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "invalid.yaml"
    config_file.write_text("invalid: yaml: content")
    
    config_manager = ConfigManager(config_dir)
    with pytest.raises(yaml.YAMLError):
        config_manager.load_config("invalid.yaml")

def test_accelerate_wrapper(monkeypatch):
    class MockModel:
        pass
    
    config = {"mixed_precision": "fp16", "device_placement": True}
    monkeypatch.setattr("accelerate.Accelerator.prepare", lambda self, model: model)
    
    wrapper = AccelerateWrapper(MockModel(), config)
    model = wrapper.get_model()
    assert isinstance(model, MockModel)

