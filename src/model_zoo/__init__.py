# hybrid-llm-inference/src/model_zoo/__init__.py
from .falcon import LocalFalcon, APIFalcon
from .llama3 import LocalLlama3, APILlama3
from .mistral import LocalMistral, APIMistral
from .tinyllama import LocalTinyLlama

def get_model(model_name, mode, config):
    """
    Factory function to get model instance.
    
    Args:
        model_name (str): Model name (e.g., 'falcon', 'llama3', 'mistral').
        mode (str): 'local' or 'api'.
        config (dict): Model configuration.
    
    Returns:
        BaseModel: Model instance.
    """
    model_map = {
        "falcon": {"local": LocalFalcon, "api": APIFalcon},
        "llama3": {"local": LocalLlama3, "api": APILlama3},
        "mistral": {"local": LocalMistral, "api": APIMistral},
        "tinyllama": {"local": LocalTinyLlama}  # TinyLlama 只支持本地模式
    }
    if model_name not in model_map or mode not in model_map[model_name]:
        raise ValueError(f"Unsupported model {model_name} or mode {mode}")
    return model_map[model_name][mode](config["model_name"], config)
