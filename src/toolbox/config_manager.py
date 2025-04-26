# hybrid-llm-inference/src/toolbox/config_manager.py
import os
from pathlib import Path
from typing import Dict, Any, Optional
from toolbox.logger import get_logger

logger = get_logger(__name__)

class ConfigManager:
    """配置管理类，用于统一管理模型相关的配置。"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化配置管理器。

        Args:
            config: 初始配置字典
        """
        self.config = config or {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置的有效性。"""
        if not isinstance(self.config, dict):
            raise ValueError("配置必须是字典类型")
        
        # 验证模型配置
        if "model_config" in self.config:
            model_config = self.config["model_config"]
            if not isinstance(model_config, dict):
                raise ValueError("model_config 必须是字典类型")
            
            # 验证模型路径
            if "model_path" in model_config:
                model_path = model_config["model_path"]
                if not isinstance(model_path, str):
                    raise ValueError("model_path 必须是字符串类型")
                if not os.path.exists(model_path):
                    raise ValueError(f"模型路径不存在: {model_path}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置。

        Returns:
            Dict[str, Any]: 模型配置字典
        """
        return self.config.get("model_config", {})
    
    def get_model_path(self) -> Optional[str]:
        """获取模型路径。

        Returns:
            Optional[str]: 模型路径，如果未配置则返回 None
        """
        model_config = self.get_model_config()
        return model_config.get("model_path")
    
    def set_model_path(self, model_path: str) -> None:
        """设置模型路径。

        Args:
            model_path: 模型路径
        """
        if not isinstance(model_path, str):
            raise ValueError("model_path 必须是字符串类型")
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")
        
        if "model_config" not in self.config:
            self.config["model_config"] = {}
        self.config["model_config"]["model_path"] = model_path
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置。

        Returns:
            Dict[str, Any]: 完整配置字典
        """
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置。

        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        self._validate_config()
    
    def save_config(self, file_path: str) -> None:
        """保存配置到文件。

        Args:
            file_path: 配置文件路径
        """
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)
    
    @classmethod
    def load_config(cls, file_path: str) -> 'ConfigManager':
        """从文件加载配置。

        Args:
            file_path: 配置文件路径

        Returns:
            ConfigManager: 配置管理器实例
        """
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(config)

