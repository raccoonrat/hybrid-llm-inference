# hybrid-llm-inference/src/toolbox/config_manager.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from toolbox.logger import get_logger

logger = get_logger(__name__)

class ConfigManager:
    """配置管理器类,用于管理和验证配置。"""
    
    def __init__(self, config: Union[str, Dict[str, Any]]) -> None:
        """初始化配置管理器。

        Args:
            config: 配置目录路径或配置字典
        """
        self.configs = {}
        
        if isinstance(config, str):
            # 如果是字符串，则视为配置目录路径
            self.config_dir = Path(config)
            if not self.config_dir.exists():
                raise ValueError(f"配置目录不存在: {config}")
        else:
            # 如果是字典，则直接使用
            self.config_dir = None
            # 检查配置结构并进行转换
            if "model" in config and "hardware" in config and "scheduler" in config:
                # 如果是 SystemPipeline 风格的配置，转换为标准格式
                converted_config = self._convert_pipeline_config(config)
                for key, value in converted_config.items():
                    try:
                        self._validate_config(value, f"{key}.yaml")
                    except Exception as e:
                        logger.error(f"配置验证失败: {e}")
                        raise
                    self.configs[key] = value
            else:
                # 如果是标准格式，直接验证
                for key, value in config.items():
                    if key in ["hardware_config", "model_config", "scheduler_config"]:
                        try:
                            self._validate_config(value, f"{key}.yaml")
                        except Exception as e:
                            logger.error(f"配置验证失败: {e}")
                            raise
                    self.configs[key] = value
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """加载指定配置文件。

        Args:
            config_file: 配置文件名

        Returns:
            配置字典
        """
        # 如果是字典配置，直接返回对应部分
        if self.config_dir is None:
            config_key = config_file.replace(".yaml", "")
            if config_key in self.configs:
                config = self.configs[config_key]
                self._validate_config(config, config_file)
                return config
            return {}
            
        # 否则从文件加载
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        
        # 验证配置
        try:
            self._validate_config(config, config_file)
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            raise
        
        # 缓存配置
        self.configs[config_file.replace(".yaml", "")] = config
        return config
    
    def _validate_config(self, config: Dict[str, Any], config_file: str) -> None:
        """验证配置的有效性。

        Args:
            config: 配置字典
            config_file: 配置文件名
        """
        if not isinstance(config, dict):
            raise ValueError(f"{config_file}: 配置必须是字典类型")
        
        if "hardware_config.yaml" in config_file:
            self._validate_hardware_config(config)
        elif "model_config.yaml" in config_file:
            self._validate_model_config(config)
        elif "scheduler_config.yaml" in config_file:
            self._validate_scheduler_config(config)
    
    def _validate_hardware_config(self, config: Dict[str, Any]) -> None:
        """验证硬件配置。

        Args:
            config: 硬件配置字典
        """
        if "devices" not in config:
            raise ValueError("hardware_config.yaml: 缺少devices配置")
        
        if not isinstance(config["devices"], dict):
            raise ValueError("devices配置必须是字典类型")
        
        required_fields = {
            "device_type", "device_id", "idle_power", "memory_limit",
            "compute_capability", "priority"
        }
        
        for device_name, device_config in config["devices"].items():
            if not isinstance(device_config, dict):
                raise ValueError(f"设备配置必须是字典类型: {device_name}")
            
            missing_fields = required_fields - set(device_config.keys())
            if missing_fields:
                raise ValueError(f"设备配置缺少必要字段 {missing_fields}: {device_name}")
            
            # 验证字段类型
            if not isinstance(device_config["device_type"], str):
                raise ValueError(f"device_type必须是字符串类型: {device_name}")
            if not isinstance(device_config["device_id"], int):
                raise ValueError(f"device_id必须是整数类型: {device_name}")
            if not isinstance(device_config["idle_power"], (int, float)):
                raise ValueError(f"idle_power必须是数值类型: {device_name}")
            if not isinstance(device_config["memory_limit"], int):
                raise ValueError(f"memory_limit必须是整数类型: {device_name}")
            if not isinstance(device_config["compute_capability"], (int, float)):
                raise ValueError(f"compute_capability必须是数值类型: {device_name}")
            if not isinstance(device_config["priority"], int):
                raise ValueError(f"priority必须是整数类型: {device_name}")
    
    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """验证模型配置。

        Args:
            config: 模型配置字典
        """
        if "models" not in config:
            raise ValueError("model_config.yaml: 缺少models配置")
        
        if not isinstance(config["models"], dict):
            raise ValueError("models配置必须是字典类型")
        
        required_fields = {
            "model_name", "mode", "max_length", "mixed_precision",
            "device_placement"
        }
        
        for model_name, model_config in config["models"].items():
            if not isinstance(model_config, dict):
                raise ValueError(f"模型配置必须是字典类型: {model_name}")
            
            missing_fields = required_fields - set(model_config.keys())
            if missing_fields:
                raise ValueError(f"模型配置缺少必要字段 {missing_fields}: {model_name}")
            
            # 验证字段类型
            if not isinstance(model_config["model_name"], str):
                raise ValueError(f"model_name必须是字符串类型: {model_name}")
            if not isinstance(model_config["mode"], str):
                raise ValueError(f"mode必须是字符串类型: {model_name}")
            if not isinstance(model_config["max_length"], int):
                raise ValueError(f"max_length必须是整数类型: {model_name}")
            if not isinstance(model_config["mixed_precision"], str):
                raise ValueError(f"mixed_precision必须是字符串类型: {model_name}")
            if not isinstance(model_config["device_placement"], bool):
                raise ValueError(f"device_placement必须是布尔类型: {model_name}")
            
            # 验证模型路径
            if "model_path" in model_config:
                model_path = model_config["model_path"]
                if not isinstance(model_path, str):
                    raise ValueError(f"model_path必须是字符串类型: {model_name}")
                if not os.environ.get('TEST_MODE'):
                    if not os.path.exists(model_path):
                        logger.warning(f"模型路径不存在: {model_path}")
    
    def _validate_scheduler_config(self, config: Dict[str, Any]) -> None:
        """验证调度器配置。

        Args:
            config: 调度器配置字典
        """
        if "scheduler" not in config:
            raise ValueError("scheduler_config.yaml: 缺少scheduler配置")
        
        if not isinstance(config["scheduler"], dict):
            raise ValueError("scheduler配置必须是字典类型")
        
        scheduler_config = config["scheduler"]
        required_fields = {
            "max_batch_size", "max_queue_size", "max_wait_time",
            "token_threshold", "dynamic_threshold", "batch_processing",
            "device_priority"
        }
        
        missing_fields = required_fields - set(scheduler_config.keys())
        if missing_fields:
            raise ValueError(f"调度器配置缺少必要字段: {missing_fields}")
        
        # 验证字段类型
        if not isinstance(scheduler_config["max_batch_size"], int):
            raise ValueError("max_batch_size必须是整数类型")
        if not isinstance(scheduler_config["max_queue_size"], int):
            raise ValueError("max_queue_size必须是整数类型")
        if not isinstance(scheduler_config["max_wait_time"], (int, float)):
            raise ValueError("max_wait_time必须是数值类型")
        if not isinstance(scheduler_config["token_threshold"], int):
            raise ValueError("token_threshold必须是整数类型")
        if not isinstance(scheduler_config["dynamic_threshold"], bool):
            raise ValueError("dynamic_threshold必须是布尔类型")
        if not isinstance(scheduler_config["batch_processing"], bool):
            raise ValueError("batch_processing必须是布尔类型")
        if not isinstance(scheduler_config["device_priority"], list):
            raise ValueError("device_priority必须是列表类型")
        
        # 验证监控配置
        if "monitoring" in scheduler_config:
            monitoring_config = scheduler_config["monitoring"]
            if not isinstance(monitoring_config, dict):
                raise ValueError("monitoring配置必须是字典类型")
            if "sample_interval" not in monitoring_config:
                raise ValueError("监控配置缺少sample_interval")
            if "metrics" not in monitoring_config:
                raise ValueError("监控配置缺少metrics")
            if not isinstance(monitoring_config["sample_interval"], int):
                raise ValueError("sample_interval必须是整数类型")
            if not isinstance(monitoring_config["metrics"], list):
                raise ValueError("metrics必须是列表类型")
    
    def get_config(self, config_file: str) -> Dict[str, Any]:
        """获取已加载的配置。

        Args:
            config_file: 配置文件名

        Returns:
            配置字典
        """
        config_key = config_file.replace(".yaml", "")
        if config_key not in self.configs:
            raise ValueError(f"配置未加载: {config_file}")
        return self.configs[config_key]
    
    def get_dataset_path(self) -> Optional[str]:
        """获取数据集路径。

        Returns:
            数据集路径，如果未配置则返回None
        """
        return self.configs.get("dataset_path")
    
    def get_output_dir(self) -> str:
        """获取输出目录。

        Returns:
            输出目录路径
        """
        return self.configs.get("output_dir", "output")
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """获取硬件配置。

        Returns:
            硬件配置字典
        """
        return self.configs.get("hardware_config", {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置。

        Returns:
            模型配置字典
        """
        return self.configs.get("model_config", {})
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """获取调度器配置。

        Returns:
            调度器配置字典
        """
        return self.configs.get("scheduler_config", {})

    def _convert_pipeline_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """将 SystemPipeline 风格的配置转换为标准格式。

        Args:
            config: SystemPipeline 风格的配置字典

        Returns:
            Dict[str, Any]: 标准格式的配置字典
        """
        # 转换硬件配置
        hardware_config = {
            "devices": {
                "rtx4050": {
                    "device_type": config["hardware"]["device_type"],
                    "device_id": config["hardware"]["device_id"],
                    "idle_power": config["hardware"]["idle_power"],
                    "memory_limit": config["hardware"].get("memory_limit", 6144),  # 默认 6GB
                    "compute_capability": config["hardware"].get("compute_capability", 8.9),
                    "priority": config["hardware"].get("priority", 1),
                    "sample_interval": config["hardware"].get("sample_interval", 200)
                }
            }
        }

        # 转换模型配置
        model_config = {
            "models": {
                config["model"]["model_name"]: {
                    "model_name": config["model"]["model_name"],
                    "model_path": config["model"]["model_path"],
                    "device": config["model"].get("device", "cuda"),
                    "dtype": config["model"].get("dtype", "float32"),
                    "mode": config["model"].get("mode", "local"),
                    "max_length": config["model"].get("max_length", 512),
                    "mixed_precision": config["model"].get("mixed_precision", "fp16"),
                    "device_placement": config["model"].get("device_placement", True),
                    "batch_size": config["model"].get("batch_size", 1)
                }
            }
        }

        # 转换调度器配置
        scheduler_config = {
            "scheduler": {
                "max_batch_size": config["scheduler"].get("max_batch_size", 4),
                "max_queue_size": config["scheduler"].get("max_queue_size", 100),
                "max_wait_time": config["scheduler"].get("max_wait_time", 1.0),
                "token_threshold": config["scheduler"].get("token_threshold", 512),
                "dynamic_threshold": config["scheduler"].get("dynamic_threshold", True),
                "batch_processing": config["scheduler"].get("batch_processing", True),
                "device_priority": ["rtx4050"],
                "monitoring": {
                    "sample_interval": config["hardware"].get("sample_interval", 200),
                    "metrics": ["power_usage", "memory_usage"]
                }
            }
        }

        return {
            "hardware_config": hardware_config,
            "model_config": model_config,
            "scheduler_config": scheduler_config
        }

