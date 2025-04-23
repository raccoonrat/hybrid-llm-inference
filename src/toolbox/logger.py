# hybrid-llm-inference/src/toolbox/logger.py
import logging
import os
from pathlib import Path

def get_logger(name, log_dir="logs", log_level=logging.INFO):
    """
    Configure and return a logger instance.
    
    Args:
        name (str): Logger name (usually module name).
        log_dir (str): Directory to save log files.
        log_level (int): Logging level (e.g., logging.INFO).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # Avoid duplicate handlers
        return logger
    
    logger.setLevel(log_level)
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path(log_dir) / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized for {name}")
    return logger

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    Setup root logger for the application.
    
    Args:
        log_dir (str): Directory to save log files.
        log_level (int): Logging level.
    """
    root_logger = get_logger("hybrid_llm", log_dir, log_level)
    root_logger.info("Root logger setup complete")

