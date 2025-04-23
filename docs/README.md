```markdown

# Hybrid LLM Inference

A system for energy-efficient LLM inference using hybrid heterogeneous clusters, based on the paper "Hybrid Heterogeneous Clusters Can Lower the Energy Consumption of LLM Inference Workloads".

## Features
- Token-based task scheduling for M1 Pro and A100 hardware.
- Energy profiling for NVIDIA GPU, Apple Silicon, Intel CPU, and AMD CPU.
- Optimization of energy-runtime tradeoffs using Alpaca dataset.
- Support for Falcon, Llama-2, and Mistral models.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

## Configuration
``` text
model_config.yaml: Specify model names and modes (local or api).
Set api_key for API mode in model_config.yaml.
```

## Contributing
See  for details.

## License
MIT License
See  for details.
```
