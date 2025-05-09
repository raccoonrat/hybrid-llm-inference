import json
import pandas as pd
import pickle
import pytest
import numpy as np
from src.data_processing.token_processing import TokenProcessing
from src.dataset_manager.token_distribution import TokenDistribution

def test_pipeline_token_distribution(tmp_path):
    # 1. 加载原始数据
    with open("data/alpaca_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # 2. 转为 DataFrame，字段名兼容
    for item in data:
        if "input" in item:
            item["instruction"] = item["input"]
        if "output" not in item:
            item["output"] = ""  # 确保有输出字段
    df = pd.DataFrame(data)
    # 3. 构造 mock model_config
    model_name = "mock"
    model_config = {"model_path": "mock_path"}
    # 4. 初始化 TokenProcessing
    token_processor = TokenProcessing(model_name=model_name, model_config=model_config)
    # 5. 处理输入token
    input_tasks = [{"input": row["instruction"]} for _, row in df.iterrows()]
    input_tokenized = token_processor.process_tokens(input_tasks)
    input_df = token_processor.get_token_data(pd.DataFrame(input_tokenized), format="dataframe")
    # 6. 处理输出token
    output_tasks = [{"input": row["output"]} for _, row in df.iterrows()]
    output_tokenized = token_processor.process_tokens(output_tasks)
    output_df = token_processor.get_token_data(pd.DataFrame(output_tokenized), format="dataframe")
    # 7. 分别计算输入输出分布
    input_dist = token_processor.compute_distribution(input_df)
    output_dist = token_processor.compute_distribution(output_df)
    # 8. 构造 TokenDistribution 并分析
    models = {"mock": type("MockModel", (), {"get_token_count": lambda self, x: len(x) if x else 0})()}
    td = TokenDistribution(df, models, output_dir=str(tmp_path))
    dist = td.analyze("mock")
    # 9. 保存分布（确保转换为Python原生类型）
    distribution_data = {
        "distribution": {
            "input_distribution": {int(k): float(v) for k, v in input_dist.items()},
            "output_distribution": {int(k): float(v) for k, v in output_dist.items()},
        },
        "stats": {
            "input_mean": float(np.mean(list(input_dist.keys()))),
            "input_std": float(np.std(list(input_dist.keys()))),
            "output_mean": float(np.mean(list(output_dist.keys()))),
            "output_std": float(np.std(list(output_dist.keys())))
        }
    }
    save_path = tmp_path / "token_distribution.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(distribution_data, f)
    # 10. 验证分布文件存在且内容合理
    assert save_path.exists()
    with open(save_path, "rb") as f:
        loaded = pickle.load(f)
    assert "distribution" in loaded
    assert "input_distribution" in loaded["distribution"]
    assert "output_distribution" in loaded["distribution"]
    # 11. 分布不应全为空
    assert len(loaded["distribution"]["input_distribution"]) > 0
    assert len(loaded["distribution"]["output_distribution"]) > 0
    # 12. 验证数据类型
    assert all(isinstance(k, int) for k in loaded["distribution"]["input_distribution"].keys())
    assert all(isinstance(v, float) for v in loaded["distribution"]["input_distribution"].values())
    assert all(isinstance(k, int) for k in loaded["distribution"]["output_distribution"].keys())
    assert all(isinstance(v, float) for v in loaded["distribution"]["output_distribution"].values()) 