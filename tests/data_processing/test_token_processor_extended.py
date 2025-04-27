"""TokenProcessor 和 TokenProcessing 的扩展测试用例。"""

import os
os.environ['TEST_MODE'] = '1'

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.data_processing.token_processor import TokenProcessor, MockTokenizer
from src.data_processing.token_processing import TokenProcessing
from src.data_processing.data_processor import DataProcessor
from src.data_processing.alpaca_loader import AlpacaLoader
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import time
import tempfile

@pytest.fixture
def model_path(tmp_path):
    """创建测试用的模型路径。"""
    model_dir = tmp_path / "models" / "TinyLlama-1.1B-Chat-v1.0"
    model_dir.mkdir(parents=True)
    return str(model_dir)

@pytest.fixture
def invalid_model_path(tmp_path):
    """创建无效的模型路径。"""
    return str(tmp_path / "nonexistent_model")

@pytest.fixture
def large_text_data():
    """创建大文本数据。"""
    return ["Hello world! " * 1000,  # 长文本
            "",  # 空文本
            "Special chars: !@#$%^&*()",  # 特殊字符
            "Numbers: 1234567890",  # 数字
            "Unicode: 你好世界🌍"  # Unicode字符
            ]

# TokenProcessor 的扩展测试

def test_token_processor_invalid_model_path(invalid_model_path):
    """测试使用无效的模型路径初始化 TokenProcessor。"""
    with pytest.raises(ValueError, match="模型路径不存在"):
        TokenProcessor(model_path=invalid_model_path, validate_path=True)

def test_token_processor_empty_input(model_path):
    """测试处理空输入。"""
    processor = TokenProcessor(model_path=model_path)
    
    # 测试空字符串
    assert processor.process("") == []
    
    # 测试空列表批处理
    assert processor.batch_process([]) == []
    
    # 测试 None 输入
    with pytest.raises(ValueError, match="输入文本不能为 None"):
        processor.process(None)

def test_token_processor_special_characters(model_path):
    """测试处理特殊字符。"""
    processor = TokenProcessor(model_path=model_path)
    special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\"
    
    tokens = processor.process(special_chars)
    decoded = processor.decode(tokens)
    assert decoded == special_chars

def test_token_processor_unicode(model_path):
    """测试处理 Unicode 字符。"""
    processor = TokenProcessor(model_path=model_path)
    unicode_text = "你好世界🌍"
    
    tokens = processor.process(unicode_text)
    decoded = processor.decode(tokens)
    assert decoded == unicode_text

def test_token_processor_large_input(model_path):
    """测试处理大输入。"""
    processor = TokenProcessor(model_path=model_path)
    large_text = "Hello world! " * 1000
    
    tokens = processor.process(large_text)
    assert len(tokens) > 1000
    decoded = processor.decode(tokens)
    assert decoded == large_text

def test_token_processor_batch_process_mixed(model_path, large_text_data):
    """测试批处理混合输入。"""
    processor = TokenProcessor(model_path=model_path)
    results = processor.batch_process(large_text_data)
    
    assert len(results) == len(large_text_data)
    for text, tokens in zip(large_text_data, results):
        decoded = processor.decode(tokens)
        assert decoded == text

def test_token_processor_max_length(model_path):
    """测试最大长度限制。"""
    processor = TokenProcessor(model_path=model_path, max_length=10)
    long_text = "This is a very long text that should be truncated"
    
    tokens = processor.process(long_text)
    assert len(tokens) <= 10

# TokenProcessing 的扩展测试

def test_token_processing_invalid_format(model_path, mock_dataframe):
    """测试无效的输出格式。"""
    processor = TokenProcessing(model_path)
    with pytest.raises(ValueError, match="不支持的格式"):
        processor.get_token_data(mock_dataframe, format='invalid')

def test_token_processing_distribution_empty_data(model_path):
    """测试空数据的分布计算。"""
    processor = TokenProcessing(model_path)
    empty_df = pd.DataFrame(columns=["input_tokens"])
    
    distribution = processor.compute_distribution(empty_df)
    assert isinstance(distribution, dict)
    assert len(distribution) == 0

def test_token_processing_distribution_single_token(model_path):
    """测试单个令牌的分布计算。"""
    processor = TokenProcessing(model_path)
    df = pd.DataFrame({
        "input_tokens": [[1]]
    })
    
    distribution = processor.compute_distribution(df)
    assert isinstance(distribution, dict)
    assert len(distribution) == 1
    assert list(distribution.values())[0] == 1.0

def test_token_processing_save_distribution_invalid_path(model_path, mock_dataframe):
    """测试保存分布图到各种无效路径的情况。"""
    processor = TokenProcessing(model_path)
    
    # 测试不同类型的无效路径
    invalid_paths = [
        "",  # 空路径
        "invalid",  # 无扩展名
        "plot",  # 无扩展名
        "plot.",  # 无效扩展名
        "plot.invalid",  # 无效扩展名
        "plot*.png",  # 包含通配符
        "plot?.png",  # 包含通配符
        "plot<>.png",  # 包含无效字符
        "plot|.png",  # 包含无效字符
        "plot\0.png",  # 包含空字符
        "   .png",    # 全空格文件名
        "plot.png ",  # 尾部空格
        " plot.png",  # 开头空格
        ".",         # 当前目录
        "..",        # 父目录
    ]
    
    if os.name == 'nt':
        # Windows特定的无效路径
        invalid_paths.extend([
            "CON.png",  # Windows保留名称
            "PRN.png",
            "AUX.png",
            "NUL.png",
            os.path.join("C:", "Windows", "System32", "plot.png"),  # 系统目录
        ])
    
    for invalid_path in invalid_paths:
        with pytest.raises(ValueError):
            processor.compute_distribution(mock_dataframe, save_path=invalid_path)

def test_token_processing_large_dataset(model_path):
    """测试处理大数据集。"""
    processor = TokenProcessing(model_path)
    large_df = pd.DataFrame({
        "text": ["Hello world! " * 100] * 100
    })
    
    result = processor.process_tokens(large_df["text"])
    assert len(result) == len(large_df)
    assert all(len(tokens) > 0 for tokens in result["input_tokens"])

def test_token_processing_mixed_data_types(model_path):
    """测试处理混合数据类型。"""
    processor = TokenProcessing(model_path)
    mixed_data = pd.DataFrame({
        "text": [
            "Normal text",
            123,  # 数字
            True,  # 布尔值
            None,  # 空值
            ["list", "of", "items"]  # 列表
        ]
    })
    
    # 应该能处理所有类型，将它们转换为字符串
    result = processor.process_tokens(mixed_data["text"])
    assert len(result) == len(mixed_data)

def test_token_processing_concurrent_processing(model_path):
    """测试并发处理。"""
    processor = TokenProcessing(model_path)
    large_df = pd.DataFrame({
        "text": ["Hello world! " * 100] * 100
    })
    
    # 使用多个线程处理
    result1 = processor.process_tokens(large_df["text"])
    result2 = processor.process_tokens(large_df["text"])
    
    # 结果应该相同
    pd.testing.assert_frame_equal(result1, result2)

def test_token_processing_visualization(model_path, tmp_path):
    """测试token分布可视化功能"""
    processor = TokenProcessing(model_path)
    
    # 准备测试数据
    df = pd.DataFrame({
        'input_tokens': [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    })
    
    # 测试保存到临时目录
    save_path = os.path.join(tmp_path, "distribution.png")
    distribution = processor.compute_distribution(df, save_path)
    
    # 验证结果
    assert os.path.exists(save_path)
    assert isinstance(distribution, dict)
    assert len(distribution) > 0
    
def test_token_processing_empty_visualization(model_path, tmp_path):
    """测试空数据的可视化处理"""
    processor = TokenProcessing(model_path)
    
    # 准备空数据
    df = pd.DataFrame({
        'input_tokens': []
    })
    
    # 测试保存到临时目录
    save_path = os.path.join(tmp_path, "empty_distribution.png")
    distribution = processor.compute_distribution(df, save_path)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert len(distribution) == 0
    
def test_token_processing_invalid_tokens(model_path):
    """测试无效token数据的处理"""
    processor = TokenProcessing(model_path)
    
    # 准备包含None和非列表数据的DataFrame
    df = pd.DataFrame({
        'input_tokens': [None, 123, [1, 2, 3], "invalid"]
    })
    
    # 测试分布计算
    distribution = processor.compute_distribution(df)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert len(distribution) > 0  # 应该只包含有效的token
    
def test_token_processing_write_permission(model_path, tmp_path):
    """测试写入权限检查"""
    processor = TokenProcessing(model_path)
    
    # 准备测试数据
    df = pd.DataFrame({
        'input_tokens': [[1, 2, 3]]
    })
    
    # 创建一个只读目录
    readonly_dir = os.path.join(tmp_path, "readonly")
    os.makedirs(readonly_dir)
    save_path = os.path.join(readonly_dir, "distribution.png")
    
    # 创建一个空文件并设置为只读
    with open(save_path, 'w') as f:
        f.write('')
    
    # 在Windows上设置文件和目录为只读
    if os.name == 'nt':
        os.system(f'attrib +r "{save_path}"')
        os.system(f'attrib +r "{readonly_dir}"')
    else:
        os.chmod(readonly_dir, 0o444)
        os.chmod(save_path, 0o444)
    
    # 测试写入权限检查
    with pytest.raises(ValueError, match="没有写入权限"):
        processor.compute_distribution(df, save_path)
        
def test_token_processing_mixed_columns(model_path):
    """测试同时包含input_tokens和token列的情况"""
    processor = TokenProcessing(model_path)
    
    # 准备包含两种列的数据
    df = pd.DataFrame({
        'input_tokens': [[1, 2, 3], [4, 5, 6]],
        'token': ['a', 'b']
    })
    
    # 测试分布计算（应该优先使用input_tokens）
    distribution = processor.compute_distribution(df)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert len(distribution) > 0
    assert 1 in [float(k) for k in distribution.keys()]  # 确认使用了input_tokens列
    
def test_token_processing_visualization_error(model_path, tmp_path):
    """测试可视化过程中的错误处理"""
    processor = TokenProcessing(model_path)
    
    # 准备测试数据
    df = pd.DataFrame({
        'input_tokens': [[1, 2, 3]]
    })
    
    # 使用无效的文件名（包含Windows上的非法字符）
    save_path = os.path.join(tmp_path, "test<>:.png")
    
    # 测试错误处理
    with pytest.raises(ValueError, match="包含无效字符"):
        processor.compute_distribution(df, save_path)

def test_token_processing_file_permissions(model_path, mock_dataframe, tmp_path):
    """测试文件权限相关的场景。"""
    import os
    import stat
    processor = TokenProcessing(model_path)

    # 创建一个只读目录
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    
    # 创建一个只读文件
    test_file = readonly_dir / "plot.png"
    test_file.touch()
    
    if os.name == 'nt':
        # Windows系统下设置目录和文件为只读
        import subprocess
        subprocess.run(['attrib', '+r', str(readonly_dir)], check=True)
        subprocess.run(['attrib', '+r', str(test_file)], check=True)
    else:
        # Unix系统下设置权限
        readonly_dir.chmod(stat.S_IREAD | stat.S_IXUSR)
        test_file.chmod(stat.S_IREAD)

    try:
        # 测试保存到只读目录
        save_path = str(test_file)
        with pytest.raises(ValueError, match="没有写入权限"):
            processor.compute_distribution(mock_dataframe, save_path=save_path)

    finally:
        # 恢复权限以便清理
        if os.name == 'nt':
            subprocess.run(['attrib', '-r', str(readonly_dir)], check=True)
            subprocess.run(['attrib', '-r', str(test_file)], check=True)
        else:
            readonly_dir.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
            if test_file.exists():
                test_file.chmod(stat.S_IREAD | stat.S_IWRITE)

def test_token_processing_directory_permissions(model_path, tmp_path):
    """测试目录权限的详细场景。"""
    import os
    import stat
    processor = TokenProcessing(model_path)
    df = pd.DataFrame({"input_tokens": [[1, 2, 3]]})

    # 创建嵌套的目录结构
    nested_dir = tmp_path / "parent" / "child"
    nested_dir.mkdir(parents=True)
    
    if os.name == 'nt':
        # Windows系统下设置目录为只读
        import subprocess
        subprocess.run(['attrib', '+r', str(nested_dir.parent)], check=True)
    else:
        # Unix系统下设置权限
        nested_dir.parent.chmod(stat.S_IREAD | stat.S_IXUSR)

    try:
        save_path = str(nested_dir / "plot.png")
        with pytest.raises(ValueError, match="没有写入权限"):
            processor.compute_distribution(df, save_path=save_path)
    finally:
        # 恢复权限以便清理
        if os.name == 'nt':
            subprocess.run(['attrib', '-r', str(nested_dir.parent)], check=True)
        else:
            nested_dir.parent.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)

def test_token_processing_path_validation(model_path, tmp_path):
    """测试路径验证逻辑的各种场景。"""
    processor = TokenProcessing(model_path)
    df = pd.DataFrame({"input_tokens": [[1, 2, 3]]})
    
    # 测试相对路径
    relative_path = "plot.png"
    with pytest.raises(ValueError, match="必须使用绝对路径"):
        processor.compute_distribution(df, save_path=relative_path)
    
    # 测试不存在的目录
    nonexistent_dir = os.path.join(tmp_path, "nonexistent", "plot.png")
    with pytest.raises(ValueError, match="目录不存在"):
        processor.compute_distribution(df, save_path=nonexistent_dir)
    
    # 测试特殊字符路径（Windows特定）
    if os.name == 'nt':
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            invalid_path = os.path.join(tmp_path, f"test{char}plot.png")
            with pytest.raises(ValueError, match="包含无效字符"):
                processor.compute_distribution(df, save_path=invalid_path)
    
    # 测试有效的绝对路径
    valid_path = os.path.join(tmp_path, "plot.png")
    distribution = processor.compute_distribution(df, save_path=valid_path)
    assert os.path.exists(valid_path)
    assert isinstance(distribution, dict)

def test_token_processing_long_path(model_path, tmp_path):
    """测试长路径场景。"""
    processor = TokenProcessing(model_path)
    df = pd.DataFrame({"input_tokens": [[1, 2, 3]]})
    
    # 创建一个非常长的路径
    long_path_components = ["subfolder"] * 50
    long_path = tmp_path
    for component in long_path_components:
        long_path = long_path / component
    
    # 在Windows系统下使用长路径前缀
    if os.name == 'nt':
        long_path_str = str(long_path)
        if len(long_path_str) > 260:
            long_path_str = '\\\\?\\' + long_path_str
        long_path = Path(long_path_str)
    
    long_path.mkdir(parents=True, exist_ok=True)
    
    save_path = str(long_path / "plot.png")
    
    if os.name == 'nt' and len(save_path) > 260:  # Windows MAX_PATH限制
        with pytest.raises(ValueError, match="路径长度超过系统限制"):
            processor.compute_distribution(df, save_path=save_path)
    else:
        distribution = processor.compute_distribution(df, save_path=save_path)
        assert os.path.exists(save_path)
        assert isinstance(distribution, dict)

def test_token_processing_unc_path(model_path):
    """测试UNC路径场景。"""
    processor = TokenProcessing(model_path)
    df = pd.DataFrame({"input_tokens": [[1, 2, 3]]})
    
    # 使用一个无效的UNC路径
    if os.name == 'nt':
        unc_path = r"\\nonexistent\share\plot.png"
        with pytest.raises(ValueError, match="无法访问网络路径"):
            processor.compute_distribution(df, save_path=unc_path)

def test_token_processing_unicode_path(model_path, tmp_path):
    """测试Unicode路径场景。"""
    processor = TokenProcessing(model_path)
    df = pd.DataFrame({"input_tokens": [[1, 2, 3]]})
    
    # 测试不同语言的文件名
    test_paths = {
        "中文路径": tmp_path / "测试" / "分布图.png",
        "日文路径": tmp_path / "テスト" / "図.png",
        "韩文路径": tmp_path / "테스트" / "그래프.png",
        "俄文路径": tmp_path / "тест" / "график.png"
    }
    
    for name, path in test_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        distribution = processor.compute_distribution(df, save_path=str(path))
        assert os.path.exists(path), f"{name}创建失败"
        assert isinstance(distribution, dict)

def test_token_processing_reserved_names(mock_dataframe, tmp_path):
    """测试Windows保留文件名的处理。
    
    验证使用Windows保留名称（如CON、PRN等）作为文件名时是否正确抛出异常。
    """
    processor = TokenProcessing("mock_model_path")
    distribution = processor.compute_distribution(mock_dataframe)
    
    reserved_names = ['CON.png', 'PRN.png', 'AUX.png', 'NUL.png', 'COM1.png']
    for name in reserved_names:
        save_path = os.path.join(tmp_path, name)
        with pytest.raises(ValueError, match="文件名不能使用Windows保留名称"):
            processor._save_distribution_plot(distribution, save_path)

def test_token_processing_empty_path(mock_dataframe):
    """测试空路径和空白字符路径的处理。
    
    验证空路径、空白字符路径和None值是否正确抛出异常。
    """
    processor = TokenProcessing("mock_model_path")
    distribution = processor.compute_distribution(mock_dataframe)
    
    invalid_paths = ['', ' ', '  ', None]
    for path in invalid_paths:
        with pytest.raises(ValueError, match="保存路径不能为空"):
            processor._save_distribution_plot(distribution, path)

def test_token_processing_extension_validation(mock_dataframe, tmp_path):
    """测试文件扩展名验证。
    
    验证：
    1. 无扩展名文件
    2. 错误扩展名（非.png）
    3. 大写PNG扩展名
    """
    processor = TokenProcessing("mock_model_path")
    distribution = processor.compute_distribution(mock_dataframe)
    
    # 测试无扩展名
    no_ext_path = os.path.join(tmp_path, "no_extension")
    with pytest.raises(ValueError, match="必须是.png格式"):
        processor._save_distribution_plot(distribution, no_ext_path)
    
    # 测试错误扩展名
    invalid_exts = ['.jpg', '.pdf', '.txt']
    for ext in invalid_exts:
        invalid_path = os.path.join(tmp_path, f"test{ext}")
        with pytest.raises(ValueError, match="必须是.png格式"):
            processor._save_distribution_plot(distribution, invalid_path)
    
    # 测试大写PNG扩展名（应该成功）
    valid_path = os.path.join(tmp_path, "test.PNG")
    processor._save_distribution_plot(distribution, valid_path)
    assert os.path.exists(valid_path)
    assert isinstance(distribution, dict)

def test_token_processing_process_tokens(model_path):
    """测试process_tokens方法的各种输入情况。"""
    processor = TokenProcessing(model_path)
    
    # 测试空输入
    empty_result = processor.process_tokens([])
    assert isinstance(empty_result, pd.DataFrame)
    assert empty_result.empty
    assert list(empty_result.columns) == ["input_tokens", "decoded_text"]
    
    # 测试Series输入
    series_input = pd.Series(["Hello", "World"])
    series_result = processor.process_tokens(series_input)
    assert isinstance(series_result, pd.DataFrame)
    assert len(series_result) == 2
    
    # 测试None值处理
    none_result = processor.process_tokens([None, "Valid"])
    assert isinstance(none_result, pd.DataFrame)
    assert len(none_result) == 2
    assert none_result["input_tokens"].iloc[0] == []
    assert none_result["decoded_text"].iloc[0] == ""
    
    # 测试非字符串类型
    mixed_result = processor.process_tokens([123, True, "Text"])
    assert isinstance(mixed_result, pd.DataFrame)
    assert len(mixed_result) == 3
    assert all(isinstance(text, str) for text in mixed_result["decoded_text"])

def test_token_processing_get_token_data(model_path, mock_dataframe):
    """测试get_token_data方法的各种情况。"""
    processor = TokenProcessing(model_path)
    
    # 测试无效格式
    with pytest.raises(ValueError, match="不支持的格式"):
        processor.get_token_data(mock_dataframe, format='invalid')
    
    # 测试空DataFrame
    empty_df = pd.DataFrame()
    empty_result = processor.get_token_data(empty_df)
    assert isinstance(empty_result, pd.DataFrame)
    assert empty_result.empty
    
    # 测试dict格式输出
    dict_result = processor.get_token_data(mock_dataframe, format='dict')
    assert isinstance(dict_result, dict)
    assert "input_tokens" in dict_result
    assert "decoded_text" in dict_result
    
    # 测试混合列名
    mixed_df = pd.DataFrame({
        "token": [[1, 2, 3]],
        "decoded_text": ["Test"]
    })
    mixed_result = processor.get_token_data(mixed_df)
    assert isinstance(mixed_result, pd.DataFrame)
    assert "input_tokens" in mixed_result.columns
    assert "decoded_text" in mixed_result.columns

def test_data_processor_validate_data():
    """测试DataProcessor的validate_data方法。"""
    processor = DataProcessor()
    
    # 测试None输入
    with pytest.raises(ValueError, match="输入数据不能为None"):
        processor.validate_data(None)
    
    # 测试空列表
    with pytest.raises(ValueError, match="输入数据列表不能为空"):
        processor.validate_data([])
    
    # 测试无效类型
    with pytest.raises(TypeError, match="输入数据必须是DataFrame、字典或字典列表"):
        processor.validate_data("invalid")
    
    # 测试单个字典
    single_dict = {"text": "test", "tokens": [1, 2, 3], "length": 3}
    result = processor.validate_data(single_dict)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    
    # 测试字典列表
    dict_list = [
        {"text": "test1", "tokens": [1, 2], "length": 2},
        {"text": "test2", "tokens": [3, 4], "length": 2}
    ]
    result = processor.validate_data(dict_list)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    
    # 测试缺少必需列
    invalid_df = pd.DataFrame({"text": ["test"]})
    with pytest.raises(ValueError, match="缺少必需的列"):
        processor.validate_data(invalid_df)

def test_data_processor_process_batch():
    """测试DataProcessor的process_batch方法。"""
    processor = DataProcessor()
    
    # 测试无效batch_size
    with pytest.raises(ValueError, match="batch_size必须是正整数"):
        processor.process_batch(pd.DataFrame(), batch_size=0)
    
    # 测试正常批处理
    data = pd.DataFrame({
        "text": ["test1", "test2", "test3"],
        "tokens": [[1, 2], [3, 4], [5, 6]],
        "length": [2, 2, 2]
    })
    
    batches = list(processor.process_batch(data, batch_size=2))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1

def test_alpaca_loader_validate_entry():
    """测试AlpacaLoader的validate_entry方法。"""
    loader = AlpacaLoader("dummy_path")
    
    # 测试有效条目
    valid_entry = {
        "instruction": "Test instruction",
        "input": "Test input",
        "output": "Test output"
    }
    assert loader.validate_entry(valid_entry) is True
    
    # 测试缺少字段
    missing_field = {"instruction": "Test"}
    assert loader.validate_entry(missing_field) is False
    
    # 测试非字符串字段
    invalid_type = {
        "instruction": 123,
        "input": "Test",
        "output": "Test"
    }
    assert loader.validate_entry(invalid_type) is False
    
    # 测试空instruction
    empty_instruction = {
        "instruction": "",
        "input": "Test",
        "output": "Test"
    }
    assert loader.validate_entry(empty_instruction) is False

def test_alpaca_loader_get_statistics():
    """测试AlpacaLoader的get_statistics方法。"""
    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = [
            {
                "instruction": "Test instruction 1",
                "input": "Test input 1",
                "output": "Test output 1"
            },
            {
                "instruction": "Test instruction 2",
                "input": "",
                "output": "Test output 2"
            }
        ]
        json.dump(test_data, f)
        test_file = f.name
    
    try:
        loader = AlpacaLoader(test_file)
        stats = loader.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_samples" in stats
        assert "avg_instruction_length" in stats
        assert "avg_input_length" in stats
        assert "avg_output_length" in stats
        assert "empty_input_ratio" in stats
        assert stats["total_samples"] == 2
        assert stats["empty_input_ratio"] == 0.5
    finally:
        os.unlink(test_file)

@pytest.fixture
def mock_dataframe():
    """创建模拟DataFrame。"""
    return pd.DataFrame({
        "text": ["Hello", "Hi", "Hey"],
        "input_tokens": [[1, 2, 3], [4, 5], [6, 7, 8]],
        "decoded_text": ["Hello", "Hi", "Hey"]
    })

def test_token_processing_large_distribution(model_path, tmp_path):
    """测试处理大量token的分布计算。"""
    processor = TokenProcessing(model_path)
    
    # 创建包含大量token的DataFrame
    large_df = pd.DataFrame({
        "input_tokens": [[i] * 1000 for i in range(100)]  # 100,000个token
    })
    
    # 测试计算分布
    distribution = processor.compute_distribution(large_df)
    assert isinstance(distribution, dict)
    assert len(distribution) == 100
    assert all(isinstance(k, str) for k in distribution.keys())
    assert all(isinstance(v, float) for v in distribution.values())
    
    # 测试保存大分布图
    save_path = os.path.join(tmp_path, "large_distribution.png")
    processor.compute_distribution(large_df, save_path=save_path)
    assert os.path.exists(save_path)

def test_token_processing_concurrent_operations(model_path, tmp_path):
    """测试并发操作场景。"""
    processor = TokenProcessing(model_path)
    
    # 创建测试数据
    df = pd.DataFrame({
        "input_tokens": [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    })
    
    # 同时进行多个操作
    save_path1 = os.path.join(tmp_path, "dist1.png")
    save_path2 = os.path.join(tmp_path, "dist2.png")
    
    # 计算分布并保存
    dist1 = processor.compute_distribution(df, save_path=save_path1)
    dist2 = processor.compute_distribution(df, save_path=save_path2)
    
    # 验证结果
    assert os.path.exists(save_path1)
    assert os.path.exists(save_path2)
    assert dist1 == dist2

def test_token_processing_error_handling(model_path):
    """测试错误处理场景。"""
    processor = TokenProcessing(model_path)
    
    # 测试无效的token数据
    invalid_df = pd.DataFrame({
        "input_tokens": [None, "invalid", [1, 2, 3]]
    })
    
    # 应该能够处理无效数据并返回空分布
    distribution = processor.compute_distribution(invalid_df)
    assert isinstance(distribution, dict)
    assert len(distribution) > 0  # 应该只包含有效token
    
    # 测试文件系统错误
    with pytest.raises(ValueError):
        processor.compute_distribution(
            invalid_df,
            save_path="/nonexistent/directory/plot.png"
        )

def test_token_processing_memory_usage(model_path):
    """测试内存使用情况。"""
    processor = TokenProcessing(model_path)
    
    # 创建大量数据
    large_data = ["test" * 1000] * 1000  # 1MB文本数据
    
    # 处理大量数据
    result = processor.process_tokens(large_data)
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(large_data)
    assert "input_tokens" in result.columns
    assert "decoded_text" in result.columns

def test_token_processing_performance(model_path):
    """测试性能相关场景。"""
    processor = TokenProcessing(model_path)
    
    # 创建测试数据
    test_data = ["test"] * 1000
    
    # 测量处理时间
    start_time = time.time()
    result = processor.process_tokens(test_data)
    end_time = time.time()
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(test_data)
    
    # 记录处理时间
    processing_time = end_time - start_time
    assert processing_time < 5.0  # 假设处理1000个样本应该在5秒内完成

def test_token_processing_resource_cleanup(model_path, tmp_path):
    """测试资源清理。"""
    processor = TokenProcessing(model_path)
    
    # 创建临时文件
    temp_file = os.path.join(tmp_path, "temp.png")
    
    # 执行操作
    df = pd.DataFrame({"input_tokens": [[1, 2, 3]]})
    processor.compute_distribution(df, save_path=temp_file)
    
    # 验证文件创建
    assert os.path.exists(temp_file)
    
    # 清理资源
    processor.cleanup()
    
    # 验证资源已清理
    assert not hasattr(processor, 'tokenizer')

def test_token_processing_special_characters(model_path):
    """测试特殊字符处理。"""
    processor = TokenProcessing(model_path)
    
    # 测试各种特殊字符
    special_chars = [
        "!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\",  # 标点符号
        "你好世界🌍",  # Unicode字符
        "测试\t换行\n测试",  # 空白字符
        "测试\r\nWindows换行",  # 不同系统的换行符
        "测试\x00空字符",  # 控制字符
    ]
    
    # 处理特殊字符
    result = processor.process_tokens(special_chars)
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(special_chars)
    assert all(isinstance(tokens, list) for tokens in result["input_tokens"])
    assert all(isinstance(text, str) for text in result["decoded_text"])

def test_token_processing_empty_tokens(model_path):
    """测试空token处理。"""
    processor = TokenProcessing(model_path)
    
    # 测试各种空token情况
    empty_cases = [
        [],  # 空列表
        [""],  # 空字符串
        [None],  # None值
        [[]],  # 空子列表
        [[], []],  # 多个空列表
    ]
    
    # 处理空token
    for case in empty_cases:
        df = pd.DataFrame({"input_tokens": case})
        distribution = processor.compute_distribution(df)
        assert isinstance(distribution, dict)
        assert len(distribution) == 0

def test_token_processing_mixed_data_types(model_path):
    """测试混合数据类型处理。"""
    processor = TokenProcessing(model_path)
    
    # 创建混合数据类型的DataFrame
    mixed_df = pd.DataFrame({
        "input_tokens": [
            [1, 2, 3],  # 整数列表
            ["1", "2", "3"],  # 字符串列表
            [1.0, 2.0, 3.0],  # 浮点数列表
            [True, False],  # 布尔值列表
            [None, None],  # None值列表
        ]
    })
    
    # 处理混合数据
    distribution = processor.compute_distribution(mixed_df)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert all(isinstance(k, str) for k in distribution.keys())
    assert all(isinstance(v, float) for v in distribution.values())

def test_token_processing_duplicate_tokens(model_path):
    """测试重复token处理。"""
    processor = TokenProcessing(model_path)
    
    # 创建包含重复token的数据
    duplicate_df = pd.DataFrame({
        "input_tokens": [
            [1, 1, 1],  # 连续重复
            [1, 2, 1],  # 间隔重复
            [1, 1, 2, 2],  # 多组重复
        ]
    })
    
    # 计算分布
    distribution = processor.compute_distribution(duplicate_df)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert len(distribution) == 2  # 应该只有两个不同的token
    assert sum(distribution.values()) == 1.0  # 概率和应该为1

def test_token_processing_nested_tokens(model_path):
    """测试嵌套token处理。"""
    processor = TokenProcessing(model_path)
    
    # 创建嵌套token数据
    nested_df = pd.DataFrame({
        "input_tokens": [
            [[1, 2], [3, 4]],  # 两层嵌套
            [[[1]], [[2]]],  # 三层嵌套
            [1, [2, [3]]],  # 混合嵌套
        ]
    })
    
    # 处理嵌套数据
    distribution = processor.compute_distribution(nested_df)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert all(isinstance(k, str) for k in distribution.keys())
    assert all(isinstance(v, float) for v in distribution.values())

def test_token_processing_large_numbers(model_path):
    """测试大数字处理。"""
    processor = TokenProcessing(model_path)
    
    # 创建包含大数字的数据
    large_numbers_df = pd.DataFrame({
        "input_tokens": [
            [2**32],  # 32位整数
            [2**64],  # 64位整数
            [1e100],  # 大浮点数
            [-2**32],  # 负大整数
        ]
    })
    
    # 处理大数字
    distribution = processor.compute_distribution(large_numbers_df)
    
    # 验证结果
    assert isinstance(distribution, dict)
    assert all(isinstance(k, str) for k in distribution.keys())
    assert all(isinstance(v, float) for v in distribution.values())

def test_token_processing_unicode_normalization(model_path):
    """测试Unicode标准化处理。"""
    processor = TokenProcessing(model_path)
    
    # 测试Unicode组合字符
    unicode_cases = [
        "café",  # 带重音
        "cafe\u0301",  # 组合重音
        "你好",  # 中文字符
        "こんにちは",  # 日文字符
        "안녕하세요",  # 韩文字符
    ]
    
    # 处理Unicode字符
    result = processor.process_tokens(unicode_cases)
    
    # 验证结果
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(unicode_cases)
    assert all(isinstance(tokens, list) for tokens in result["input_tokens"])
    assert all(isinstance(text, str) for text in result["decoded_text"]) 