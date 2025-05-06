# main.py 调试计划

## 当前问题
1. 模块导入错误：
   - 错误：`ModuleNotFoundError: No module named 'dataset_manager.data_processing'`
   - 原因：模块路径已更改，但导入语句未更新

## 调试步骤

### 1. 修复导入问题
- [x] 更新 main.py 中的导入语句
- [ ] 验证所有依赖模块是否正确定位
- [ ] 检查 Python 路径设置

### 2. 配置文件验证
- [ ] 检查配置文件格式
- [ ] 验证配置文件路径
- [ ] 测试配置加载功能

### 3. 数据集处理
- [ ] 验证数据集加载
- [ ] 检查数据预处理
- [ ] 测试令牌分布计算

### 4. 模型推理
- [ ] 验证模型加载
- [ ] 测试推理功能
- [ ] 检查性能指标

### 5. 调度和分配
- [ ] 验证任务调度
- [ ] 测试任务分配
- [ ] 检查资源利用

## 测试用例
1. 基本功能测试：
```bash
python src/main.py --config-dir configs --dataset test_dataset.json
```

2. 错误处理测试：
```bash
python src/main.py --config-dir invalid_configs --dataset invalid_dataset.json
```

## 预期结果
1. 成功加载配置和数据集
2. 正确处理数据预处理
3. 正确执行模型推理
4. 生成有效的性能报告

## 调试日志
- 记录所有错误和异常
- 记录关键步骤的执行状态
- 记录性能指标 