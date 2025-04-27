"""
模型推理类。
"""
class ModelInference:
    """模型推理类，用于执行模型推理任务。"""
    
    def __init__(self, config, model, tokenizer):
        """初始化模型推理器。
        
        Args:
            config (dict): 模型配置，包含以下字段：
                - model_name: 模型名称
                - model_path: 模型路径
                - device: 设备类型
                - batch_size: 批次大小
                - max_length: 最大长度
            model: 模型对象
            tokenizer: 分词器对象
            
        Raises:
            ValueError: 当配置参数无效时抛出
        """
        # 验证配置参数
        if not isinstance(config.get("model_name"), str):
            raise ValueError("model_name 必须是字符串类型")
            
        if not isinstance(config.get("model_path"), str):
            raise ValueError("model_path 必须是字符串类型")
            
        if config.get("device") not in ["cpu", "cuda", "mps"]:
            raise ValueError("device 必须是 'cpu'、'cuda' 或 'mps'")
            
        if not isinstance(config.get("batch_size"), int) or config["batch_size"] <= 0:
            raise ValueError("batch_size 必须是正整数")
            
        if not isinstance(config.get("max_length"), int) or config["max_length"] <= 0:
            raise ValueError("max_length 必须是正整数")
        
        self.model_name = config["model_name"]
        self.model_path = config["model_path"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.max_length = config["max_length"]
        self.model = model
        self.tokenizer = tokenizer
        
        # 将模型移动到指定设备
        if self.device == "cuda":
            self.model.cuda()
        elif self.device == "mps":
            self.model.mps()
        else:
            self.model.cpu()
            
        self.is_initialized = True
    
    def infer(self, input_text):
        """执行推理任务。
        
        Args:
            input_text (str): 输入文本
            
        Returns:
            dict: 推理结果，包含以下字段：
                - generated_text: 生成的文本
                - input_tokens: 输入token数量
                - output_tokens: 输出token数量
                - runtime: 运行时间
        """
        if not self.is_initialized:
            raise RuntimeError("ModelInference 未初始化")
            
        if not isinstance(input_text, str):
            raise ValueError("输入文本必须是字符串")
            
        if not input_text:
            raise ValueError("输入文本不能为空")
            
        # 记录开始时间
        import time
        start_time = time.time()
        
        # 编码输入文本
        input_ids = self.tokenizer.encode(input_text)
        
        # 执行推理
        output = self.model.generate(input_ids)
        
        # 解码输出
        generated_text = self.tokenizer.decode(output["output_ids"])
        
        # 计算运行时间
        runtime = time.time() - start_time
        
        return {
            "generated_text": generated_text,
            "input_tokens": len(input_ids),
            "output_tokens": len(output["output_ids"]),
            "runtime": runtime
        }
    
    def cleanup(self):
        """清理资源。"""
        # 将模型移动到 CPU
        if self.device != "cpu":
            self.model.cpu()
        self.is_initialized = False 