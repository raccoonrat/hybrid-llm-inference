import json
import random
from typing import List, Dict, Any
from pathlib import Path

class AlpacaLoader:
    """Alpaca数据集加载器"""
    
    def __init__(self, dataset_path: str):
        """初始化数据加载器
        
        Args:
            dataset_path: 数据集文件路径
        """
        self.dataset_path = Path(dataset_path)
        self._load_dataset()
        
    def _load_dataset(self) -> None:
        """加载数据集"""
        try:
            if not self.dataset_path.exists():
                # 如果文件不存在，创建示例数据
                self._create_sample_data()
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"加载数据集失败: {e}")
            self.data = []
            
    def _create_sample_data(self) -> None:
        """创建示例数据用于测试"""
        sample_data = [
            {
                "instruction": "解释什么是机器学习",
                "input": "",
                "output": "机器学习是人工智能的一个子领域..."
            },
            {
                "instruction": "分析以下文本",
                "input": "这是一个很长的输入文本，用于测试任务分配...",
                "output": "文本分析结果..."
            },
            {
                "instruction": "总结这段话",
                "input": "人工智能正在改变我们的生活方式...",
                "output": "总结：AI技术发展影响深远。"
            }
        ]
        
        # 生成更多样本
        templates = [
            "解释{topic}的概念",
            "分析{topic}的优缺点",
            "比较{topic}和{other_topic}的区别"
        ]
        
        topics = [
            "深度学习", "神经网络", "强化学习",
            "自然语言处理", "计算机视觉", "知识图谱"
        ]
        
        for _ in range(20):
            template = random.choice(templates)
            topic = random.choice(topics)
            other_topic = random.choice([t for t in topics if t != topic])
            
            instruction = template.format(topic=topic, other_topic=other_topic)
            input_text = f"请详细说明{topic}的应用场景..." if random.random() > 0.5 else ""
            
            sample_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": f"{topic}的详细解释..."
            })
            
        # 保存示例数据
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
    def load_samples(self, n: int = 10) -> List[Dict[str, Any]]:
        """加载指定数量的样本
        
        Args:
            n: 需要的样本数量
            
        Returns:
            包含n个样本的列表
        """
        if not self.data:
            return []
            
        return random.sample(self.data, min(n, len(self.data)))
        
    def get_total_samples(self) -> int:
        """获取数据集总样本数"""
        return len(self.data) 