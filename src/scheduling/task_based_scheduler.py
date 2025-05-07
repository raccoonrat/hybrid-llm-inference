from typing import List, Dict, Any, Callable
import numpy as np
import torch
import math
from .task_allocator import TaskAllocator
from .base_scheduler import BaseScheduler

class TaskBasedScheduler(BaseScheduler):
    """任务调度器，负责管理和分配任务"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化调度器"""
        super().__init__(config)
        if config is None:
            config = {
                "hardware_config": {
                    "apple_m1_pro": {
                        "device_type": "m1_pro",
                        "idle_power": 10.0,
                        "sample_interval": 200
                    },
                    "nvidia_rtx4050": {
                        "device_type": "rtx4050",
                        "idle_power": 15.0,
                        "sample_interval": 200
                    }
                },
                "model_config": {
                    "models": {
                        "tinyllama": {
                            "model_name": "tinyllama",
                            "model_path": "path/to/model",
                            "mode": "local",
                            "batch_size": 1,
                            "max_length": 128
                        }
                    }
                }
            }
        self.allocator = TaskAllocator(
            hardware_config=config["hardware_config"],
            model_config=config["model_config"]
        )
        self.tasks = []
        self.device_queues = {
            "apple_m1_pro": [],
            "nvidia_rtx4050": []
        }
        self.gpu_cache = {}  # 缓存GPU计算结果
        self.cpu_cache = {}  # 缓存CPU计算结果
        self.is_warmed_up = False
        self.device_affinity = {}  # 设备亲和性记录
        self.affinity_threshold = 0.7  # 亲和性阈值
        self.batch_size = 4  # 批处理大小
        self.current_batch = {"apple_m1_pro": [], "nvidia_rtx4050": []}  # 当前批次任务
        self.batch_results = {}  # 批次结果缓存
        
    def _init_scheduler(self) -> None:
        """初始化调度器的具体实现"""
        # 初始化设备队列
        for device in self.device_queues:
            self.device_queues[device] = []
        
        # 初始化缓存
        self.gpu_cache = {}
        self.cpu_cache = {}
        
        # 初始化设备亲和性
        self.device_affinity = {}
        
        # 初始化批次
        self.current_batch = {"apple_m1_pro": [], "nvidia_rtx4050": []}
        self.batch_results = {}
        
        # 预热设备
        self.warmup()
        
    def add_task(self, task: Dict[str, Any]) -> None:
        """添加新任务到队列"""
        self.tasks.append(task)
        
    def schedule(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """调度所有任务

        Args:
            tasks: 待调度的任务列表

        Returns:
            调度后的任务列表
        """
        scheduled_tasks = []
        
        for task in tasks:
            # 兼容新结构：所有字段从 task['query'] 获取
            query = task.get("query", {})
            allocate_task = {
                "query": {
                    "input_tokens": query.get("input_tokens", 0),
                    "output_tokens": query.get("output_tokens", 0),
                    "prompt": query.get("prompt", "")
                },
                "model": task.get("model", "tinyllama")
            }
            allocations = self.allocator.allocate([allocate_task], model_name=task.get("model", "tinyllama"))
            
            if allocations:
                allocation = allocations[0]
                # 将任务添加到对应设备队列
                self.device_queues[allocation["hardware"]].append(task)
                scheduled_tasks.append({
                    "task": task,
                    "device": allocation["hardware"]
                })
        
        return scheduled_tasks
    
    def get_queue_lengths(self) -> Dict[str, int]:
        """获取各设备队列长度"""
        return {
            device: len(queue)
            for device, queue in self.device_queues.items()
        }
    
    def clear_queues(self) -> None:
        """清空所有队列"""
        self.tasks = []
        for device in self.device_queues:
            self.device_queues[device] = []
    
    def warmup(self):
        """预热GPU和CPU"""
        if not self.is_warmed_up:
            # GPU预热
            if torch.cuda.is_available():
                x = torch.randn(32, 128, 768, device="cuda")
                for _ in range(3):
                    x = torch.matmul(x, torch.randn(768, 768, device="cuda"))
                torch.cuda.synchronize()
            
            # CPU预热
            x = np.random.rand(8, 64, 256)
            for _ in range(2):
                x = np.tanh(np.dot(x, np.random.rand(256, 256)))
            
            self.is_warmed_up = True
    
    def get_cache_key(self, task_fn: Callable) -> str:
        """生成任务缓存键"""
        return f"{task_fn.__name__}"
    
    def update_device_affinity(self, task_key: str, device: str) -> None:
        """更新设备亲和性"""
        if task_key not in self.device_affinity:
            self.device_affinity[task_key] = {"apple_m1_pro": 0, "nvidia_rtx4050": 0}
        
        self.device_affinity[task_key][device] += 1
        
        # 归一化亲和性
        total = sum(self.device_affinity[task_key].values())
        for d in self.device_affinity[task_key]:
            self.device_affinity[task_key][d] /= total
    
    def get_preferred_device(self, task_key: str) -> str:
        """获取任务的首选设备"""
        if task_key not in self.device_affinity:
            return "apple_m1_pro"  # 默认使用apple_m1_pro
        
        affinity = self.device_affinity[task_key]
        if affinity["apple_m1_pro"] >= self.affinity_threshold:
            return "apple_m1_pro"
        elif affinity["nvidia_rtx4050"] >= self.affinity_threshold:
            return "nvidia_rtx4050"
        else:
            return "apple_m1_pro"  # 亲和性不明显时使用apple_m1_pro
    
    def execute_batch(self, device: str) -> None:
        """执行当前批次的任务"""
        if not self.current_batch[device]:
            return
        
        # 执行设备特定的计算任务
        if device == "apple_m1_pro" and torch.cuda.is_available():
            # GPU任务：矩阵运算 + 注意力计算
            batch_size = 32
            seq_len = 128
            hidden_size = 768
            
            # 模拟输入嵌入
            x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
            
            # 模拟注意力计算
            for _ in range(3):  # 3层注意力
                q = torch.matmul(x, torch.randn(hidden_size, hidden_size, device="cuda"))
                k = torch.matmul(x, torch.randn(hidden_size, hidden_size, device="cuda"))
                v = torch.matmul(x, torch.randn(hidden_size, hidden_size, device="cuda"))
                
                # 计算注意力分数
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hidden_size)
                attn = torch.softmax(scores, dim=-1)
                x = torch.matmul(attn, v)
            
            torch.cuda.synchronize()
        else:
            # CPU任务：轻量级文本处理
            batch_size = 8
            seq_len = 64
            hidden_size = 256
            
            # 模拟轻量级处理
            x = np.random.rand(batch_size, seq_len, hidden_size)
            for _ in range(2):  # 2层处理
                x = np.tanh(np.dot(x, np.random.rand(hidden_size, hidden_size)))
        
        # 执行批次中的任务
        for task_fn in self.current_batch[device]:
            result = task_fn()
            task_key = self.get_cache_key(task_fn)
            self.batch_results[task_key] = result
        
        # 清空当前批次
        self.current_batch[device] = []
    
    def schedule_task(self, task_fn: Callable, device: str) -> Any:
        """调度并执行任务
        
        Args:
            task_fn: 任务函数，返回任务结果
            device: 执行设备 ("apple_m1_pro" 或 "nvidia_rtx4050")
            
        Returns:
            Any: 任务执行结果
        """
        # 预热设备
        self.warmup()
        
        # 获取任务键
        task_key = self.get_cache_key(task_fn)
        
        # 检查缓存
        if task_key in self.batch_results:
            return self.batch_results[task_key]
        
        # 检查设备亲和性
        preferred_device = self.get_preferred_device(task_key)
        if preferred_device != device:
            # 如果当前设备不是首选设备，考虑切换
            if preferred_device == "apple_m1_pro" and torch.cuda.is_available():
                device = "apple_m1_pro"
            elif preferred_device == "nvidia_rtx4050":
                device = "nvidia_rtx4050"
        
        # 将任务添加到当前批次
        self.current_batch[device].append(task_fn)
        
        # 如果批次已满，执行批次
        if len(self.current_batch[device]) >= self.batch_size:
            self.execute_batch(device)
        
        # 更新设备亲和性
        self.update_device_affinity(task_key, device)
        
        # 返回结果（如果已执行）
        if task_key in self.batch_results:
            return self.batch_results[task_key]
        
        # 如果批次未满且结果未缓存，执行当前批次
        self.execute_batch(device)
        
        return self.batch_results[task_key] 