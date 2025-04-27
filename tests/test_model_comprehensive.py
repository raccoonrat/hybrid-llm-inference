import os
import json
import time
import logging
import ctypes
import platform
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler
from tests.benchmarking.test_benchmarking import SystemBenchmarking, ModelBenchmarking
from src.benchmarking.report_generator import ReportGenerator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComprehensiveTest:
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化配置
        self.config = {
            "model_path": model_path,
            "output_dir": output_dir,
            "dataset_path": "tests/data/test_dataset.json",
            "hardware_config": {
                "device_id": 0,
                "idle_power": 20.0,
                "sample_interval": 1
            },
            "model_config": {
                "model_type": "tinyllama",
                "model_name": "TinyLlama-1.1B-Chat-v1.0",
                "max_length": 2048,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "scheduler_config": {
                "strategy": "token_based",
                "num_workers": 4
            }
        }
        
        # 初始化组件
        self.profiler = RTX4050Profiler(config=self.config["hardware_config"])
        self.benchmarker = SystemBenchmarking(config=self.config)
        self.report_generator = ReportGenerator(output_dir=output_dir)
        
        # 初始化 NVML
        self.nvml = None
        self._init_nvml()
    
    def _init_nvml(self) -> bool:
        """初始化 NVML"""
        try:
            logger.info("正在初始化 NVML...")
            
            # 检查 NVML 库路径
            nvml_path = r"C:\Windows\System32\nvml.dll"
            if not os.path.exists(nvml_path):
                logger.error(f"NVML 库不存在: {nvml_path}")
                return False
            
            # 加载 NVML DLL
            self.nvml = ctypes.CDLL(nvml_path)
            
            # 初始化 NVML
            result = self.nvml.nvmlInit_v2()
            if result != 0:
                logger.error(f"NVML 初始化失败，错误代码: {result}")
                return False
            
            logger.info("NVML 初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"NVML 初始化失败: {str(e)}")
            return False
    
    def test_nvml(self) -> Dict[str, Any]:
        """测试 NVML 功能"""
        try:
            if not self.nvml:
                logger.error("NVML 未初始化")
                return {}
            
            logger.info("开始 NVML 测试...")
            
            # 获取设备数量
            device_count = ctypes.c_uint(0)
            result = self.nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))
            if result != 0:
                logger.error(f"获取设备数量失败，错误代码: {result}")
                return {}
            
            logger.info(f"发现 {device_count.value} 个 NVIDIA GPU 设备")
            
            # 获取设备信息
            nvml_results = {}
            for i in range(device_count.value):
                device = ctypes.c_void_p(0)
                result = self.nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(device))
                if result != 0:
                    continue
                
                # 获取设备名称
                name = ctypes.create_string_buffer(96)
                result = self.nvml.nvmlDeviceGetName(device, name, ctypes.c_uint(96))
                if result == 0:
                    device_name = name.value.decode()
                    logger.info(f"设备 {i}: {device_name}")
                    
                    # 获取内存信息
                    memory = ctypes.c_ulonglong * 3
                    memory_info = memory()
                    result = self.nvml.nvmlDeviceGetMemoryInfo(device, ctypes.byref(memory_info))
                    if result == 0:
                        nvml_results[f"device_{i}_memory"] = {
                            "total": memory_info[0] / (1024 * 1024),  # MB
                            "used": memory_info[2] / (1024 * 1024),
                            "free": memory_info[1] / (1024 * 1024)
                        }
                    
                    # 获取功耗信息
                    power = ctypes.c_uint(0)
                    result = self.nvml.nvmlDeviceGetPowerUsage(device, ctypes.byref(power))
                    if result == 0:
                        nvml_results[f"device_{i}_power"] = power.value / 1000.0  # W
                    
                    # 获取利用率信息
                    class nvmlUtilization_t(ctypes.Structure):
                        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
                    
                    utilization = nvmlUtilization_t()
                    result = self.nvml.nvmlDeviceGetUtilizationRates(device, ctypes.byref(utilization))
                    if result == 0:
                        nvml_results[f"device_{i}_utilization"] = {
                            "gpu": utilization.gpu,
                            "memory": utilization.memory
                        }
            
            logger.info("NVML 测试完成")
            return nvml_results
            
        except Exception as e:
            logger.error(f"NVML 测试失败: {str(e)}")
            return {}
    
    def __del__(self):
        """清理 NVML 资源"""
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
                logger.info("NVML 已关闭")
            except Exception as e:
                logger.error(f"关闭 NVML 失败: {str(e)}")
    
    def test_model_load(self) -> bool:
        """测试模型加载功能"""
        try:
            logger.info("开始模型加载测试...")
            
            # 检查模型路径
            if not os.path.exists(self.model_path):
                logger.error(f"模型路径不存在: {self.model_path}")
                return False
            
            # 加载tokenizer
            logger.info("正在加载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 加载模型
            logger.info("正在加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # 基本推理测试
            test_input = "你好，请介绍一下你自己。"
            logger.info(f"测试输入: {test_input}")
            
            inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"模型输出: {response}")
            
            logger.info("模型加载测试完成！")
            return True
            
        except Exception as e:
            logger.error(f"模型加载测试失败: {str(e)}")
            return False
    
    def test_performance(self) -> Dict[str, Any]:
        """测试模型性能"""
        try:
            logger.info("开始性能测试...")
            
            # 加载模型和tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # 测试输入
            test_inputs = [
                "你好，请介绍一下你自己。",
                "请解释一下什么是机器学习。",
                "写一首关于春天的诗。"
            ]
            
            performance_metrics = {
                "energy_consumption": [],
                "runtime": [],
                "throughput": []
            }
            
            for input_text in test_inputs:
                # 开始性能监控
                self.profiler.start_monitoring()
                
                # 执行推理
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
                end_time = time.time()
                
                # 停止性能监控
                metrics = self.profiler.stop_monitoring()
                
                # 计算性能指标
                runtime = end_time - start_time
                num_tokens = len(outputs[0])
                throughput = num_tokens / runtime
                
                performance_metrics["energy_consumption"].append(metrics["energy_consumption"])
                performance_metrics["runtime"].append(runtime)
                performance_metrics["throughput"].append(throughput)
            
            logger.info("性能测试完成！")
            return performance_metrics
            
        except Exception as e:
            logger.error(f"性能测试失败: {str(e)}")
            return {}
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """运行基准测试"""
        try:
            logger.info("开始基准测试...")
            results = self.benchmarker.run_benchmarks()
            
            # 生成报告
            self.report_generator.generate_report(
                data=results,
                output_dir=self.output_dir,
                output_format="markdown"
            )
            
            logger.info("基准测试完成！")
            return results
            
        except Exception as e:
            logger.error(f"基准测试失败: {str(e)}")
            return {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        results = {
            "model_load": False,
            "nvml": {},
            "performance": {},
            "benchmarks": {}
        }
        
        # 运行 NVML 测试
        results["nvml"] = self.test_nvml()
        
        # 运行模型加载测试
        results["model_load"] = self.test_model_load()
        
        if results["model_load"]:
            # 运行性能测试
            results["performance"] = self.test_performance()
            
            # 运行基准测试
            results["benchmarks"] = self.run_benchmarks()
        
        return results

def main():
    # 配置测试参数
    model_path = r"\\wsl.localhost\Ubuntu-24.04\home\mpcblock\models\TinyLlama-1.1B-Chat-v1.0"
    output_dir = "tests/output/comprehensive_test"
    
    # 创建测试实例
    tester = ModelComprehensiveTest(model_path, output_dir)
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 输出测试结果
    logger.info("\n测试结果摘要:")
    logger.info(f"模型加载测试: {'通过' if results['model_load'] else '失败'}")
    
    if results["nvml"]:
        logger.info("\nNVML 测试结果:")
        for device_id, metrics in results["nvml"].items():
            logger.info(f"\n设备 {device_id}:")
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        logger.info(f"  {metric}:")
                        for k, v in value.items():
                            logger.info(f"    {k}: {v}")
                    else:
                        logger.info(f"  {metric}: {value}")
            else:
                logger.info(f"  {metrics}")
    
    if results["performance"]:
        logger.info("\n性能测试结果:")
        for metric, values in results["performance"].items():
            avg_value = sum(values) / len(values)
            logger.info(f"{metric}: 平均值 = {avg_value:.2f}")
    
    if results["benchmarks"]:
        logger.info("\n基准测试报告已生成在: " + output_dir)

if __name__ == "__main__":
    main() 