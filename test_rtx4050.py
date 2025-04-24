import logging
import ctypes
import sys
logging.basicConfig(level=logging.INFO)

# 检查管理员权限
is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
if not is_admin:
    print("警告：当前没有管理员权限，某些功能可能无法使用")
    print("请以管理员权限重新运行程序")
    sys.exit(1)

from src.hardware_profiling.rtx4050_profiler import RTX4050Profiler

def main():
    try:
        # 测试所有可用的NVIDIA GPU
        for device_id in range(2):  # 假设最多有2个NVIDIA GPU
            try:
                print(f"\n测试 GPU {device_id}:")
                profiler = RTX4050Profiler({'device_id': device_id})
                print('Power:', profiler.measure_power())
                print('Memory:', profiler.get_memory_info())
            except Exception as e:
                print(f'GPU {device_id} 测试失败:', str(e))
                continue
    except Exception as e:
        print('Error:', str(e))
        import traceback
        print('Traceback:', traceback.format_exc())

if __name__ == '__main__':
    main() 