import os
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# 设置测试模式
os.environ['TEST_MODE'] = '1' 