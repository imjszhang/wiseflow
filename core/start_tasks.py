import os
import subprocess
from dotenv import load_dotenv

# 加载 .env 文件
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.env')
load_dotenv(dotenv_path=env_path)

# 执行 tasks.py 脚本
subprocess.run(["python", "tasks.py"])