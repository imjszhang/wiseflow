import subprocess
import os
from dotenv import load_dotenv
# 加载 .env 文件
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.env')
load_dotenv(dotenv_path=env_path)
# 设置 uvicorn 启动参数
command = [
    "uvicorn",
    "backend:app",
    "--reload",
    "--host",
    "localhost",
    "--port",
    "8077"
]

# 执行 uvicorn 命令
subprocess.run(command)