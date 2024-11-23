import os
import subprocess
from dotenv import load_dotenv

# 加载 .env 文件
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.env')
load_dotenv(dotenv_path=env_path)

# 设置 uvicorn 启动命令
command = [
    "uvicorn",
    "main:app",  # 指定 FastAPI 应用的路径
    "--reload",  # 开启自动重载（开发模式）
    "--host", "localhost",  # 绑定主机
    "--port", "7777"        # 绑定端口
]

# 执行 uvicorn 命令
subprocess.run(command)