import os
import sys
from dotenv import load_dotenv
import uvicorn

# 获取项目根目录的路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 Python 路径中
sys.path.append(root_dir)
env_path = os.path.join(root_dir, '.env')

# 加载环境变量并验证
load_dotenv(dotenv_path=env_path)


# 使用 uvicorn 的 Python API 启动服务器
if __name__ == "__main__":
    uvicorn.run(
        "core.backend:app",
        host="localhost",
        port=8077,
        reload=True,
        reload_dirs=["core"]
    )