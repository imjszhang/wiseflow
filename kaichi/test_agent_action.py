import asyncio
import os
from agents.action import ActionAgent, ActionConfig


async def main():
    # 获取当前文件所在的目录（kaichi目录）
    kaichi_dir = os.path.abspath(os.path.dirname(__file__))

    # 配置 ActionAgent
    config = ActionConfig(
        ckpt_dir=os.path.join(kaichi_dir, "work_dir/ckpt"),  # 检查点目录
        observation_dir=os.path.join(kaichi_dir, "../"),  # 观察目录（项目根目录）
        resume=False,  
        mode="auto",
        max_retries=3,
        log_level="DEBUG",
        cache_size=10,
        temperature=0.7,
        request_timeout=60
    )

    # 初始化 ActionAgent
    print("Initializing ActionAgent...")
    agent = ActionAgent(config)


if __name__ == "__main__":
    asyncio.run(main())