import asyncio
import os
from kaichi import Kaichi, AgentConfig

async def main():
    # 获取当前文件所在的目录（kaichi目录）
    kaichi_dir = os.path.abspath(os.path.dirname(__file__))
    config = AgentConfig(
        max_iterations=160,
        max_retries=5,
        env_timeout=5,  # ProjectEnv 的超时时间
        log_path=os.path.join(kaichi_dir, "work_dir/logs"),  # 日志目录
        ckpt_dir=os.path.join(kaichi_dir, "work_dir/ckpt"),  # 检查点目录
        observation_dir=os.path.abspath(os.path.join(kaichi_dir, "../")),  # 观察目录（kaichi目录的上级目录）
        resume=False,
        log_level="INFO"
    )
    
    agent = Kaichi(config)
    results = await agent.learn()
    print(f"Run completed with results: {results}")

if __name__ == "__main__":
    asyncio.run(main())