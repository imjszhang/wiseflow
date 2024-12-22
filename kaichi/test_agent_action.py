import asyncio
import os
from agents.action import ActionAgent, ActionConfig


async def main():
    # 获取当前文件所在的目录（kaichi目录）
    kaichi_dir = os.path.abspath(os.path.dirname(__file__))

    # 配置 ActionAgent
    config = ActionConfig(
        ckpt_dir=os.path.join(kaichi_dir, "work_dir/ckpt"),  # 检查点目录
        observation_dir=os.path.abspath(os.path.join(kaichi_dir, "../")),  # 观察目录（kaichi目录的上级目录）
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

    # 测试加载的 prompts
    print("Loaded prompts:", agent.prompts)

    # 测试生成系统消息
    skills = ["Skill1", "Skill2", "Skill3"]
    system_message = agent.render_system_message(skills)
    print("System message:", system_message)

    # 测试生成 human 消息
    events = [{"event": "test_event"}]
    code = "print('Hello, World!')"
    task = "Write a Python script"
    context = "This is a test context"
    critique = "No critique"
    human_message = agent.render_human_message(events, code, task, context, critique)
    print("Human message:", human_message)


if __name__ == "__main__":
    asyncio.run(main())