import asyncio
import os
from agents.action import ActionAgent, ActionConfig

async def main():
    # 获取当前文件所在的目录（kaichi目录）
    kaichi_dir = os.path.abspath(os.path.dirname(__file__))

    # 配置 ActionAgent
    config = ActionConfig(
        ckpt_dir=os.path.join(kaichi_dir, "work_dir/ckpt"),  # 检查点目录
        observation_dir=os.path.join(kaichi_dir, "../core"),        # 观察目录（项目根目录下的 core）
        resume=False,  
        mode="auto",
        project_name="test",
        max_retries=3,
        log_level="DEBUG",
        cache_size=10,
        temperature=0.7,
        request_timeout=60
    )

    # 初始化 ActionAgent
    print("Initializing ActionAgent...")
    agent = ActionAgent(config)

    try:
        # 更新技能知识库
        print("\nUpdating skill knowledge...")
        skill_info = {
            "program_name": "list_project_files",
            "program_code": """
import os            
def list_project_files():
    return [f for f in os.listdir('.') if os.path.isfile(f)]
"""
        }
        await agent.update_skill_knowledge(skill_info)

        # 列出所有技能
        print("\nListing all skills...")
        skills = agent.list_skills()
        print(f"Available Skills: {skills}")

        # 执行技能
        print("\nExecuting skill...")
        skill_name = "list_project_files"
        skill_code = """
import os            
def list_project_files():
    return [f for f in os.listdir('.') if os.path.isfile(f)]
"""
        result = await agent.execute_skill(skill_name, skill_code)
        print(f"Execution Result: {result}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())