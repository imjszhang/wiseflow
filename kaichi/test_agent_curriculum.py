import asyncio
import os
from agents.curriculum import CurriculumAgent, CurriculumConfig

async def main():
    # 获取当前文件所在的目录（kaichi目录）
    kaichi_dir = os.path.abspath(os.path.dirname(__file__))   

    # 配置 CurriculumAgent
    config = CurriculumConfig(
        ckpt_dir=os.path.join(kaichi_dir, "work_dir/ckpt"),  # 检查点目录
        observation_dir=os.path.join(kaichi_dir, "../"),  # 观察目录（项目根目录）
        mode="auto",
        max_retries=3,
        log_level="DEBUG",
        cache_size=10
    )

    # 初始化 CurriculumAgent
    print("Initializing CurriculumAgent...")
    agent = CurriculumAgent(config)

    # 测试项目观察数据的提取和保存
    """
    try:
        print("\nExtracting and saving project observation data...")
        agent.extract_and_save_observation()
        print("Project observation data extracted and saved successfully.")
    except Exception as e:
        print(f"Failed to extract and save observation data: {e}")
        return
    """
        
    # 测试任务提议功能
    try:
        print("\nProposing next task...")
        next_task, context = await agent.propose_next_task()
        agent._save_state()
        print(f"Next Task: {next_task}")
        print(f"Task Context:\n{context}")
    except Exception as e:
        print(f"Failed to propose next task: {e}")
        return

    # 测试任务上下文生成
    """
    try:
        print("\nGetting task context for a specific task...")
        task = "用python写一个脚本获取当前项目的文件目录"
        context = await agent.get_task_context(task)
        print(f"Task: {task}")
        print(f"Task Context:\n{context}")
    except Exception as e:
        print(f"Failed to get task context: {e}")
        return
    """

if __name__ == "__main__":
    asyncio.run(main())