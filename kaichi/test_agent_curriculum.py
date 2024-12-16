import asyncio
from agents.curriculum import CurriculumAgent, CurriculumConfig

async def main():
    # 配置 CurriculumAgent
    config = CurriculumConfig(
        ckpt_dir="work_dir/ckpt",
        mode="auto",
        source_name="test_source",
        source_content="This is a test source content",
        project_name="test",
        max_retries=3,
        log_level="DEBUG",
        cache_size=10
    )

    # 初始化 CurriculumAgent
    print("Initializing CurriculumAgent...")
    agent = CurriculumAgent(config)

    task="用python写一个脚本获取当前项目的文件目录"
    try:
        # Get task and context
        if not task:
            # 提议下一个任务
            print("\nProposing next task...")
            next_task, context = await agent.propose_next_task()
            print(f"Next Task: {next_task}")
        else:
            print("\nGetting task context...")
            print(f"Current Task:{task}")
            context = await agent.get_task_context(task)

        print(f"Task Context:\n{context}")
    except Exception as e:
        print(f"Failed to propose next task: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())