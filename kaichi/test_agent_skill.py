from agents import SkillManager, SkillManagerConfig

async def main():
    # 创建配置
    config = SkillManagerConfig(
        retrieval_top_k=10,
        cache_size=200,
        log_level="DEBUG"
    )
    
    # 初始化SkillManager
    skill_manager = SkillManager(config)
    
    # 添加技能
    await skill_manager.add_skill(
        "test_skill",
        """async function test_skill(bot) {
            // Test skill implementation
            console.log('Testing skill');
        }"""
    )
    
    # 检索技能
    skills = await skill_manager.retrieve_skills("test")
    print("Retrieved skills:", skills)
    
    # 获取技能信息
    skill_info = skill_manager.get_skill("test_skill")
    print("Skill info:", skill_info)
    
    # 分析技能
    analysis = await skill_manager.analyze_skill(skill_info["code"])
    print("Skill analysis:", analysis)
    
    # 列出所有技能
    all_skills = skill_manager.list_skills()
    print("All skills:", all_skills)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())