import os
import time
import logging
from dotenv import load_dotenv
from typing import Dict, List, Optional
from dataclasses import dataclass
import utils as U
from llms.dify_wrapper import dify_llm_async
from vectordb.dify_datasets import AsyncDifyDatasetsAPI

@dataclass
class SkillManagerConfig:
    """技能管理器配置"""
    retrieval_top_k: int = 5
    ckpt_dir: str = "work_dir/ckpt"
    resume: bool = False
    dataset_name: str = "skill_dataset"
    cache_size: int = 100
    log_level: str = "INFO"

class SkillCache:
    """技能缓存管理"""
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.skills = {}
        self.usage_count = {}
        self.last_used = {}
        
    def add(self, name: str, skill_data: Dict):
        """添加技能到缓存"""
        if len(self.skills) >= self.max_size:
            min_usage = min(self.usage_count.values())
            to_remove = [k for k,v in self.usage_count.items() if v == min_usage][0]
            self._remove(to_remove)
            
        self.skills[name] = skill_data
        self.usage_count[name] = 0
        self.last_used[name] = time.time()
        
    def get(self, name: str) -> Optional[Dict]:
        """获取缓存的技能"""
        if name in self.skills:
            self.usage_count[name] += 1
            self.last_used[name] = time.time()
            return self.skills[name]
        return None
        
    def _remove(self, name: str):
        """从缓存中移除技能"""
        self.skills.pop(name, None)
        self.usage_count.pop(name, None)
        self.last_used.pop(name, None)

class SkillManager:
    """技能管理器"""
    def __init__(self, config: Optional[SkillManagerConfig] = None):
        # 加载环境变量
        load_dotenv(override=True)
        
        # 初始化配置
        self.config = config or SkillManagerConfig()
        self.llm = dify_llm_async
        self.dataset_api = AsyncDifyDatasetsAPI()
        self.dataset_id = os.getenv('DIFY_DATASETS_ID')
        
        # 加载提示词
        self.prompts = self._load_prompts()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化系统
        self._init_system()
        
        self.logger.info("SkillManager initialized successfully")

    def _load_prompts(self) -> Dict[str, str]:
        """加载所有提示词"""
        from prompts import load_prompt
        
        return {
            'description': load_prompt("skill/skill_description"),
            'review': load_prompt("skill/skill_review"),
            'analysis': load_prompt("skill/skill_analysis"),
            'integration': load_prompt("skill/skill_integration")
        }

    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger("SkillManager")
        self.logger.setLevel(self.config.log_level)
        
        log_dir = f"{self.config.ckpt_dir}/skill/logs"  # 修改此行
        os.makedirs(log_dir, exist_ok=True)
        
        handler = logging.FileHandler(f"{log_dir}/skill_manager.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _init_system(self):
        """初始化系统"""
        try:
            self._init_directories()
            self.skill_cache = SkillCache(self.config.cache_size)
            self._load_skills()
            self._initialize_dataset()
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise RuntimeError("Failed to initialize SkillManager") from e

    def _init_directories(self):
        """初始化目录结构"""
        dirs = [
            f"{self.config.ckpt_dir}/skill/code",  # 修改此行
            f"{self.config.ckpt_dir}/skill/description"  # 修改此行
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.debug(f"Directory created/verified: {dir_path}")

    def _load_skills(self):
        """加载技能数据"""
        if self.config.resume:
            self.logger.info(f"Loading skills from {self.config.ckpt_dir}/skill")  # 修改此行
            try:
                self.skills = U.load_json(f"{self.config.ckpt_dir}/skill/skills.json")  # 修改此行
                for name, data in self.skills.items():
                    self.skill_cache.add(name, data)
            except Exception as e:
                self.logger.warning(f"Failed to load skills: {e}")
                self.skills = {}
        else:
            self.skills = {}

    def _initialize_dataset(self):
        """初始化数据集"""
        try:
            if not self.dataset_id:
                datasets = self.dataset_api.list_datasets()
                for dataset in datasets.get("data", []):
                    if dataset["name"] == self.config.dataset_name:
                        self.dataset_id = dataset["id"]
                        break
                        
                if not self.dataset_id:
                    response = self.dataset_api.create_dataset(self.config.dataset_name)
                    self.dataset_id = response.get("id")
                    
            if not self.dataset_id:
                raise ValueError("Failed to create or retrieve dataset ID")
                
            self.logger.info(f"Dataset initialized with ID: {self.dataset_id}")
            
        except Exception as e:
            self.logger.error(f"Dataset initialization failed: {e}")
            raise

    async def generate_skill_description(self, program_name: str, program_code: str) -> str:
        """生成技能描述"""
        self.logger.info(f"Generating description for skill: {program_name}")
        
        try:
            prompt = self.prompts['description'].replace(
                "{{code}}", program_code
            ).replace(
                "{{function_name}}", program_name
            )
            
            inputs = {'system': prompt}
            response = await self.llm(
                query="Please generate a skill description based on the provided code.",
                user="SkillManager",
                inputs=inputs,
                logger=self.logger
            )
            
            if "error" in response:
                raise ValueError(f"Error generating description: {response['error']}")
                
            skill_description = f"    // {response['answer']}"
            result = f"async function {program_name}(bot) {{\n{skill_description}\n}}"
            
            self.logger.debug(f"Generated description for {program_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate skill description: {e}")
            raise

    async def review_skill(self, skill_name: str, skill_code: str) -> Dict:
        """审查技能"""
        try:
            prompt = self.prompts['review'].replace(
                "{{skill_name}}", skill_name
            ).replace(
                "{{skill_code}}", skill_code
            )
            
            inputs = {'system': prompt}
            response = await self.llm(
                query="Please review this skill and provide detailed feedback.",
                user="SkillManager",
                inputs=inputs,
                logger=self.logger
            )
            
            return {
                'review': response['answer'],
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to review skill: {e}")
            raise

    async def analyze_skill(self, skill_content: str) -> Dict:
        """分析技能"""
        try:
            prompt = self.prompts['analysis'].replace(
                "{{skill_content}}", skill_content
            )
            
            inputs = {'system': prompt}
            response = await self.llm(
                query="Please analyze this skill and provide detailed insights.",
                user="SkillManager",
                inputs=inputs,
                logger=self.logger
            )
            
            return {
                'analysis': response['answer'],
                'timestamp': time.time()
            }
        
        except Exception as e:
            self.logger.error(f"Failed to analyze skill: {e}")
            raise

    async def retrieve_skills(self, query: str) -> List[str]:
        """检索技能"""
        self.logger.info(f"Retrieving skills for query: {query}")
        
        try:
            if not self.dataset_id:
                raise ValueError("Dataset ID not initialized")
                
            documents = await self.dataset_api.list_documents(self.dataset_id)
            if not documents.get("data"):
                return []
                
            retrieved_skills = []
            matched_names = []
            
            for document in documents["data"]:
                if len(retrieved_skills) >= self.config.retrieval_top_k:
                    break
                    
                if query.lower() in document["name"].lower():
                    cached_skill = self.skill_cache.get(document["name"])
                    if cached_skill:
                        retrieved_skills.append(cached_skill["code"])
                    else:
                        skill_code = self.skills.get(document["name"], {}).get("code", "")
                        retrieved_skills.append(skill_code)
                        
                    matched_names.append(document["name"])
                    
            self.logger.info(f"Retrieved {len(retrieved_skills)} skills: {', '.join(matched_names)}")
            return retrieved_skills
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve skills: {e}")
            raise

    async def add_skill(self, skill_name: str, skill_code: str):
        """添加新技能"""
        self.logger.info(f"Adding new skill: {skill_name}")
        
        try:
            skill_description = await self.generate_skill_description(skill_name, skill_code)
            
            response = await self.dataset_api.create_document_by_text(
                dataset_id=self.dataset_id,
                name=skill_name,
                text=skill_code
            )
            
            if "error" in response:
                raise ValueError(f"Error adding skill to dataset: {response['error']}")
                
            skill_data = {
                "code": skill_code,
                "description": skill_description
            }
            self.skills[skill_name] = skill_data
            
            self.skill_cache.add(skill_name, skill_data)
            
            U.dump_json(self.skills, f"{self.config.ckpt_dir}/skill/skills.json")  # 修改此行
            
            self.logger.info(f"Successfully added skill: {skill_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add skill: {e}")
            raise

    def get_skill(self, skill_name: str) -> Optional[Dict]:
        """获取技能信息"""
        cached_skill = self.skill_cache.get(skill_name)
        if cached_skill:
            return cached_skill
        return self.skills.get(skill_name)

    def list_skills(self) -> List[str]:
        """列出所有技能"""
        return list(self.skills.keys())
    


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