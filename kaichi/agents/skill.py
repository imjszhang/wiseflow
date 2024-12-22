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
        
        log_dir = f"{self.config.ckpt_dir}/skill/logs"  
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
            f"{self.config.ckpt_dir}/skill/code",  
            f"{self.config.ckpt_dir}/skill/description"  
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.debug(f"Directory created/verified: {dir_path}")

    def _load_skills(self):
        """加载技能数据"""
        if self.config.resume:
            self.logger.info(f"Loading skills from {self.config.ckpt_dir}/skill")  
            try:
                self.skills = U.load_json(f"{self.config.ckpt_dir}/skill/skills.json")  
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

    async def generate_skill_description(self, program_name: str, program_code: str):
        """
        生成技能描述，返回 JSON Schema 格式的描述。
        
        Args:
            program_name (str): 函数名称。
            program_code (str): 函数代码。
        
        Returns:
            Dict: JSON Schema 格式的函数描述。
        """
        self.logger.info(f"Generating JSON Schema description for skill: {program_name}")
        
        try:
            # 构建提示词
            prompt = self.prompts['description'].replace(
                "{{code}}", program_code
            ).replace(
                "{{function_name}}", program_name
            )
            
            inputs = {'system': prompt}
            
            # 调用 LLM 生成描述
            response = await self.llm(
                query="Please generate a JSON Schema description for the provided Python function.",
                user="SkillManager",
                inputs=inputs,
                logger=self.logger
            )
            
            if "error" in response:
                raise ValueError(f"Error generating JSON Schema description: {response['error']}")
            
            # 解析 LLM 的返回结果
            answer= U.extract_json_from_markdown(response['answer'])
            skill_description = answer
            
            # 验证返回的 JSON Schema 格式
            try:
                skill_schema = U.fix_and_parse_json(skill_description)
            except Exception as e:
                self.logger.error(f"Invalid JSON Schema format: {e}")
                raise ValueError("Generated description is not a valid JSON Schema")
            
            self.logger.debug(f"Generated JSON Schema for {program_name}: {skill_schema}")
            return U.json_dumps(skill_schema, indent=4)
        
        except Exception as e:
            self.logger.error(f"Failed to generate JSON Schema description: {e}")
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

    async def add_new_skill(self, info: Dict) -> Dict:
        """添加新技能"""
        self.logger.info(f"Adding new skill: {info['program_name']}")
        
        try:
            program_name = info["program_name"]
            program_code = info["program_code"]
            
            # 生成技能描述
            skill_description = await self.generate_skill_description(program_name, program_code)
            self.logger.info(f"Generated description for {program_name}:\n{skill_description}")
            
            # 检查技能字典文件是否存在
            skills_file_path = f"{self.config.ckpt_dir}/skill/skills.json"
            if os.path.exists(skills_file_path):
                self.skills = U.load_json(skills_file_path)
            else:
                self.logger.warning(f"Skills file not found. Creating a new one at {skills_file_path}")
                self.skills = {}
                U.dump_json(self.skills, skills_file_path)
            
            # 检查技能是否已存在
            if program_name in self.skills:
                self.logger.warning(f"Skill {program_name} already exists. Rewriting!")
                
                # 删除旧技能的向量数据库记录
                await self.dataset_api.delete_document(self.dataset_id, program_name)
                
                # 处理版本号
                i = 2
                while f"{program_name}V{i}.py" in os.listdir(f"{self.config.ckpt_dir}/skill/code"):
                    i += 1
                dumped_program_name = f"{program_name}V{i}"
            else:
                dumped_program_name = program_name
            
            # 添加技能到向量数据库
            response = await self.dataset_api.create_document_by_text(
                dataset_id=self.dataset_id,
                name=program_name,
                text=skill_description
            )
            if "error" in response:
                raise ValueError(f"Error adding skill to dataset: {response['error']}")
            
            # 更新技能字典
            self.skills[program_name] = {
                "code": program_code,
                "description": skill_description,
            }
            
            # 确保向量数据库与技能字典同步
            documents = await self.dataset_api.list_documents(self.dataset_id)
            assert len(documents.get("data", [])) == len(self.skills), "Dataset is not synced with skills.json"
            
            # 保存技能代码和描述到本地文件
            U.dump_text(
                program_code, f"{self.config.ckpt_dir}/skill/code/{dumped_program_name}.py"
            )
            U.dump_text(
                skill_description,
                f"{self.config.ckpt_dir}/skill/description/{dumped_program_name}.txt",
            )
            U.dump_json(self.skills, skills_file_path)
            
            self.logger.info(f"Successfully added skill: {program_name}")
            
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