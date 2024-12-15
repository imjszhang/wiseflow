from __future__ import annotations
import os
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from llms.dify_wrapper import dify_llm_async
import utils as U
from prompts import load_prompt
from agents.skill import SkillManager, SkillManagerConfig

@dataclass
class ActionConfig:
    """Action Agent Configuration"""
    ckpt_dir: str = "work_dir/ckpt"
    mode: str = "auto"
    source_name: str = "init"
    knowledge_name: str = "default"
    max_retries: int = 5
    log_level: str = "INFO"
    cache_size: int = 100
    temperature: float = 0.8
    request_timeout: int = 120
    execution_error: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError(f"Invalid temperature: {self.temperature}")
        if self.request_timeout <= 0:
            raise ValueError(f"Invalid request timeout: {self.request_timeout}")
        if self.max_retries <= 0:
            raise ValueError(f"Invalid max retries: {self.max_retries}")

class ActionCache:
    """Cache management for Action Agent"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.usage_count = {}
        self.last_used = {}
        
    def add(self, key: str, value: Dict):
        """Add item to cache"""
        if len(self.cache) >= self.max_size:
            self._remove_least_used()
            
        self.cache[key] = value
        self.usage_count[key] = 0
        self.last_used[key] = time.time()
        
    def get(self, key: str) -> Optional[Dict]:
        """Retrieve item from cache"""
        if key in self.cache:
            self.usage_count[key] += 1
            self.last_used[key] = time.time()
            return self.cache[key]
        return None
        
    def _remove_least_used(self):
        """Remove least used item"""
        if not self.usage_count:
            return
            
        min_usage = min(self.usage_count.values())
        to_remove = next(k for k, v in self.usage_count.items() if v == min_usage)
        
        self.cache.pop(to_remove, None)
        self.usage_count.pop(to_remove, None)
        self.last_used.pop(to_remove, None)

    def to_dict(self) -> Dict:
        """Convert cache to dictionary"""
        return {
            "cache": self.cache,
            "usage_count": self.usage_count,
            "last_used": self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ActionCache':
        """Create cache from dictionary"""
        cache = cls()
        cache.cache = data.get("cache", {})
        cache.usage_count = data.get("usage_count", {})
        cache.last_used = data.get("last_used", {})
        return cache

class ActionAgent:
    """Action Agent for code generation and skill management"""
    def __init__(self, config: Optional[ActionConfig] = None):
        self.config = config or ActionConfig()
        self.llm = dify_llm_async
        
        # Setup logging
        self._setup_logging()
        
        # Initialize system
        self._init_system()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Initialize skill manager
        skill_config = SkillManagerConfig(
            ckpt_dir=self.config.ckpt_dir,
            cache_size=self.config.cache_size,
            log_level=self.config.log_level
        )
        self.skill_manager = SkillManager(skill_config)
        
        self.logger.info("ActionAgent initialized successfully")

    def _setup_logging(self):
        """Setup logging system"""
        self.logger = logging.getLogger("ActionAgent")
        self.logger.setLevel(self.config.log_level)
        
        log_dir = f"{self.config.ckpt_dir}/action/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        handler = logging.FileHandler(f"{log_dir}/action_agent.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates"""
        return {
            'system': load_prompt("action/system"),
            'task': load_prompt("action/task")
        }

    def _init_system(self):
        """Initialize system"""
        try:
            self._init_directories()
            self.cache = ActionCache(self.config.cache_size)
            self._load_state()
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise RuntimeError("Failed to initialize ActionAgent") from e

    def _init_directories(self):
        """Initialize directory structure"""
        base_dir = f"{self.config.ckpt_dir}/action/{self.config.knowledge_name}/{self.config.source_name}"
        dirs = [
            base_dir,
            f"{base_dir}/cache"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.debug(f"Directory created/verified: {dir_path}")

    def _load_state(self):
        """Load saved state"""
        try:
            base_path = self._get_base_path()
            
            # Load cache states
            cache_data = U.load_json(f"{base_path}/cache/action_cache.json")
            self.cache = ActionCache.from_dict(cache_data)
            
            self.logger.info(f"Loaded {len(self.cache.cache)} cache entries")
            
        except Exception as e:
            self.logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save current state"""
        try:
            base_path = self._get_base_path()
            
            # Save cache
            U.dump_json(
                self.cache.to_dict(),
                f"{base_path}/cache/action_cache.json"
            )
            
            self.logger.debug("State saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise

    def _get_base_path(self) -> str:
        """Get base path for state files"""
        return f"{self.config.ckpt_dir}/action/{self.config.knowledge_name}/{self.config.source_name}"

    async def update_skill_knowledge(self, info: Dict):
        """Update skill knowledge base using SkillManager"""
        try:
            # Validate input
            required = ["program_name", "program_code"]
            if missing := [f for f in required if f not in info]:
                raise ValueError(f"Missing required fields: {missing}")
            
            # Use SkillManager to add/update skill
            await self.skill_manager.add_skill(
                skill_name=info["program_name"],
                skill_code=info["program_code"]
            )
            
            self.logger.info(f"Successfully updated skill: {info['program_name']}")
            
        except Exception as e:
            self.logger.error(f"Failed to update skill: {e}")
            raise

    def get_skill(self, name: str) -> Optional[Dict]:
        """Get skill by name"""
        return self.skill_manager.get_skill(name)

    def list_skills(self) -> List[str]:
        """List all skills"""
        return self.skill_manager.list_skills()

    async def search_skills(self, query: str) -> List[str]:
        """Search skills by query"""
        try:
            return await self.skill_manager.retrieve_skills(query)
        except Exception as e:
            self.logger.error(f"Failed to search skills: {e}")
            raise

    async def analyze_code(self, code: str) -> Dict:
        """Analyze code using SkillManager"""
        try:
            return await self.skill_manager.analyze_skill(code)
        except Exception as e:
            self.logger.error(f"Failed to analyze code: {e}")
            raise

    async def execute_skill(self, query: str, *args, **kwargs) -> Optional[Dict]:
        """
        Find and execute a skill from the skill library based on the query.

        Args:
            query (str): The query to search for a relevant skill.
            *args: Positional arguments to pass to the skill function.
            **kwargs: Keyword arguments to pass to the skill function.

        Returns:
            Optional[Dict]: The result of the skill execution, or None if no skill is found.
        """
        try:
            # Step 1: Search for relevant skills
            self.logger.info(f"Searching for skills related to query: {query}")
            skills = await self.skill_manager.retrieve_skills(query)

            if not skills:
                self.logger.warning(f"No skills found for query: {query}")
                return None

            # Step 2: Select the first relevant skill (you can implement more complex selection logic if needed)
            skill_code = skills[0]
            self.logger.info(f"Executing skill: {skill_code}")

            # Step 3: Dynamically execute the skill code
            exec_globals = {}
            exec_locals = {}
            exec(skill_code, exec_globals, exec_locals)

            # Assuming the skill defines a function named `main` as the entry point
            if "main" in exec_locals:
                result = await exec_locals["main"](*args, **kwargs)
                self.logger.info(f"Skill executed successfully: {result}")
                return {"result": result}
            else:
                self.logger.error("No 'main' function found in the skill code")
                return None

        except Exception as e:
            self.logger.error(f"Failed to execute skill: {e}")
            raise