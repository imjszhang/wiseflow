from __future__ import annotations
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from llms.dify_wrapper import dify_llm_async
import utils as U
from prompts import load_prompt
from utils.json_utils import fix_and_parse_json

@dataclass
class CriticConfig:
    """Critic Agent Configuration"""
    ckpt_dir: str = "work_dir/ckpt"
    mode: str = "auto"  # auto or manual
    source_name: str = "init"
    knowledge_name: str = "default"
    max_retries: int = 5
    log_level: str = "INFO"
    cache_size: int = 100
    temperature: float = 0

    def __post_init__(self):
        """Validate configuration"""
        if self.mode not in ["auto", "manual"]:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.max_retries <= 0:
            raise ValueError(f"Invalid max retries: {self.max_retries}")

class CriticCache:
    """Cache management for Critic results"""
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

class CriticAgent:
    """Critic Agent for code and task evaluation"""
    def __init__(self, config: Optional[CriticConfig] = None):
        self.config = config or CriticConfig()
        self.llm = dify_llm_async
        
        # Setup logging
        self._setup_logging()
        
        # Initialize system
        self._init_system()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        self.logger.info("CriticAgent initialized successfully")

    def _setup_logging(self):
        """Setup logging system"""
        self.logger = logging.getLogger("CriticAgent")
        self.logger.setLevel(self.config.log_level)
        
        log_dir = f"{self.config.ckpt_dir}/critic/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        handler = logging.FileHandler(f"{log_dir}/critic_agent.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates"""
        return {
            'system': load_prompt("critic/system"),
            'task': load_prompt("critic/task"),
            'code': load_prompt("critic/code")
        }

    def _init_system(self):
        """Initialize system"""
        try:
            self._init_directories()
            self.cache = CriticCache(self.config.cache_size)
            self._load_state()
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise RuntimeError("Failed to initialize CriticAgent") from e

    def _init_directories(self):
        """Initialize directory structure"""
        base_dir = f"{self.config.ckpt_dir}/critic/{self.config.knowledge_name}/{self.config.source_name}"
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
            cache_path = f"{base_path}/cache/critic_cache.json"
            if os.path.exists(cache_path):
                cache_data = U.load_json(cache_path)
                self.cache = CriticCache()
                for key, value in cache_data.items():
                    self.cache.add(key, value)
                
            self.logger.info(f"Loaded {len(self.cache.cache)} cache entries")
            
        except Exception as e:
            self.logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save current state"""
        try:
            base_path = self._get_base_path()
            
            # Save cache
            cache_data = {
                key: value 
                for key, value in self.cache.cache.items()
            }
            U.dump_json(
                cache_data,
                f"{base_path}/cache/critic_cache.json"
            )
            
            self.logger.debug("State saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise

    def _get_base_path(self) -> str:
        """Get base path for state files"""
        return f"{self.config.ckpt_dir}/critic/{self.config.knowledge_name}/{self.config.source_name}"

    def render_system_message(self) -> Dict[str, str]:
        """Render system message"""
        try:
            return {"content": self.prompts['system']}
            
        except Exception as e:
            self.logger.error(f"Failed to render system message: {e}")
            raise

    def render_human_message(
        self,
        task: str,
        context: str,
        code: str
    ) -> Dict[str, str]:
        """Render human message"""
        try:
            message = [
                f"Task: {task}",
                f"Context: {context}",
                f"Code: {code}"
            ]
            
            content = "\n\n".join(message)
            self.logger.debug(f"Human message:\n{content}")
            
            return {"content": content}
            
        except Exception as e:
            self.logger.error(f"Failed to render human message: {e}")
            raise

    async def check_task_success(
        self,
        task: str,
        context: str,
        code: str,
        max_retries: Optional[int] = None
    ) -> Tuple[bool, str]:
        """Check if task implementation is successful"""
        retries = max_retries or self.config.max_retries
        
        try:
            # Check cache
            cache_key = f"{task}_{code}"
            if cached := self.cache.get(cache_key):
                self.logger.info(f"Using cached result for task: {task}")
                return cached["success"], cached["critique"]
            
            if self.config.mode == "manual":
                return await self._human_check_task(task, context, code)
            
            # Prepare messages
            system_msg = self.render_system_message()
            human_msg = self.render_human_message(task, context, code)
            
            # Call LLM
            response = await self.llm(
                query=human_msg["content"],
                user="CriticAgent",
                inputs={"system": system_msg["content"]},
                logger=self.logger
            )
            
            if "error" in response:
                raise ValueError(f"LLM error: {response['error']}")
                
            # Parse response
            result = fix_and_parse_json(response["answer"])
            
            success = result.get("success", False)
            critique = result.get("critique", "")
            
            # Cache result
            self.cache.add(cache_key, {
                "success": success,
                "critique": critique,
                "timestamp": time.time()
            })
            
            return success, critique
            
        except Exception as e:
            self.logger.error(f"Task check failed: {e}")
            if retries > 0:
                self.logger.info(f"Retrying... ({retries} attempts left)")
                return await self.check_task_success(
                    task,
                    context, 
                    code,
                    retries - 1
                )
            return False, str(e)

    async def _human_check_task(
        self,
        task: str,
        context: str,
        code: str
    ) -> Tuple[bool, str]:
        """Manual task checking by human"""
        while True:
            print("\nTask Review:")
            print(f"Task: {task}")
            print(f"Context: {context}")
            print(f"Code:\n{code}")
            
            success = input("\nIs implementation successful? (y/n): ").lower() == 'y'
            critique = input("Enter critique (leave empty if none): ")
            
            confirm = input("\nConfirm review? (y/n): ").lower()
            if confirm in ['y', '']:
                return success, critique

    async def check_code_quality(
        self,
        code: str,
        requirements: Optional[str] = None
    ) -> Dict:
        """Check code quality and standards"""
        try:
            prompt = self.prompts['code'].replace(
                "{{code}}", code
            ).replace(
                "{{requirements}}", requirements or ""
            )
            
            response = await self.llm(
                query="Evaluate code quality",
                user="CriticAgent",
                inputs={"system": prompt},
                logger=self.logger
            )
            
            if "error" in response:
                raise ValueError(f"LLM error: {response['error']}")
                
            result = fix_and_parse_json(response["answer"])
            result["timestamp"] = time.time()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Code quality check failed: {e}")
            raise