from __future__ import annotations
import os
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from llms.dify_wrapper import dify_llm_async
import utils as U
from prompts import load_prompt

@dataclass
class ActionConfig:
    """Action Agent Configuration"""
    observation_dir: str = "."
    ckpt_dir: str = "work_dir/ckpt"
    resume: bool = False
    mode: str = "auto"
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
            'human': load_prompt("action/human")
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
        base_dir = f"{self.config.ckpt_dir}/action"
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
        return f"{self.config.ckpt_dir}/action"

    def render_system_message(self, skills: List[str]) -> Dict:
        """Render system message for the LLM"""
        system_message = {
            "content": self.prompts['system'].replace("{{skills}}", "\n".join(skills))
        }
        return system_message

    def render_human_message(self, events: List[Dict], code: str, task: str, context: str, critique: str) -> Dict:
        """Render human message for the LLM"""
        human_message = {
            "content": self.prompts['human']
                .replace("{{events}}", "\n".join([str(event) for event in events]))
                .replace("{{code}}", code)
                .replace("{{task}}", task)
                .replace("{{context}}", context)
                .replace("{{critique}}", critique)
        }
        return human_message

    async def llm(self, messages: List[Dict]) -> str:
        """Call the LLM to generate a response"""
        try:
            response = await self.llm(
                query=messages[1]["content"],
                user="ActionAgent",
                inputs={"system": messages[0]["content"]},
                temperature=self.config.temperature,
                timeout=self.config.request_timeout,
                logger=self.logger
            )
            return response["answer"]
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise RuntimeError("Failed to call LLM") from e

    def process_ai_message(self, message: str) -> Dict:
        """Process the AI-generated message"""
        # This method should parse the AI message and extract the code or instructions
        # For now, we assume the message is directly the code
        return {
            "program_code": message,
            "exec_code": ""
        }