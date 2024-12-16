from __future__ import annotations
import os
import time
import logging
import shutil
import signal
import platform
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from llms.dify_wrapper import dify_llm_async
import utils as U
from prompts import load_prompt
from agents.skill import SkillManager, SkillManagerConfig

@dataclass
class ActionConfig:
    """Action Agent Configuration"""
    observation_dir: str = "."
    ckpt_dir: str = "work_dir/ckpt"
    resume: bool = False
    mode: str = "auto"
    project_name: str = "default"
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
            project_name=self.config.project_name,
            resume=self.config.resume,
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
        
        log_dir = f"{self.config.ckpt_dir}/{self.config.project_name}/action/logs"
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
        base_dir = f"{self.config.ckpt_dir}/{self.config.project_name}/action"
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
        return f"{self.config.ckpt_dir}/{self.config.project_name}/action"

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

    async def execute_skill(self, skill_name: str, skill_code: str, *args, **kwargs) -> Dict:
        """
        Execute a specific skill by its name and code with security restrictions.

        Args:
            skill_name (str): The name of the skill to execute.
            skill_code (str): The code of the skill to execute.
            *args: Positional arguments to pass to the skill function.
            **kwargs: Keyword arguments to pass to the skill function.

        Returns:
            Dict: A structured response with execution status, message, data, and error details.
        """
        original_dir = os.getcwd()  # Save the original working directory
        try:
            # Change to the observation directory
            observation_dir = self.config.observation_dir
            if not os.path.exists(observation_dir):
                raise FileNotFoundError(f"Execution directory does not exist: {observation_dir}")
            os.chdir(observation_dir)

            # Log the execution start
            self.logger.info(f"Starting execution of skill: {skill_name} in directory: {observation_dir}")

            # Prepare safe execution environment
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "list": list,
                    "dict": dict,
                    "len": len,
                    "__import__": __import__,
                },
                "os": os,  # Explicitly include the 'os' module
            }
            safe_locals = {}

            # Step 1: Execute the skill code
            try:
                exec(skill_code, safe_globals, safe_locals)

                # Log the successful execution
                self.logger.info(f"Skill code executed successfully: {skill_name}")

                # Step 2: Find the function matching the skill_name
                if skill_name not in safe_locals or not callable(safe_locals[skill_name]):
                    raise ValueError(f"No callable function named '{skill_name}' found in the executed code.")

                # Retrieve the function
                skill_function = safe_locals[skill_name]
                self.logger.info(f"Executing function: {skill_name}")

                # Step 3: Call the function and store the result
                result = skill_function(*args, **kwargs)

                # Return the execution result
                return {
                    "status": "success",
                    "message": f"Skill executed successfully: {skill_name}",
                    "data": result,  # Store the result of the function execution
                    "error": None
                }
            except TimeoutError:
                self.logger.error("Code execution timed out!")
                return {
                    "status": "timeout",
                    "message": "Code execution timed out!",
                    "data": None,
                    "error": "TimeoutError"
                }
            except Exception as e:
                self.logger.error(f"Code execution failed: {e}")
                return {
                    "status": "error",
                    "message": f"Skill execution failed: {skill_name}",
                    "data": None,
                    "error": str(e)
                }
        except Exception as e:
            self.logger.error(f"Failed to execute skill: {e}")
            return {
                "status": "error",
                "message": "Failed to execute skill!",
                "data": None,
                "error": str(e)
            }
        finally:
            # Restore the original working directory
            os.chdir(original_dir)

    def _backup_file(self, file_path: str, backup_dir: str):
        """
        Backup a file before it is modified or deleted.

        Args:
            file_path (str): The path of the file to backup.
            backup_dir (str): The directory where backups are stored.
        """
        try:
            if os.path.exists(file_path):
                # Create a timestamped backup file
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                backup_path = os.path.join(
                    backup_dir, f"{os.path.basename(file_path)}.{timestamp}.bak"
                )
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"Backup created for {file_path} at {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to backup file {file_path}: {e}")

    def _setup_logging(self):
        """Setup logging system"""
        self.logger = logging.getLogger("ActionAgent")
        self.logger.setLevel(self.config.log_level)

        log_dir = f"{self.config.ckpt_dir}/{self.config.project_name}/action/logs"
        os.makedirs(log_dir, exist_ok=True)

        handler = logging.FileHandler(f"{log_dir}/action_agent.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)