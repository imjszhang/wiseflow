from __future__ import annotations
import os
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import ast
import re

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


    def render_system_message(self, skills: List[str] = None) -> Dict:
        """
        Render system message for the LLM with only the essential base skills.

        Args:
            skills (List[str]): List of additional skills to include.

        Returns:
            Dict: Rendered system message.
        """
        # 如果 skills 为 None，则初始化为空列表
        if skills is None:
            skills = []

        # Define the essential base skills with Python implementations
        base_skills_code = {
            "readFile": """
        def read_file(file_path: str) -> str:
            \"\"\"Read the content of a specific file.\"\"\"
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                return f"Error reading file {file_path}: {str(e)}"
            """,
            "writeFile": """
        def write_file(file_path: str, content: str) -> bool:
            \"\"\"Write content to a specific file.\"\"\"
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                return True
            except Exception as e:
                print(f"Error writing to file {file_path}: {str(e)}")
                return False
            """
        }

        # Combine the base skills
        base_skills = "\n\n".join([base_skills_code["readFile"], base_skills_code["writeFile"]])

        # Combine base skills and additional skills
        all_skills = base_skills
        if skills:
            all_skills += "\n\n".join(skills)

        # Render the system message
        system_message = {
            "content": self.prompts['system'].replace("{{skills}}", all_skills)
        }

        return system_message

    def render_human_message(self, events: List[Dict], code: str, task: str, context: str, critique: str) -> Dict:
        """Render human message for the LLM"""
        # 如果 events 为空，则使用默认值
        events_content = "\n".join([str(event) for event in events]) if events else "No events available."
        
        human_message = {
            "content": self.prompts['human']
                .replace("{{events}}", events_content)
                .replace("{{code}}", code)
                .replace("{{task}}", task)
                .replace("{{context}}", context)
                .replace("{{critique}}", critique)
        }
        return human_message

    async def generate_code(self, messages: List[Dict]) -> str:
        """Call the LLM to generate a response"""
        try:
            response = await self.llm(
                query=messages[1]["content"],
                user="ActionAgent",
                inputs={"system": messages[0]["content"]},
                logger=self.logger
            )
            return response["answer"]
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise RuntimeError("Failed to call LLM") from e

    def process_ai_message(self, message: str) -> Dict:
        """
        Parse the AI response message, extract Python code blocks, and analyze their structure.

        Args:
            message (str): AI response message containing Python code blocks.

        Returns:
            Dict: A dictionary containing the program code, main function name, and execution code.

        Raises:
            RuntimeError: If the message cannot be parsed or code cannot be extracted.
        """
        assert isinstance(message, str), "Input message must be a string."

        retry = 1  # Set the number of retries
        error = None  # To store error information
        while retry > 0:
            try:
                self.logger.debug("Starting to parse AI message...")  # Log debug information

                # Extract Python code blocks
                code_pattern = re.compile(r"```(?:python)(.*?)```", re.DOTALL)
                code = "\n".join(code_pattern.findall(message))
                assert code.strip(), "No Python code found in the message."
                self.logger.debug(f"Extracted code block:\n{code}")  # Log the extracted code block

                # Parse the code using the AST module
                parsed = ast.parse(code)
                self.logger.debug("Code parsed successfully, extracting function information...")  # Log successful parsing

                # Extract function information
                functions = []
                for node in ast.walk(parsed):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function_info = {
                            "name": node.name,
                            "type": "AsyncFunction" if isinstance(node, ast.AsyncFunctionDef) else "Function",
                            "body": ast.unparse(node) if hasattr(ast, "unparse") else compile(ast.Module([node], []), filename="<ast>", mode="exec"),
                            "params": [arg.arg for arg in node.args.args],
                        }
                        functions.append(function_info)
                        self.logger.debug(f"Extracted function: {function_info}")  # Log extracted function information

                # Validate that at least one function exists
                assert len(functions) > 0, "No functions found in the code."
                self.logger.info(f"Extracted {len(functions)} functions in total.")  # Log the number of extracted functions

                # Find the main function (the last async function)
                main_function = None
                for function in reversed(functions):
                    if function["type"] == "AsyncFunction":
                        main_function = function
                        break

                # Validate that the main function exists
                assert main_function is not None, "No async function found. Your main function must be async."
                self.logger.info(f"Main function name: {main_function['name']}")  # Log the main function name

                # Generate the return result
                program_code = "\n\n".join(function["body"] for function in functions)
                exec_code = f"await {main_function['name']}()"
                self.logger.debug("Code parsing and main function extraction completed successfully.")  # Log success

                return {
                    "program_code": program_code,
                    "program_name": main_function["name"],
                    "exec_code": exec_code,
                }

            except Exception as e:
                retry -= 1
                error = e
                self.logger.error(f"Failed to parse AI message. Remaining retries: {retry}. Error: {e}")  # Log error

        # If retries are exhausted, return an error message
        self.logger.critical(f"Failed to parse AI message after retries. Error: {error}")  # Log critical error
        return f"Error parsing action response (before program execution): {error}"