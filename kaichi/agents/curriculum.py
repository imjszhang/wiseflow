# File: agents/curriculum.py
from __future__ import annotations
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import re

from llms.dify_wrapper import dify_llm_async
import utils as U
from prompts import load_prompt
from envs.project_observer import ProjectObserver
import re

def extract_json_from_markdown(content: str) -> str:
    """
    Extract JSON content from a Markdown code block (e.g., ```json ... ```).
    
    Args:
        content (str): The content containing the Markdown code block.
    
    Returns:
        str: The extracted JSON content, or an empty string if no valid JSON block is found.
    """
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, content, re.DOTALL)  # DOTALL allows `.` to match newlines
    if match:
        return match.group(1).strip()  # Extract the JSON content inside the code block
    return ""  # Return an empty string if no match is found


@dataclass
class CurriculumConfig:
    """Curriculum Manager Configuration"""
    ckpt_dir: str = "work_dir/ckpt"  # 工作目录
    mode: str = "auto"  # auto 或 manual
    source_content: str = "init"
    observation_dir: str = os.path.abspath(os.path.dirname(__file__))  # 源目录
    max_retries: int = 5
    log_level: str = "INFO"
    cache_size: int = 100
    warm_up: Optional[Dict] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.mode not in ["auto", "manual"]:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.cache_size <= 0:
            raise ValueError(f"Invalid cache size: {self.cache_size}")
        if self.max_retries <= 0:
            raise ValueError(f"Invalid max retries: {self.max_retries}")
        # 确保工作目录存在
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
class QAPair:
    """Question-Answer Pair Structure"""
    def __init__(self, question: str, concept: str, answer: Optional[str] = None):
        self.question = question
        self.concept = concept
        self.answer = answer
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "question": self.question,
            "concept": self.concept,
            "answer": self.answer,
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'QAPair':
        """Create instance from dictionary"""
        pair = cls(data["question"], data["concept"])
        pair.answer = data.get("answer")
        pair.timestamp = data.get("timestamp", time.time())
        return pair

class QAManager:
    """Manager for QA operations"""
    def __init__(self, cache_size: int = 100):
        self.qa_pairs: Dict[str, QAPair] = {}
        self.cache_size = cache_size
        self.usage_count: Dict[str, int] = {}
        
    def add_pair(self, question: str, concept: str, answer: Optional[str] = None):
        """Add new QA pair"""
        if len(self.qa_pairs) >= self.cache_size:
            self._remove_least_used()
            
        self.qa_pairs[question] = QAPair(question, concept, answer)
        self.usage_count[question] = 0
        
    def get_pair(self, question: str) -> Optional[QAPair]:
        """Retrieve QA pair"""
        if question in self.qa_pairs:
            self.usage_count[question] += 1
            return self.qa_pairs[question]
        return None
        
    def update_answer(self, question: str, answer: str):
        """Update answer for existing question"""
        if pair := self.get_pair(question):
            pair.answer = answer
            pair.timestamp = time.time()
            
    def _remove_least_used(self):
        """Remove least used QA pair"""
        if not self.usage_count:
            return
            
        min_usage = min(self.usage_count.values())
        to_remove = next(k for k, v in self.usage_count.items() if v == min_usage)
        
        self.qa_pairs.pop(to_remove, None)
        self.usage_count.pop(to_remove, None)
        
    def to_dict(self) -> Dict:
        """Convert all QA pairs to dictionary"""
        return {
            q: pair.to_dict() 
            for q, pair in self.qa_pairs.items()
        }
        
    @classmethod
    def from_dict(cls, data: Dict, cache_size: int = 100) -> 'QAManager':
        """Create instance from dictionary"""
        manager = cls(cache_size)
        for q, pair_data in data.items():
            pair = QAPair.from_dict(pair_data)
            manager.qa_pairs[q] = pair
            manager.usage_count[q] = 0
        return manager

class TaskProgress:
    """Task Progress Manager"""
    def __init__(self):
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.last_updated: Dict[str, float] = {}
        
    def add_completed_task(self, task: str):
        """Add completed task"""
        if task not in self.completed_tasks:
            self.completed_tasks.append(task)
            self.last_updated[task] = time.time()
            
        if task in self.failed_tasks:
            self.failed_tasks.remove(task)
            
    def add_failed_task(self, task: str):
        """Add failed task"""
        if task not in self.failed_tasks and task not in self.completed_tasks:
            self.failed_tasks.append(task)
            self.last_updated[task] = time.time()
            
    def get_status(self, task: str) -> Dict:
        """Get task status"""
        return {
            "completed": task in self.completed_tasks,
            "failed": task in self.failed_tasks,
            "last_updated": self.last_updated.get(task)
        }
        
    @property
    def progress(self) -> int:
        """Get overall progress"""
        return len(self.completed_tasks)
        
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "last_updated": self.last_updated
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskProgress':
        """Create instance from dictionary"""
        progress = cls()
        progress.completed_tasks = data.get("completed_tasks", [])
        progress.failed_tasks = data.get("failed_tasks", [])
        progress.last_updated = data.get("last_updated", {})
        return progress

class CurriculumAgent:
    """Curriculum Manager"""
    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self.llm = dify_llm_async
        
        # 初始化 ProjectObserver
        self.project_observer = ProjectObserver(
            source_dir=self.config.observation_dir,
            target_dir=os.path.join("work_dir", "observation_data")
        )
        
        # Setup logging
        self._setup_logging()
        
        # Initialize system
        self._init_system()
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        # Initialize warm up config
        self._init_warm_up()
        
        self.logger.info("CurriculumAgent initialized successfully")

    def _setup_logging(self):
        """Setup logging system"""
        self.logger = logging.getLogger("CurriculumAgent")
        self.logger.setLevel(self.config.log_level)
        
        log_dir = f"{self.config.ckpt_dir}/curriculum/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        handler = logging.FileHandler(f"{log_dir}/curriculum_agent.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _load_prompts(self) -> Dict[str, str]:
        """Load all prompts"""
        return {
            'task_proposal': load_prompt("curriculum/task_proposal"),
            'task_context': load_prompt("curriculum/task_context"),
            'qa_step1': load_prompt("curriculum/qa_step1"),
            'qa_step2': load_prompt("curriculum/qa_step2")
        }

    def _init_system(self):
        """Initialize system"""
        try:
            self._init_directories()
            self.task_progress = TaskProgress()
            self.qa_manager = QAManager(self.config.cache_size)
            self._load_state()
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise RuntimeError("Failed to initialize CurriculumAgent") from e

    def _init_directories(self):
        """Initialize directory structure"""
        dirs = [
            f"{self.config.ckpt_dir}/curriculum",
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.debug(f"Directory created/verified: {dir_path}")

    def _init_warm_up(self):
        """Initialize warm up configuration"""
        self.default_warm_up = {
            "context": 15,
            "completed_tasks": 0,
            "failed_tasks": 0
        }
        if not self.config.warm_up:
            self.warm_up = self.default_warm_up
        else:
            self.warm_up = {}
            for key in self.curriculum_observations:
                self.warm_up[key] = self.config.warm_up.get(
                    key, 
                    self.default_warm_up[key]
                )
                
        self.warm_up["completed_tasks"] = 0
        self.warm_up["failed_tasks"] = 0
    def _load_state(self):
        """Load saved state"""
        try:
            base_path = self._get_base_path()
            
            # Load progress
            progress_data = U.load_json(f"{base_path}/progress.json")
            self.task_progress = TaskProgress.from_dict(progress_data)
            
            # Load QA pairs
            qa_data = U.load_json(f"{base_path}/qa_pairs.json")
            self.qa_manager = QAManager.from_dict(
                qa_data, 
                self.config.cache_size
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save current state"""
        try:
            base_path = self._get_base_path()
            
            # Save progress
            U.dump_json(
                self.task_progress.to_dict(),
                f"{base_path}/progress.json"
            )
            
            # Save QA pairs
            U.dump_json(
                self.qa_manager.to_dict(),
                f"{base_path}/qa_pairs.json"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise

    def _get_base_path(self) -> str:
        """Get base path for state files"""
        return f"{self.config.ckpt_dir}/curriculum"  # 修改此行

    async def propose_next_task(self, max_retries: Optional[int] = None) -> Tuple[str, str]:
        """Propose next task"""
        max_retries = max_retries or self.config.max_retries
        
        try:
            if self.config.mode == "auto":
                return await self._propose_next_ai_task(max_retries)
            elif self.config.mode == "manual":
                return await self._propose_next_manual_task()
            else:
                raise ValueError(f"Invalid mode: {self.config.mode}")
                
        except Exception as e:
            self.logger.error(f"Failed to propose next task: {e}")
            raise

    def _prepare_task_proposal_messages(self) -> List[Dict]:
        """Prepare messages for task proposal"""
        try:
            completed_tasks_str = ", ".join(self.task_progress.completed_tasks) or "None"
            failed_tasks_str = ", ".join(self.task_progress.failed_tasks) or "None"
            
            system_message = {
                "content": self.prompts['task_proposal'].replace(
                    "{{completed_tasks}}", completed_tasks_str
                ).replace(
                    "{{failed_tasks}}", failed_tasks_str
                ).replace(
                    "{{knowledge_level}}", "intermediate"
                )
            }
            
            human_message = {
                "content": "Based on the current progress and context, propose the next task."
            }
            
            return [system_message, human_message]
            
        except Exception as e:
            self.logger.error(f"Failed to prepare task proposal messages: {e}")
            raise

    async def _propose_next_ai_task(self, retries: int) -> Tuple[str, str]:
        if retries <= 0:
            raise RuntimeError("Max retries reached")
        
        try:
            # Prepare messages
            messages = self._prepare_task_proposal_messages()
            
            # Call LLM
            response = await self.llm(
                query=messages[1]["content"],
                user="CurriculumAgent",
                inputs={"system": messages[0]["content"]},
                logger=self.logger
            )
            
            # Log the raw response
            self.logger.debug(f"Raw LLM Response: {response}")
            
            if "error" in response:
                raise ValueError(f"LLM error: {response['error']}")
            
            # Parse response
            curriculum = response["answer"]
            
            # Log the raw curriculum response
            self.logger.debug(f"Raw curriculum response: {curriculum}")
            
            # Extract JSON content from Markdown
            curriculum = extract_json_from_markdown(curriculum)
            
            # Log the cleaned curriculum response
            self.logger.debug(f"Extracted JSON response: {curriculum}")
            
            # Parse the cleaned JSON
            parsed_response = json.loads(curriculum)
            
            if "next_task" not in parsed_response:
                raise ValueError("No next task in response")
            
            # Get task context
            next_task = parsed_response["next_task"]
            context = await self.get_task_context(next_task)
            
            return next_task, context
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.error(f"Invalid LLM response: {response.get('answer', 'No answer')}")
            return await self._propose_next_ai_task(retries - 1)
        except Exception as e:
            self.logger.warning(f"Error in AI task proposal: {e}")
            return await self._propose_next_ai_task(retries - 1)

    async def _propose_next_manual_task(self) -> Tuple[str, str]:
        """Manual task proposal (placeholder)"""
        raise NotImplementedError("Manual mode not implemented yet")

    async def get_task_context(self, task: str) -> str:
        """Get task context"""
        try:
            qa_pair = self.qa_manager.get_pair(task)
            if qa_pair and qa_pair.answer:
                return self._format_qa_context(task, [qa_pair])
                
            context = await self._generate_task_context(task)
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get task context: {e}")
            raise

    async def _generate_task_context(self, task: str) -> str:
        """Generate task context"""
        try:
            questions, concepts = await self.run_qa_step1(task)
            
            # Add questions and concepts to QA manager
            for q, c in zip(questions, concepts):
                self.qa_manager.add_pair(q, c)
                
            # Generate answers
            answers = await self.run_qa_step2(questions)
            
            # Update QA pairs with answers
            for q, a in zip(questions, answers):
                self.qa_manager.update_answer(q, a)
                
            qa_pairs = [
                self.qa_manager.get_pair(q)
                for q in questions
                if self.qa_manager.get_pair(q)
            ]
            
            return self._format_qa_context(task, qa_pairs)
            
        except Exception as e:
            self.logger.error(f"Failed to generate context: {e}")
            raise

    def _format_qa_context(self, task: str, qa_pairs: List[QAPair]) -> str:
        """Format QA pairs into context"""
        context_parts = [
            f"Task: {task}",
            "\nKey Concepts:",
            *[f"- {pair.concept}" for pair in qa_pairs],
            "\nTechnical Q&A:"
        ]
        
        for pair in qa_pairs:
            if pair.answer:
                context_parts.extend([
                    "\nQuestion:",
                    pair.question,
                    "Answer:",
                    pair.answer
                ])
                
        return "\n".join(context_parts)

    async def run_qa_step1(self, task) -> Tuple[List[str], List[str]]:
        """Execute QA Step 1 - Generate questions and concepts"""
        try:
            system_message = {
                "content": self.prompts['qa_step1'].replace(
                    "{{Task}}", task
                )
            }

            content = f"Source Material:\n{self.config.source_content}"
            
            response = await self.llm(
                query=content,
                user="CurriculumAgent",
                inputs={"system": system_message["content"]},
                logger=self.logger
            )
            
            if "error" in response:
                raise ValueError(f"LLM error: {response['error']}")
                
            qa_response = response["answer"]
            pattern = r"Question (\d+): (.+)\nConcept \d+: (.+)"
            pairs = re.findall(pattern, qa_response)
            
            if not pairs:
                self.logger.warning("No valid question-concept pairs found")
                return [], []
                
            questions = [pair[1].strip() for pair in pairs]
            concepts = [pair[2].strip() for pair in pairs]
            
            self.logger.info(f"Generated {len(questions)} question-concept pairs")
            return questions, concepts
            
        except Exception as e:
            self.logger.error(f"Failed in QA step 1: {e}")
            return [], []

    async def run_qa_step2(self, questions: List[str]) -> List[Optional[str]]:
        """Execute QA Step 2 - Generate answers"""
        answers = []
        
        for question in questions:
            try:
                system_message = self.prompts['qa_step2']
                content = (
                    f"Question:\n{question}\n\n"
                    f"Source Material:\n{self.config.source_content}"
                )
                
                response = await self.llm(
                    query=content,
                    user="CurriculumAgent",
                    inputs={"system": system_message},
                    logger=self.logger
                )
                
                if "error" in response:
                    raise ValueError(f"LLM error: {response['error']}")
                    
                answers.append(response["answer"])
                self.logger.debug(f"Generated answer for: {question[:50]}...")
                
            except Exception as e:
                self.logger.error(f"Failed to answer: {question[:50]}... - {e}")
                answers.append(None)
                
        self.logger.info(f"Generated {len(answers)} answers")
        return answers

    def update_exploration_progress(self, info: Dict):
        """Update exploration progress"""
        task = info["task"]
        
        try:
            if info["success"]:
                self.logger.info(f"Task completed: {task}")
                self.task_progress.add_completed_task(task)
            else:
                self.logger.info(f"Task failed: {task}")
                self.task_progress.add_failed_task(task)
                
            self._save_state()
            
        except Exception as e:
            self.logger.error(f"Failed to update progress: {e}")
            raise

    @property
    def curriculum_observations(self) -> List[str]:
        """Curriculum observation items"""
        return [
            "context",
            "completed_tasks", 
            "failed_tasks",
        ]

    @property
    def progress(self) -> int:
        """Get total progress"""
        return self.task_progress.progress

    def extract_and_save_observation(self):
        """
        提取项目的观察数据并保存到目标目录。
        """
        try:
            self.logger.info("Extracting project observation data...")
            observation = self.project_observer.observe_project()
            self.project_observer.save_observation(observation)
            self.logger.info("Project observation data saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to extract and save observation data: {e}")

    def get_saved_observation(self) -> Optional[Dict]:
        """
        获取已保存的项目观察数据。
        :return: 观察数据字典，或 None（如果文件不存在）
        """
        try:
            observation_file = os.path.join(self.project_observer.target_dir, "project_observation.json")
            if not os.path.exists(observation_file):
                self.logger.warning("No saved observation data found.")
                return None
            
            with open(observation_file, "r", encoding="utf-8") as f:
                observation = json.load(f)
                self.logger.info("Loaded saved observation data.")
                return observation
        except Exception as e:
            self.logger.error(f"Failed to load saved observation data: {e}")
            return None

    async def get_task_context(self, task: str) -> str:
        """
        获取任务的上下文信息。
        :param task: 任务名称
        :return: 上下文字符串
        """
        try:
            # 提取并保存观察数据
            self.extract_and_save_observation()
            
            # 加载已保存的观察数据
            observation = self.get_saved_observation()
            if not observation:
                return f"No observation data available for task: {task}"
            
            # 格式化观察数据为上下文
            return self._format_observation_context(task, observation)
        except Exception as e:
            self.logger.error(f"Failed to get task context: {e}")
            raise

    def _format_observation_context(self, task: str, observation: Dict) -> str:
        """
        将观察数据格式化为上下文字符串。
        :param task: 任务名称
        :param observation: 观察数据字典
        :return: 上下文字符串
        """
        context_parts = [
            f"Task: {task}",
            "\nDirectory Structure:",
            *[f"- {item}" for item in observation.get("directory_structure", [])],
            "\nKey Files:",
        ]
        
        key_files = observation.get("key_files", {})
        for file_name, content in key_files.items():
            context_parts.append(f"\n{file_name}:\n{content[:200]}...")  # 只显示前 200 个字符
        
        context_parts.extend([
            "\nProject Meta:",
            f"File Count: {observation.get('meta', {}).get('file_count', 0)}",
            f"Directory Count: {observation.get('meta', {}).get('dir_count', 0)}",
            f"Total Size: {observation.get('meta', {}).get('total_size', 0)} bytes",
            "\nLog Summary:",
            *[f"- {log}" for log in observation.get("log_summary", [])],
            "\nCode Statistics:",
            f"Total Lines: {observation.get('code_statistics', {}).get('total_lines', 0)}",
            "File Types:",
        ])
        
        file_types = observation.get("code_statistics", {}).get("file_types", {})
        for ext, count in file_types.items():
            context_parts.append(f"- {ext}: {count} files")
        
        return "\n".join(context_parts)