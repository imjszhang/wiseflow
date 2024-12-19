# File: agents/curriculum.py
from __future__ import annotations
import os
import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from llms.dify_wrapper import dify_llm_async
import utils as U
from prompts import load_prompt
from envs.project_observer import ProjectObserver
from agents.qa import QAPair, QAManager
from agents.task import TaskProgress

@dataclass
class CurriculumConfig:
    """Curriculum Manager Configuration"""
    ckpt_dir: str = "work_dir/ckpt"  # 工作目录
    mode: str = "auto"  # auto 或 manual
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
            # 获取当前任务状态
            completed_tasks_str = ", ".join(self.task_progress.completed_tasks) or "None"
            failed_tasks_str = ", ".join(self.task_progress.failed_tasks) or "None"
            iteration_count = self.task_progress.iteration_count
            success_rate = self.task_progress.success_rate

            # 提取并保存观察数据
            self.extract_and_save_observation()

            # 加载已保存的观察数据
            observation = self.get_saved_observation()
            if not observation:
                raise ValueError("No observation data available")

            # 格式化项目观察数据
            directory_structure = "\n".join(observation.get("directory_structure", []))
            key_files = "\n".join(observation.get("key_files", {}).keys())
            project_meta = json.dumps(observation.get("meta", {}), indent=2)
            code_analysis = json.dumps(observation.get("code_analysis", {}), indent=2)

            # 构建系统消息
            system_message = {
                "content": self.prompts['task_proposal']
                    .replace("{{completed_tasks}}", completed_tasks_str)
                    .replace("{{failed_tasks}}", failed_tasks_str)
                    .replace("{{knowledge_level}}", "intermediate")
                    .replace("{{iteration_count}}", str(iteration_count))
                    .replace("{{success_rate}}", f"{success_rate:.2f}")
                    .replace("{{directory_structure}}", directory_structure)
                    .replace("{{key_files}}", key_files)
                    .replace("{{project_meta}}", project_meta)
                    .replace("{{code_analysis}}", code_analysis)
            }

            # 构建用户消息
            human_message = {
                "content": "Based on the current project observation and progress, propose the next learning task."
            }

            return [system_message, human_message]

        except Exception as e:
            self.logger.error(f"Failed to prepare task proposal messages: {e}")
            raise

    async def _propose_next_ai_task(self, retries: int) -> Tuple[str, str]:
        if retries <= 0:
            raise RuntimeError("Max retries reached")

        try:
            # 准备消息
            messages = self._prepare_task_proposal_messages()

            # 调用 LLM
            response = await self.llm(
                query=messages[1]["content"],
                user="CurriculumAgent",
                inputs={"system": messages[0]["content"]},
                logger=self.logger
            )

            # 记录原始响应
            self.logger.debug(f"Raw LLM Response: {response}")

            if "error" in response:
                raise ValueError(f"LLM error: {response['error']}")

            # 解析响应
            curriculum = response["answer"]

            # 提取 JSON 内容
            curriculum = U.extract_json_from_markdown(curriculum)
            parsed_response = json.loads(curriculum)

            if "next_task" not in parsed_response:
                raise ValueError("No next task in response")

            # 获取任务上下文
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
        """
        获取任务的上下文信息。
        :param task: 任务名称
        :return: 上下文字符串
        """
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

            # 加载已保存的观察数据
            observation = self.get_saved_observation()
            if not observation:
                return f"No observation data available for task: {task}"

            # 格式化观察数据为上下文
            source_content = self._format_observation_context(task, observation)
            content = f"Source Material:\n{source_content}"
            
            response = await self.llm(
                query=content,
                user="CurriculumAgent",
                inputs={"system": system_message["content"]},
                logger=self.logger
            )
            
            if "error" in response:
                raise ValueError(f"LLM error: {response['error']}")
                
            qa_response = response["answer"]
            
            # 提取 JSON 内容
            qa_response = U.extract_json_from_markdown(qa_response)
            parsed_response = json.loads(qa_response)
            
            # 从 JSON 中提取问题和概念
            question_concept_pairs = parsed_response.get("questions", [])
            if not question_concept_pairs:
                self.logger.warning("No valid question-concept pairs found")
                return [], []
            
            # 分离问题和概念
            questions = [pair["question"].strip() for pair in question_concept_pairs]
            concepts = [pair["concept"].strip() for pair in question_concept_pairs]
            
            self.logger.info(f"Generated {len(questions)} question-concept pairs")
            return questions, concepts

        except Exception as e:
            self.logger.error(f"Error in run_qa_step1: {str(e)}")
            return [], []

    async def run_qa_step2(self, questions: List[str]) -> List[Optional[str]]:
        """Execute QA Step 2 - Generate answers"""
        answers = []
        
        for question in questions:
            try:

                # 加载已保存的观察数据
                observation = self.get_saved_observation()
                if not observation:
                    return f"No observation data available for question: {question}"

                # 格式化观察数据为上下文
                source_content= self._format_observation_context(question, observation)
                system_message = self.prompts['qa_step2']
                content = (
                    f"Question:\n{question}\n\n"
                    f"Source Material:\n{source_content}"
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

    def _format_observation_context(self, task: str, observation: Dict) -> str:
        """
        将观察数据格式化为上下文字符串。
        :param task: 任务名称
        :param observation: 观察数据字典
        :return: 上下文字符串
        """
        if not observation or not isinstance(observation, dict):
            self.logger.warning("Observation data is empty, None, or not a dictionary.")
            return f"No valid observation data available for task: {task}"

        try:
            context_parts = [
                f"Task: {task}",
                "\nDirectory Structure:",
            ]

            # 处理目录结构
            directory_structure = observation.get("directory_structure", [])
            if isinstance(directory_structure, list):
                context_parts.extend([f"- {item}" for item in directory_structure])
            else:
                self.logger.warning("Invalid directory structure format.")
                context_parts.append("- (Invalid directory structure)")

            # 处理关键文件
            context_parts.append("\nKey Files:")
            key_files = observation.get("key_files", {})
            if isinstance(key_files, dict):
                for file_name, content in key_files.items():
                    if isinstance(file_name, str) and isinstance(content, str):
                        context_parts.append(f"\n{file_name}:\n{content[:200]}...")  # 只显示前 200 个字符
                    else:
                        self.logger.warning(f"Invalid key file entry: {file_name}")
                        context_parts.append(f"\n{file_name}: (Invalid content)")
            else:
                self.logger.warning("Invalid key files format.")
                context_parts.append("(No valid key files)")

            # 处理项目元信息
            meta = observation.get("meta", {})
            if isinstance(meta, dict):
                context_parts.extend([
                    "\nProject Meta:",
                    f"File Count: {meta.get('file_count', 0)}",
                    f"Directory Count: {meta.get('dir_count', 0)}",
                    f"Total Size: {meta.get('total_size', 0)} bytes",
                ])
            else:
                self.logger.warning("Invalid meta format.")
                context_parts.append("\nProject Meta: (Invalid meta data)")

            # 处理代码分析
            code_analysis = observation.get("code_analysis", {})
            if isinstance(code_analysis, dict):
                context_parts.append("\nCode Analysis:")
                modules = code_analysis.get("modules", [])
                if isinstance(modules, list):
                    context_parts.append(f"Modules: {', '.join(modules)}")
                else:
                    self.logger.warning("Invalid modules format.")
                    context_parts.append("Modules: (Invalid modules data)")

                # 处理类信息
                context_parts.append("\nClasses:")
                classes = code_analysis.get("classes", {})
                if isinstance(classes, dict):
                    for class_name, class_info in classes.items():
                        if isinstance(class_info, dict):
                            methods = ", ".join(class_info.get("methods", []))
                            context_parts.append(f"- {class_name}: Methods: {methods}")
                        else:
                            self.logger.warning(f"Invalid class info for {class_name}.")
                            context_parts.append(f"- {class_name}: (Invalid class info)")
                else:
                    self.logger.warning("Invalid classes format.")
                    context_parts.append("(No valid classes)")

                # 处理函数信息
                context_parts.append("\nFunctions:")
                functions = code_analysis.get("functions", {})
                if isinstance(functions, dict):
                    for func_name, func_info in functions.items():
                        if isinstance(func_info, dict):
                            args = ", ".join(func_info.get("args", []))
                            context_parts.append(f"- {func_name}: Args: {args}")
                        else:
                            self.logger.warning(f"Invalid function info for {func_name}.")
                            context_parts.append(f"- {func_name}: (Invalid function info)")
                else:
                    self.logger.warning("Invalid functions format.")
                    context_parts.append("(No valid functions)")
            else:
                self.logger.warning("Invalid code analysis format.")
                context_parts.append("\nCode Analysis: (Invalid code analysis data)")

            return "\n".join(context_parts)

        except Exception as e:
            self.logger.error(f"Error while formatting observation context: {e}")
            return f"An error occurred while processing the observation data for task: {task}"