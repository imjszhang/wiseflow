# File: agents/qa.py
import os
import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from llms.dify_wrapper import dify_llm_async
import utils as U
from env.project_observer import ProjectObserver
from prompts import load_prompt

@dataclass
class QAManagerConfig:
    """问答对管理器配置"""
    ckpt_dir: str = "work_dir/ckpt"
    observation_dir: str = os.path.abspath(os.path.dirname(__file__))  # 源目录
    resume: bool = False
    cache_size: int = 100
    log_level: str = "INFO"

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
    def __init__(self, config: Optional[QAManagerConfig] = None):

        self.config = config or QAManagerConfig()
        self.llm = dify_llm_async
        self.qa_pairs: Dict[str, QAPair] = {}
        self.cache_size = self.config.cache_size
        self.usage_count: Dict[str, int] = {}

        # 初始化 ProjectObserver
        self.project_observer = ProjectObserver(
            source_dir=self.config.observation_dir,
            target_dir=os.path.join("work_dir", "observation_data")
        )

        # 加载提示词
        self.prompts = self._load_prompts()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化系统
        self._init_system()
        
        self.logger.info("QAManager initialized successfully")


    def _load_prompts(self) -> Dict[str, str]:
        """Load all prompts"""
        return {
            'qa_step1': load_prompt("qa/qa_step1"),
            'qa_step2': load_prompt("qa/qa_step2")
        }
    
    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger("QAManager")
        self.logger.setLevel(self.config.log_level)
        
        log_dir = f"{self.config.ckpt_dir}/qa/logs"  
        os.makedirs(log_dir, exist_ok=True)
        
        handler = logging.FileHandler(f"{log_dir}/QA_manager.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _init_system(self):
        """初始化系统"""
        try:
            self._init_directories()
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise RuntimeError("Failed to initialize QAManager") from e

    def _init_directories(self):
        """初始化目录结构"""
        dirs = [
            f"{self.config.ckpt_dir}/qa/",  
            f"{self.config.ckpt_dir}/qa/logs"  
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.debug(f"Directory created/verified: {dir_path}")


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

    async def _generate_task_context(self, task: str) -> str:
        """Generate task context"""
        try:
            questions, concepts = await self.run_qa_step1(task)
            
            # Add questions and concepts to QA manager
            for q, c in zip(questions, concepts):
                self.add_pair(q, c)
                
            # Generate answers
            answers = await self.run_qa_step2(questions)
            
            # Update QA pairs with answers
            for q, a in zip(questions, answers):
                self.update_answer(q, a)
                
            qa_pairs = [
                self.get_pair(q)
                for q in questions
                if self.get_pair(q)
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


    @classmethod
    def from_dict(cls, data: Dict, cache_size: int = 100) -> 'QAManager':
        """Create instance from dictionary"""
        manager = cls(cache_size)
        for q, pair_data in data.items():
            pair = QAPair.from_dict(pair_data)
            manager.qa_pairs[q] = pair
            manager.usage_count[q] = 0
        return manager