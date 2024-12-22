#Filename: kaichi.py
import os
import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from env import ProjectEnv  
from agents.curriculum import CurriculumAgent, CurriculumConfig
from agents.skill import SkillManager, SkillManagerConfig, SkillCache
from agents.action import ActionAgent, ActionConfig
from agents.critic import CriticAgent, CriticConfig
import utils as U

@dataclass
class AgentConfig:
    max_iterations: int = 160
    max_retries: int = 5
    env_timeout: int = 5  # ProjectEnv 的超时时间
    env_log_path: str = "work_dir/env"  # ProjectEnv 的日志路径
    observation_dir: str = os.path.abspath(os.path.dirname(__file__))
    ckpt_dir: str = "work_dir/ckpt"
    resume: bool = False
    log_level: str = "INFO"
    skill_cache_size: int = 100

@dataclass 
class AgentState:
    env: Optional[ProjectEnv] = None  # 使用 ProjectEnv
    current_task: Optional[str] = None
    context: Optional[str] = None
    iteration: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_state: Optional[Dict] = None  # 记录上一次的状态

class AgentMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.steps = 0
        self.success_rate = 0.0
        self.avg_response_time = 0.0
        self.total_tokens = 0
        
    def update(self, success: bool, response_time: float, tokens: int):
        self.steps += 1
        self.total_tokens += tokens
        self.success_rate = (self.success_rate * (self.steps - 1) + int(success)) / self.steps
        self.avg_response_time = (self.avg_response_time * (self.steps - 1) + response_time) / self.steps

class Kaichi:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.state = AgentState()
        self.metrics = AgentMetrics()
        
        self._setup_logging()
        self._init_env()
        self._init_agents()
        self._init_skill_cache()
        
        if self.config.resume:
            self._load_checkpoints()
            
        self.logger.info("Kaichi initialized successfully")


    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("Kaichi")
        self.logger.setLevel(self.config.log_level)
        
        # 使用 RotatingFileHandler 实现日志轮转
        log_file = f"{self.config.ckpt_dir}/agent.log"
        handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)  # 每个日志文件最大 5MB，保留 3 个备份
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _init_env(self):
        """Initialize the environment"""
        try:
            self.state.env = ProjectEnv(
                timeout=self.config.env_timeout,
                log_path=self.config.env_log_path
            )
            self.logger.info("Environment initialized successfully with timeout=%s and log_path=%s",
                            self.config.env_timeout, self.config.env_log_path)
        except Exception as e:
            self.logger.error("Failed to initialize environment: %s", e, exc_info=True)
            raise

    def _init_agents(self):
        """Initialize all required agents"""
        try:
            # 配置 CurriculumAgent
            curriculum_config = CurriculumConfig(
                ckpt_dir=self.config.ckpt_dir,  
                observation_dir=self.config.observation_dir,  
                mode="auto",
                max_retries=3,
                log_level=self.config.log_level,
                cache_size=10
            )

            # 配置 ActionAgent
            action_config = ActionConfig(
                ckpt_dir=self.config.ckpt_dir,  
                observation_dir=self.config.observation_dir,  
                resume=False,  
                mode="auto",
                max_retries=3,
                log_level=self.config.log_level,
                cache_size=10,
                temperature=0.7,
                request_timeout=60
            )

            # 配置 CriticAgent
            critic_config = CriticConfig(
                ckpt_dir=self.config.ckpt_dir,  
                observation_dir=self.config.observation_dir,  
                log_level=self.config.log_level
            )

            # Initialize skill manager
            skill_config = SkillManagerConfig(
                resume=self.config.resume,
                ckpt_dir=self.config.ckpt_dir,
                cache_size=self.config.skill_cache_size,
                log_level=self.config.log_level
            )

            self.curriculum_agent = CurriculumAgent(curriculum_config)
            self.action_agent = ActionAgent(action_config)
            self.critic_agent = CriticAgent(critic_config)            
            self.skill_manager = SkillManager(skill_config)
            self.logger.info("All agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise RuntimeError("Agent initialization failed") from e

    def _init_skill_cache(self):
        """Initialize skill cache"""
        self.skill_cache = SkillCache(self.config.skill_cache_size)

    def _load_checkpoints(self):
        """Load checkpoints if resume is enabled"""
        try:
            # Load skills from checkpoint
            skills = U.load_text(f"{self.config.ckpt_dir}/skills/skill.txt")
            for skill in skills.split("\n"):
                if skill.strip():
                    self.skill_cache.add("basic_skill", skill)
            self.logger.info("Checkpoints loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoints: {e}")
            raise RuntimeError("Checkpoint loading failed") from e

    async def reset(self, task: str, context: str = "") -> List:
        """Reset agent state for new task"""
        self.logger.info("Resetting agent for task: %s with context: %s", task, context)
        
        try:
            self.state.current_task = task
            self.state.context = context
            self.state.iteration = 0
            
            # Reset metrics for new task
            self.metrics.reset()
            self.logger.debug("Metrics reset successfully")

            # Reset environment
            initial_state, _ = self.state.env.reset()
            self.state.last_state = initial_state
            self.logger.info("Environment reset successfully. Initial state: %s", initial_state)

            # Load skills and prepare messages
            skills = await self.skill_manager.retrieve_skills(query=self.state.context)
            self.logger.debug("Retrieved skills: %s", skills)

            system_message = self.action_agent.render_system_message(skills=skills)
            human_message = self.action_agent.render_human_message(
                events=[], code="", task=task, context=self.state.context, critique=""
            )
            
            self.messages = [system_message, human_message]
            self.conversations = []
            
            self.logger.debug("Human message: %s", human_message['content'])
            return self.messages
        except Exception as e:
            self.logger.error("Failed to reset agent: %s", e, exc_info=True)
            raise

    async def step(self) -> Tuple:
        """Execute one step of the agent"""
        start_time = time.time()
        self.logger.info("Starting step %d for task: %s", self.state.iteration, self.state.current_task)
        
        try:
            if not self._validate_state():
                raise ValueError("Invalid agent state")

            # Generate code
            ai_message = await self._generate_code()
            self.logger.debug("Generated AI message: %s", ai_message)

            # Parse and execute code
            parsed_result = self._parse_code(ai_message)
            code = parsed_result["program_code"]
            self.logger.info("Parsed program code: %s", code)

            state, reward, done, _, info = self.state.env.step(code)
            self.state.last_state = state
            self.logger.info("Environment step executed. State: %s, Reward: %s, Done: %s", state, reward, done)

            # Validate code execution
            success, critique = self._validate_code(parsed_result, state)
            self.logger.info("Code validation result: Success=%s, Critique=%s", success, critique)

            # Update messages and state
            self._update_messages(parsed_result, critique, state)
            self.state.iteration += 1

            # Update metrics
            response_time = time.time() - start_time
            self.metrics.update(success, response_time, len(ai_message))
            self.logger.info("Step completed. Success=%s, Response time=%.2f seconds", success, response_time)

            return self.messages, reward, done, self._get_step_info(success, parsed_result)

        except Exception as e:
            self.logger.error("Step error: %s", e, exc_info=True)
            return self.messages, 0, True, {"success": False, "error": str(e)}

    def _validate_code(self, parsed_result: Dict, state: Dict) -> Tuple[bool, str]:
        """Validate generated code"""
        return self.critic_agent.check_task_success(
            task=self.state.current_task,
            context=self.state.context,
            code=parsed_result["program_code"],
            state=state,
            max_retries=self.config.max_retries
        )

    def _update_messages(self, parsed_result: Dict, critique: str, state: Dict):
        """Update system and human messages"""
        system_message = self.action_agent.render_system_message(skills="")
        human_message = self.action_agent.render_human_message(
            events=[],  # ProjectEnv 不返回事件，使用空列表
            code=parsed_result["program_code"],
            task=self.state.current_task,
            context=self.state.context,
            critique=critique
        )
        self.messages = [system_message, human_message]

    def _validate_state(self) -> bool:
        """Validate current agent state"""
        return (
            self.state.iteration >= 0 and
            self.messages is not None and
            len(self.messages) == 2 and
            self.state.current_task is not None
        )

    async def _generate_code(self):
        """Generate code using action agent"""
        return await self.action_agent.generate_code(self.messages)

    def _parse_code(self, content: str) -> Dict:
        """Parse generated code content"""
        return self.action_agent.process_ai_message(content)

    def _get_step_info(self, success: bool, parsed_result: Optional[Dict] = None) -> Dict:
        """Prepare step information"""
        info = {
            "task": self.state.current_task,
            "success": success,
            "conversations": self.conversations
        }
        
        if success and parsed_result:
            info.update({
                "program_code": parsed_result["program_code"],
                "program_name": parsed_result["program_name"]
            })
            
        return info

    async def rollout(self, task: str, context: str) -> Tuple:
        """Execute complete task rollout"""
        self.logger.info(f"Starting rollout for task: {task}")
        
        messages = await self.reset(task=task, context=context)
        
        while True:
            messages, reward, done, info = await self.step()
            if done:
                break
                
        self.logger.info(f"Rollout completed. Success: {info['success']}")
        return messages, reward, done, info

    async def learn(self, task: str = "", maxloop: int = 1):
        """Run agent with given task or curriculum"""
        self.logger.info("Starting agent run. Task: %s, maxloop: %d", task, maxloop)
        
        loop = 1
        while loop <= maxloop:
            try:
                """
                if not task:
                    task, context = await self.curriculum_agent.propose_next_task(
                        max_retries=self.config.max_retries
                    )
                else:
                    context = await self.curriculum_agent.get_task_context(task)
                """
                task="Implement a basic insight extraction feature using the get_insights function"
                context=U.load_text("test_context.txt")


                self.logger.info("Executing task %s (Loop %d/%d)", task, loop, maxloop)
                
                messages, reward, done, info = await self.rollout(task=task, context=context)
                
                if info["success"]:
                    self.logger.info("Task %s completed successfully", task)
                    await self.skill_manager.add_new_skill(info)
                    self.state.success_count += 1
                else:
                    self.logger.warning("Task %s failed", task)
                    self.state.failure_count += 1
                    
                self.curriculum_agent.update_exploration_progress(info)
            except Exception as e:
                self.logger.error("Error in run loop: %s", e, exc_info=True)
                time.sleep(3)
                info = {"task": task, "success": False}
            loop += 1
            
        self.logger.info("Agent run completed. Success rate: %.2f", self.metrics.success_rate)
        return {
            "success_rate": self.metrics.success_rate,
            "total_steps": self.metrics.steps,
            "avg_response_time": self.metrics.avg_response_time
        }