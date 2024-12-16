import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from agents.curriculum import CurriculumAgent
from agents.skill import SkillManager
from agents.action import ActionAgent
from agents.critic import CriticAgent
import utils as U

@dataclass
class AgentConfig:
    max_iterations: int = 160
    max_retries: int = 5
    ckpt_dir: str = "work_dir/ckpt"
    resume: bool = False
    log_level: str = "INFO"
    skill_cache_size: int = 100

@dataclass 
class AgentState:
    current_task: Optional[str] = None
    iteration: int = 0
    success_count: int = 0
    failure_count: int = 0

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

class SkillCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.skills = {}
        self.usage_count = {}
        self.last_used = {}
        
    def add(self, name: str, content: str):
        if len(self.skills) >= self.max_size:
            # Remove least used skill
            min_usage = min(self.usage_count.values())
            to_remove = [k for k,v in self.usage_count.items() if v == min_usage][0]
            self._remove(to_remove)
            
        self.skills[name] = content
        self.usage_count[name] = 0
        self.last_used[name] = time.time()
        
    def get(self, name: str) -> Optional[str]:
        if name in self.skills:
            self.usage_count[name] += 1
            self.last_used[name] = time.time()
            return self.skills[name]
        return None
        
    def _remove(self, name: str):
        self.skills.pop(name, None)
        self.usage_count.pop(name, None) 
        self.last_used.pop(name, None)

class Kaichi:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.state = AgentState()
        self.metrics = AgentMetrics()
        
        self._setup_logging()
        self._init_agents()
        self._init_skill_cache()
        
        if self.config.resume:
            self._load_checkpoints()
            
        self.logger.info("Kaichi initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("Kaichi")
        self.logger.setLevel(self.config.log_level)
        
        handler = logging.FileHandler(f"{self.config.ckpt_dir}/agent.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _init_agents(self):
        """Initialize all required agents"""
        try:
            self.curriculum_agent = CurriculumAgent()
            self.action_agent = ActionAgent()
            self.critic_agent = CriticAgent()
            self.skill_manager = SkillManager()
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

    def reset(self, task: str, context: str = "") -> List:
        """Reset agent state for new task"""
        self.logger.info(f"Resetting agent for task: {task}")
        
        self.state.current_task = task
        self.state.iteration = 0
        
        # Reset metrics for new task
        self.metrics.reset()
        
        # Load skills and prepare messages
        skills = U.load_text(f"{self.config.ckpt_dir}/skills/skill.txt")
        system_message = self.action_agent.render_system_message(skills=skills)
        human_message = self.action_agent.render_human_message(
            code="", task=task, context=context, critique=""
        )
        
        self.messages = [system_message, human_message]
        self.conversations = []
        
        self.logger.debug(f"Human message: {human_message.content}")
        return self.messages

    def step(self) -> Tuple:
        """Execute one step of the agent"""
        start_time = time.time()
        
        try:
            if not self._validate_state():
                raise ValueError("Invalid agent state")

            # Generate code
            ai_message = self._generate_code()
            self.conversations.append((
                self.messages[0].content,
                self.messages[1].content,
                ai_message.content
            ))

            # Parse and validate code
            parsed_result = self._parse_code(ai_message.content)
            success, critique = self._validate_code(parsed_result)

            # Update messages and state
            self._update_messages(parsed_result, critique)
            self.state.iteration += 1

            # Calculate reward and check if done
            done = success or self.state.iteration >= self.config.max_retries
            reward = 1.0 if success else 0.0

            # Update metrics
            response_time = time.time() - start_time
            self.metrics.update(success, response_time, len(ai_message.content))

            return self.messages, reward, done, self._get_step_info(success, parsed_result)

        except Exception as e:
            self.logger.error(f"Step error: {e}")
            return self.messages, 0, True, {"success": False, "error": str(e)}

    def _validate_state(self) -> bool:
        """Validate current agent state"""
        return (
            self.state.iteration >= 0 and
            self.messages is not None and
            len(self.messages) == 2 and
            self.state.current_task is not None
        )

    def _generate_code(self):
        """Generate code using action agent"""
        return self.action_agent.llm(self.messages)

    def _parse_code(self, content: str) -> Dict:
        """Parse generated code content"""
        return {
            "program_code": content,
            "program_name": self.state.current_task
        }

    def _validate_code(self, parsed_result: Dict) -> Tuple[bool, str]:
        """Validate generated code"""
        return self.critic_agent.check_task_success(
            task=self.state.current_task,
            context=self.context,
            code=parsed_result["program_code"],
            max_retries=self.config.max_retries
        )

    def _update_messages(self, parsed_result: Dict, critique: str):
        """Update system and human messages"""
        system_message = self.action_agent.render_system_message(skills="")
        human_message = self.action_agent.render_human_message(
            code=parsed_result["program_code"],
            task=self.state.current_task,
            context=self.context,
            critique=critique
        )
        self.messages = [system_message, human_message]

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

    def rollout(self, task: str, context: str) -> Tuple:
        """Execute complete task rollout"""
        self.logger.info(f"Starting rollout for task: {task}")
        
        messages = self.reset(task=task, context=context)
        
        while True:
            messages, reward, done, info = self.step()
            if done:
                break
                
        self.logger.info(f"Rollout completed. Success: {info['success']}")
        return messages, reward, done, info

    def run(self, task: str = "", maxloop: int = 1):
        """Run agent with given task or curriculum"""
        self.logger.info(f"Starting agent run. Task: {task}, maxloop: {maxloop}")
        
        loop = 1
        while loop <= maxloop:
            try:
                # Get task and context
                if not task:
                    task, context = self.curriculum_agent.propose_next_task(
                        max_retries=self.config.max_retries
                    )
                else:
                    context = self.curriculum_agent.get_task_context(task)
                
                self.logger.info(f"Executing task {task} (Loop {loop}/{maxloop})")
                
                # Execute task
                messages, reward, done, info = self.rollout(
                    task=task,
                    context=context
                )
                
                # Handle results
                if info["success"]:
                    self.action_agent.update_skill_knowledge(info)
                    self.state.success_count += 1
                else:
                    self.state.failure_count += 1
                    
                self.curriculum_agent.update_exploration_progress(info)
                
            except Exception as e:
                self.logger.error(f"Error in run loop: {e}")
                time.sleep(3)
                info = {"task": task, "success": False}
                
            loop += 1
            
        self.logger.info("Agent run completed")
        return {
            "success_rate": self.metrics.success_rate,
            "total_steps": self.metrics.steps,
            "avg_response_time": self.metrics.avg_response_time
        }

def main():
    config = AgentConfig(
        max_iterations=160,
        max_retries=5,
        ckpt_dir="work_dir/ckpt",
        resume=False,
        log_level="INFO"
    )
    
    agent = Kaichi(config)
    results = agent.run()
    print(f"Run completed with results: {results}")

if __name__ == "__main__":
    main()