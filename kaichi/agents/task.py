from typing import Dict, List
import time

class TaskProgress:
    """Task Progress Manager"""
    def __init__(self):
        self.completed_tasks: List[str] = []  # 已完成的任务列表
        self.failed_tasks: List[str] = []  # 失败的任务列表
        self.last_updated: Dict[str, float] = {}  # 任务最后更新时间
        self.iteration_count: int = 0  # 当前迭代次数
        self.success_count: int = 0  # 成功任务计数
        self.failure_count: int = 0  # 失败任务计数

    def add_completed_task(self, task: str):
        """Add completed task"""
        if task not in self.completed_tasks:
            self.completed_tasks.append(task)
            self.last_updated[task] = time.time()
            self.success_count += 1  # 增加成功计数

        if task in self.failed_tasks:
            self.failed_tasks.remove(task)

    def add_failed_task(self, task: str):
        """Add failed task"""
        if task not in self.failed_tasks and task not in self.completed_tasks:
            self.failed_tasks.append(task)
            self.last_updated[task] = time.time()
            self.failure_count += 1  # 增加失败计数

    def get_status(self, task: str) -> Dict:
        """Get task status"""
        return {
            "completed": task in self.completed_tasks,
            "failed": task in self.failed_tasks,
            "last_updated": self.last_updated.get(task)
        }

    def increment_iteration(self):
        """Increment iteration count"""
        self.iteration_count += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total_tasks = self.success_count + self.failure_count
        if total_tasks == 0:
            return 0.0
        return self.success_count / total_tasks

    @property
    def progress(self) -> int:
        """Get overall progress (number of completed tasks)"""
        return len(self.completed_tasks)

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "last_updated": self.last_updated,
            "iteration_count": self.iteration_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskProgress':
        """Create instance from dictionary"""
        progress = cls()
        progress.completed_tasks = data.get("completed_tasks", [])
        progress.failed_tasks = data.get("failed_tasks", [])
        progress.last_updated = data.get("last_updated", {})
        progress.iteration_count = data.get("iteration_count", 0)
        progress.success_count = data.get("success_count", 0)
        progress.failure_count = data.get("failure_count", 0)
        return progress