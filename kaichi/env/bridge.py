#Filename: project_env.py
import gymnasium as gym
from gymnasium.core import ObsType
from typing import SupportsFloat, Any, Tuple, Dict
import subprocess
import os
import tempfile
import json

class ProjectEnv(gym.Env):
    def __init__(self, timeout: int = 5, log_path: str = "./logs"):
        """
        初始化虚拟 Python 运行环境。

        Args:
            timeout (int): 每段代码的最大执行时间（秒）。
            log_path (str): 日志文件存储路径。
        """
        self.timeout = timeout
        self.log_path = log_path
        self.has_reset = False
        self.temp_dir = None
        self.execution_log = []

        # 创建日志目录
        os.makedirs(self.log_path, exist_ok=True)

    def reset(self, *, seed=None, options=None) -> Tuple[ObsType, Dict[str, Any]]:
        """
        重置环境，清空临时目录和日志。

        Returns:
            Tuple: 初始状态和额外信息。
        """
        # 清理之前的临时目录
        if self.temp_dir:
            self._cleanup_temp_dir()

        # 创建新的临时目录
        self.temp_dir = tempfile.mkdtemp()

        # 清空执行日志
        self.execution_log = []

        self.has_reset = True
        return {"status": "ready", "temp_dir": self.temp_dir}, {}

    def step(self, code: str) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        执行一段 Python 代码。

        Args:
            code (str): 要执行的 Python 代码。

        Returns:
            Tuple: (状态, 奖励, 是否结束, 是否截断, 额外信息)。
        """
        if not self.has_reset:
            raise RuntimeError("Environment has not been reset yet.")

        # 创建一个临时文件来存储代码
        code_file = os.path.join(self.temp_dir, "script.py")
        with open(code_file, "w") as f:
            f.write(code)

        # 执行代码并捕获输出
        try:
            result = subprocess.run(
                ["python", code_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.temp_dir,
            )
            # 记录执行结果
            output = result.stdout
            error = result.stderr
            return_code = result.returncode

            # 更新执行日志
            self.execution_log.append({
                "code": code,
                "output": output,
                "error": error,
                "return_code": return_code,
            })

            # 奖励机制：如果代码执行成功（return_code == 0），给予正奖励；否则给予负奖励
            reward = 1.0 if return_code == 0 else -1.0

            # 是否结束：可以根据某些条件定义结束逻辑，这里简单地设置为永不结束
            done = False

            # 返回状态和信息
            state = {
                "output": output,
                "error": error,
                "return_code": return_code,
            }
            info = {
                "log": self.execution_log,
            }
            return state, reward, done, False, info

        except subprocess.TimeoutExpired:
            # 如果代码执行超时
            error_message = f"Code execution exceeded timeout of {self.timeout} seconds."
            self.execution_log.append({
                "code": code,
                "output": "",
                "error": error_message,
                "return_code": -1,
            })
            state = {
                "output": "",
                "error": error_message,
                "return_code": -1,
            }
            reward = -1.0  # 超时给予负奖励
            done = False
            info = {
                "log": self.execution_log,
            }
            return state, reward, done, False, info

    def render(self):
        """
        渲染环境状态（可选）。
        """
        print("Execution Log:")
        for entry in self.execution_log:
            print(json.dumps(entry, indent=2))

    def close(self):
        """
        关闭环境，清理资源。
        """
        self._cleanup_temp_dir()
        self.has_reset = False

    def _cleanup_temp_dir(self):
        """
        清理临时目录。
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)
            self.temp_dir = None