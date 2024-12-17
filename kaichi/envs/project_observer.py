import os
import json
from typing import Dict, List, Optional


class ProjectObserver:
    """观察项目目录，提取信息并保存到目标目录"""
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = source_dir  # 源目录
        self.target_dir = target_dir  # 目标目录

        # 确保目标目录存在
        os.makedirs(self.target_dir, exist_ok=True)

    def observe_project(self) -> Dict:
        """
        提取项目的观察信息。
        :return: 提取的信息字典
        """
        try:
            project_info = {
                "directory_structure": self._get_directory_structure(),
                "key_files": self._extract_key_files(),
                "meta": self._get_project_meta(),
                "log_summary": self._summarize_logs(),
                "code_statistics": self._analyze_code()
            }
            return project_info
        except Exception as e:
            print(f"Failed to observe project: {e}")
            return {}

    def _get_directory_structure(self) -> List[str]:
        """
        获取项目目录的结构。
        :return: 目录结构列表
        """
        structure = []
        for root, dirs, files in os.walk(self.source_dir):
            for name in files:
                structure.append(os.path.relpath(os.path.join(root, name), self.source_dir))
        return structure

    def _extract_key_files(self) -> Dict[str, Optional[str]]:
        """
        提取关键文件的内容（如 README、配置文件等）。
        :return: 关键文件内容字典
        """
        key_files = ["README.md", "config.yaml"]
        extracted_files = {}
        for key_file in key_files:
            file_path = os.path.join(self.source_dir, key_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        extracted_files[key_file] = f.read()
                except Exception as e:
                    print(f"Failed to read key file {key_file}: {e}")
                    extracted_files[key_file] = None
            else:
                extracted_files[key_file] = None
        return extracted_files

    def _get_project_meta(self) -> Dict:
        """
        获取项目的元信息（如文件数量、总大小等）。
        :return: 元信息字典
        """
        total_size = 0
        file_count = 0
        dir_count = 0
        for root, dirs, files in os.walk(self.source_dir):
            dir_count += len(dirs)
            file_count += len(files)
            for name in files:
                file_path = os.path.join(root, name)
                total_size += os.path.getsize(file_path)
        return {
            "file_count": file_count,
            "dir_count": dir_count,
            "total_size": total_size
        }

    def _summarize_logs(self) -> List[str]:
        """
        提取日志文件的摘要。
        :return: 日志摘要列表
        """
        log_dir = os.path.join(self.source_dir, "logs")
        summaries = []
        if os.path.exists(log_dir):
            for log_file in os.listdir(log_dir):
                log_path = os.path.join(log_dir, log_file)
                if os.path.isfile(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            summaries.append(f"{log_file}: {lines[:5]}")  # 提取前 5 行作为摘要
                    except Exception as e:
                        print(f"Failed to read log file {log_file}: {e}")
        return summaries

    def _analyze_code(self) -> Dict:
        """
        分析代码文件，统计代码行数和文件类型分布。
        :return: 代码统计信息字典
        """
        code_stats = {"total_lines": 0, "file_types": {}}
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith((".py", ".js", ".java", ".cpp")):  # 支持的代码文件类型
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            code_stats["total_lines"] += len(lines)
                            ext = os.path.splitext(file)[1]
                            code_stats["file_types"][ext] = code_stats["file_types"].get(ext, 0) + 1
                    except Exception as e:
                        print(f"Failed to analyze code file {file}: {e}")
        return code_stats

    def save_observation(self, observation: Dict):
        """
        保存观察信息到目标目录。
        :param observation: 观察信息字典
        """
        try:
            file_path = os.path.join(self.target_dir, "project_observation.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(observation, f, indent=4)
        except Exception as e:
            print(f"Failed to save observation: {e}")