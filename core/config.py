import os

# 配置项
CONFIG = {
    "GET_INFO_MODEL": os.environ.get("GET_INFO_MODEL", "gpt-4o-mini-2024-07-18"),
    "REWRITE_MODEL": os.environ.get("REWRITE_MODEL", "gpt-4o-mini-2024-07-18"),
    "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "dify").lower(),
    "PROJECT_DIR": os.environ.get("PROJECT_DIR", ""),
    "PROMPT_DIR": "prompts",
    "LANGUAGE": "zh"  # 默认语言，可动态检测
}