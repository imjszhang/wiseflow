import os

# 配置项
CONFIG = {
    "GET_INFO_MODEL": os.environ.get("GET_INFO_MODEL", "gpt-4o-mini-2024-07-18"),
    "REWRITE_MODEL": os.environ.get("REWRITE_MODEL", "gpt-4o-mini-2024-07-18"),
    "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "dify").lower(),
    "PROJECT_DIR": os.environ.get("PROJECT_DIR", ""),
    "POCKETBASE_URL": "http://127.0.0.1:8091",#os.environ.get("PB_API_BASE", ""),
    "POCKETBASE_ADMIN_EMAIL": "test@example.com",#os.environ.get("PB_EMAIL", ""),
    "POCKETBASE_ADMIN_PASSWORD": "1234567890",#os.environ.get("PB_PASSWORD", ""),
    "PROMPT_DIR": "core/prompts",
    "LANGUAGE": "zh"  # 默认语言，可动态检测
}

# 导出配置项
__all__ = ['CONFIG']