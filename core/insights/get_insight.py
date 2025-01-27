import os
import re
from typing import List, Dict
from loguru import logger
from core.llms.openai_wrapper import openai_llm  
from core.llms.dify_wrapper import dify_llm
from core.utils.general_utils import get_logger_level, is_chinese
from core.utils.prompt_utils import load_prompt_template
from config import CONFIG
from pocketbase import PocketBase
from pocketbase.client import ClientResponseError


class InsightExtractor:
    def __init__(self):
        # 初始化配置
        self.project_dir = CONFIG["PROJECT_DIR"]
        self.get_info_model = os.environ.get("GET_INFO_MODEL", "gpt-4o-mini-2024-07-18")
        self.rewrite_model = os.environ.get("REWRITE_MODEL", "gpt-4o-mini-2024-07-18")
        self.llm_provider = os.environ.get("LLM_PROVIDER", "dify").lower()  # 默认使用 dify
        self.logger = logger

        # 初始化日志
        self._init_logger()

        # 初始化 PocketBase 客户端
        pocketbase_url = CONFIG.get("POCKETBASE_URL")
        if not pocketbase_url:
            logger.error("POCKETBASE_URL is not defined in the configuration.")
            exit(1)
        self.pb_client = PocketBase(pocketbase_url)
        self._authenticate_pocketbase()

        # 加载激活标签
        self.focus_data = self._load_focus_data()
        self.focus_list, self.focus_dict, self.focus_statement = self._extract_focus_info()

        # 加载提示词模板
        self.system_prompt, self.rewrite_prompt = self._load_prompts()

        # 动态选择 LLM 提供商
        self.llm = self._select_llm()

    def _init_logger(self):
        """初始化日志系统"""
        if self.project_dir:
            os.makedirs(self.project_dir, exist_ok=True)
        logger_file = os.path.join(self.project_dir, 'wiseflow.log')
        dsw_log = get_logger_level()
        self.logger.add(
            logger_file,
            level=dsw_log,
            backtrace=True,
            diagnose=True,
            rotation="50 MB"
        )

    def _authenticate_pocketbase(self):
        """对 PocketBase 进行身份验证（如果需要）"""
        try:
            admin_email = CONFIG.get("POCKETBASE_ADMIN_EMAIL")
            admin_password = CONFIG.get("POCKETBASE_ADMIN_PASSWORD")
            if admin_email and admin_password:
                self.pb_client.admins.auth_with_password(admin_email, admin_password)
                self.logger.info("Authenticated with PocketBase successfully.")
            else:
                self.logger.warning("PocketBase admin credentials are not provided. Proceeding without authentication.")
        except ClientResponseError as e:
            error_msg = getattr(e, 'data', {}).get('message', str(e))
            self.logger.error(f"Failed to authenticate with PocketBase: {error_msg}")
            self.logger.warning("Continuing without authentication...")
        except Exception as e:
            self.logger.error(f"Unexpected error during PocketBase authentication: {str(e)}")
            raise

    def _load_focus_data(self) -> List[Dict]:
        """从 PocketBase 数据库中加载激活的标签"""
        try:
            focus_data = []
            page = 1
            per_page = 50  # 每页记录数
            while True:
                response = self.pb_client.collection('tags').get_list(
                    page=page,
                    per_page=per_page,
                    query_params={
                        "filter": "activated=True"
                    }
                )
                focus_data.extend(response.items)
                if page >= response.total_pages:
                    break
                page += 1
            if not focus_data:
                self.logger.error('No activated tag found, please set at least one.')
                exit(1)
            return focus_data
        except ClientResponseError as e:
            self.logger.error(f"Failed to load focus data from PocketBase: {e.message}")
            exit(1)

    def _extract_focus_info(self):
        """提取标签和说明信息"""
        focus_list = [item.name for item in self.focus_data]
        focus_dict = {item.name: item.id for item in self.focus_data}
        focus_statement = '\n'.join(
            [f'<tag>{item.name}</tag>\n{getattr(item, "explaination", "")}\n' 
             for item in self.focus_data]
        )
        return focus_list, focus_dict, focus_statement

    def _load_prompts(self):
        """加载提示词模板"""
        PROMPT_DIR = os.path.join("prompts", "insights")
        language = "zh"  # 默认语言为中文
        PROMPT_DIR = os.path.join(PROMPT_DIR, language)

        SYSTEM_PROMPT_FILE = os.path.join(PROMPT_DIR, "system_prompt.txt")
        REWRITE_PROMPT_FILE = os.path.join(PROMPT_DIR, "rewrite_prompt.txt")

        try:
            system_prompt = load_prompt_template(
                SYSTEM_PROMPT_FILE,
                focus_list="\n".join(self.focus_list),
                focus_statement=self.focus_statement if self.focus_statement else "",
            )
            rewrite_prompt = load_prompt_template(REWRITE_PROMPT_FILE)
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load prompt templates: {e}")
            exit(1)

        return system_prompt, rewrite_prompt

    def _select_llm(self):
        """动态选择 LLM 提供商"""
        if self.llm_provider == "openai":
            return openai_llm
        elif self.llm_provider == "dify":
            return dify_llm
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def get_insights(self, article_content: str) -> List[Dict]:
        """
        从文章内容中提取与用户关注的标签相关的洞察信息。

        :param article_content: str - 文章内容
        :return: list[dict] - 包含提取信息的列表，每个元素是一个字典，包含 'content' 和 'tag'
        """
        inputs = {'system': self.system_prompt}
        try:
            # 调用 LLM 提取信息
            if self.llm_provider == "openai":
                result = self.llm(
                    [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': article_content}],
                    model=self.get_info_model,
                    logger=self.logger,
                    temperature=0.1
                )
            elif self.llm_provider == "dify":
                response = self.llm(article_content, 'wiseflow', inputs=inputs, logger=self.logger)
                result = response['answer']
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        except Exception as e:
            self.logger.error(f"Error during LLM call: {e}")
            return []

        # 解析 LLM 输出
        return self._parse_llm_output(result, article_content)

    def _parse_llm_output(self, result: str, article_content: str) -> List[Dict]:
        """解析 LLM 的输出，提取标签和相关信息"""
        texts = [text.strip() for text in result.split('<tag>') if '</tag>' in text.strip()]
        if not texts:
            self.logger.debug(f'Cannot find info, LLM result:\n{result}')
            return []

        insights = []
        for text in texts:
            tag, info = self._extract_tag_and_info(text)
            if not tag or not info:
                continue

            # 添加来源信息
            info = self._append_source_info(info, article_content)

            # 构造结果
            insights.append({'content': info, 'tag': self.focus_dict[tag]})

        return insights

    def _extract_tag_and_info(self, text: str) -> tuple:
        """从文本块中提取标签和信息"""
        try:
            tag, info = text.split('</tag>', 1)
            tag = tag.strip()
            info = info.split('\n\n')[0].strip()

            # 验证标签和信息
            if tag not in self.focus_list:
                self.logger.info(f'Tag not in focus_list: {tag}, skipping')
                return '', ''
            if len(info) < 7 or info.startswith(('无相关信息', '该新闻未提及', '未提及')):
                self.logger.info(f'Invalid or irrelevant info: {info}')
                return '', ''

            # 去掉多余的引号
            info = info.rstrip('"').strip()
            return tag, info
        except Exception as e:
            self.logger.info(f'Error parsing text block: {e}')
            return '', ''

    def _append_source_info(self, info: str, article_content: str) -> str:
        """为信息添加来源信息"""
        source = next(iter(re.findall(r'\[from (.*?)]', article_content)), None)
        if source:
            return f"[from {source}] {info}"
        return info

    def insight_rewrite(self, contents: List[str]) -> str:
        """
        重写洞察内容。

        :param contents: list[str] - 洞察内容列表
        :return: str - 重写后的洞察内容
        """
        context = f"<content>{'</content><content>'.join(contents)}</content>"
        try:
            if self.llm_provider == "openai":
                result = self.llm(
                    [{'role': 'system', 'content': self.rewrite_prompt}, {'role': 'user', 'content': context}],
                    model=self.rewrite_model,
                    temperature=0.1,
                    logger=self.logger
                )
            elif self.llm_provider == "dify":
                inputs = {'system': self.rewrite_prompt}
                response = self.llm(context, 'wiseflow', inputs=inputs, logger=self.logger)
                result = response['answer']
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
            return result.strip()
        except Exception as e:
            self.logger.warning(f"Rewrite process LLM generate failed: {e}")
            return ''


# 实例化 InsightExtractor 类
insight_extractor = InsightExtractor()

# 明确声明模块级别的导出
__all__ = ['InsightExtractor', 'insight_extractor']