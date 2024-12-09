from llms.openai_wrapper import openai_llm
from llms.dify_wrapper import dify_llm
import re
import os
from loguru import logger
from utils.general_utils import get_logger_level, is_chinese
from utils.pb_api import PbTalker
from utils.prompt_utils import load_prompt_template
from config import CONFIG

get_info_model = os.environ.get("GET_INFO_MODEL", "gpt-4o-mini-2024-07-18")
rewrite_model = os.environ.get("REWRITE_MODEL", "gpt-4o-mini-2024-07-18")

project_dir = CONFIG["PROJECT_DIR"]
if project_dir:
    os.makedirs(project_dir, exist_ok=True)
logger_file = os.path.join(project_dir, 'wiseflow.log')
dsw_log = get_logger_level()
logger.add(
    logger_file,
    level=dsw_log,
    backtrace=True,
    diagnose=True,
    rotation="50 MB"
)

pb = PbTalker(logger)

focus_data = pb.read(collection_name='tags', filter=f'activated=True')
if not focus_data:
    logger.error('no activated tag found, please set at least one')
    exit(1)

focus_list = [item["name"] for item in focus_data if item["name"]]
focus_dict = {item["name"]: item["id"] for item in focus_data if item["name"]}
lang_term = ''.join([f'{item["name"]}{item["explaination"]}' for item in focus_data if item["name"]])
focus_statement = '\n'.join([f'<tag>{item["name"]}</tag>{item["explaination"]}' for item in focus_data if item["name"] and item["explaination"]])

# 定义模板文件路径
PROMPT_DIR = "prompts"
PROMPT_DIR = os.path.join(PROMPT_DIR, "insights")
language = "zh" if is_chinese(lang_term) else "zh"
PROMPT_DIR = os.path.join(PROMPT_DIR, language)

SYSTEM_PROMPT_FILE = os.path.join(PROMPT_DIR, "system_prompt.txt")
REWRITE_PROMPT_FILE = os.path.join(PROMPT_DIR, "rewrite_prompt.txt")

# 加载系统提示词
try:
    system_prompt = load_prompt_template(
        SYSTEM_PROMPT_FILE,
        focus_list="\n".join(focus_list),
        focus_statement=focus_statement if focus_statement else "",
    )
    rewrite_prompt = load_prompt_template(REWRITE_PROMPT_FILE)
except FileNotFoundError as e:
    logger.error(f"Failed to load prompt templates: {e}")
    exit(1)


# 增加 LLM 选择逻辑
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "dify").lower()  # 默认使用 dify

def select_llm():
    """
    根据环境变量选择 LLM 提供商。
    """
    if LLM_PROVIDER == "openai":
        return openai_llm
    elif LLM_PROVIDER == "dify":
        return dify_llm
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

# 动态选择 LLM
llm = select_llm()

def get_insights(article_content: str) -> list[dict]:
    """
    从文章内容中提取与用户关注的标签相关的洞察信息。

    :param article_content: 文章内容
    :return: 包含提取信息的列表，每个元素是一个字典，包含 'content' 和 'tag'
    """
    inputs = {'system': system_prompt}
    try:
        # 调用 LLM 提取信息
        if LLM_PROVIDER == "openai":
            result = llm(
                [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': article_content}],
                model=get_info_model,
                logger=logger,
                temperature=0.1
            )
        elif LLM_PROVIDER == "dify":
            response = llm(article_content, 'wiseflow', inputs=inputs, logger=logger)
            result = response['answer']
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
    except Exception as e:
        logger.error(f"Error during LLM call: {e}")
        return []

    # 解析 LLM 输出
    insights = parse_llm_output(result, article_content)
    return insights


def parse_llm_output(result: str, article_content: str) -> list[dict]:
    """
    解析 LLM 的输出，提取标签和相关信息。

    :param result: LLM 的输出结果
    :param article_content: 原始文章内容，用于提取来源信息
    :return: 包含提取信息的列表，每个元素是一个字典，包含 'content' 和 'tag'
    """
    # 分割并过滤 LLM 输出
    texts = [text.strip() for text in result.split('<tag>') if '</tag>' in text.strip()]
    if not texts:
        logger.debug(f'Cannot find info, LLM result:\n{result}')
        return []

    insights = []
    for text in texts:
        tag, info = extract_tag_and_info(text)
        if not tag or not info:
            continue

        # 添加来源信息
        info = append_source_info(info, article_content)

        # 构造结果
        insights.append({'content': info, 'tag': focus_dict[tag]})

    return insights


def extract_tag_and_info(text: str) -> tuple[str, str]:
    """
    从文本块中提取标签和信息。

    :param text: 文本块，格式为 "<tag>标签</tag>信息"
    :return: 标签和信息的元组
    """
    try:
        tag, info = text.split('</tag>', 1)
        tag = tag.strip()
        info = info.split('\n\n')[0].strip()

        # 验证标签和信息
        if tag not in focus_list:
            logger.info(f'Tag not in focus_list: {tag}, skipping')
            return '', ''
        if len(info) < 7 or info.startswith(('无相关信息', '该新闻未提及', '未提及')):
            logger.info(f'Invalid or irrelevant info: {info}')
            return '', ''

        # 去掉多余的引号
        info = info.rstrip('"').strip()
        return tag, info
    except Exception as e:
        logger.info(f'Error parsing text block: {e}')
        return '', ''


def append_source_info(info: str, article_content: str) -> str:
    """
    为信息添加来源信息。

    :param info: 提取的信息
    :param article_content: 原始文章内容，用于提取来源
    :return: 添加来源信息后的信息
    """
    sources = re.findall(r'\[from (.*?)]', article_content)
    if sources and sources[0]:
        return f"[from {sources[0]}] {info}"
    return info

# insight_rewrite 函数
def insight_rewrite(contents: list[str]) -> str:
    context = f"<content>{'</content><content>'.join(contents)}</content>"
    try:
        if LLM_PROVIDER == "openai":
            # OpenAI LLM 调用
            result = llm(
                [{'role': 'system', 'content': rewrite_prompt}, {'role': 'user', 'content': context}],
                model=rewrite_model,
                temperature=0.1,
                logger=logger
            )
        elif LLM_PROVIDER == "dify":
            # Dify LLM 调用
            inputs = {'system': rewrite_prompt}
            response = llm(context, 'wiseflow', inputs=inputs, logger=logger)
            result = response['answer']
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
        return result.strip()
    except Exception as e:
        logger.warning(f"Rewrite process LLM generate failed: {e}")
        return ''
