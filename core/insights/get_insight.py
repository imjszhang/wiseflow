from llms.openai_wrapper import openai_llm
from llms.dify_wrapper import dify_llm
import re
import os
from loguru import logger
from utils.general_utils import get_logger_level, is_chinese
from utils.pb_api import PbTalker
from utils.prompt_utils import load_prompt_template

get_info_model = os.environ.get("GET_INFO_MODEL", "gpt-4o-mini-2024-07-18")
rewrite_model = os.environ.get("REWRITE_MODEL", "gpt-4o-mini-2024-07-18")

project_dir = os.environ.get("PROJECT_DIR", "")
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
language = "zh" if is_chinese(lang_term) else "en"
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

# get_insights 函数
def get_insights(article_content: str) -> list[dict]:
    inputs = {'system': system_prompt}
    try:
        if LLM_PROVIDER == "openai":
            # OpenAI LLM 调用
            result = llm(
                [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': article_content}],
                model=get_info_model,
                logger=logger,
                temperature=0.1
            )
        elif LLM_PROVIDER == "dify":
            # Dify LLM 调用
            response = llm(article_content, 'wiseflow', inputs=inputs, logger=logger)
            result = response['answer']
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
    except Exception as e:
        logger.error(f"Error during LLM call: {e}")
        return []

    # 解析结果
    texts = result.split('<tag>')
    texts = [_.strip() for _ in texts if '</tag>' in _.strip()]
    if not texts:
        logger.debug(f'Cannot find info, LLM result:\n{result}')
        return []

    cache = []
    for text in texts:
        try:
            strings = text.split('</tag>')
            tag = strings[0].strip()
            if tag not in focus_list:
                logger.info(f'Tag not in focus_list: {tag}, aborting')
                continue
            info = strings[1].split('\n\n')[0].strip()
        except Exception as e:
            logger.info(f'Parse error: {e}')
            tag = ''
            info = ''

        if not info or not tag:
            logger.info(f'Parse failed-{text}')
            continue

        if len(info) < 7:
            logger.info(f'Info too short, possible invalid: {info}')
            continue

        if info.startswith('无相关信息') or info.startswith('该新闻未提及') or info.startswith('未提及'):
            logger.info(f'No relevant info: {text}')
            continue

        while info.endswith('"'):
            info = info[:-1].strip()

        # 拼接来源信息
        sources = re.findall(r'\[from (.*?)]', article_content)
        if sources and sources[0]:
            info = f"[from {sources[0]}] {info}"

        cache.append({'content': info, 'tag': focus_dict[tag]})

    return cache

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
