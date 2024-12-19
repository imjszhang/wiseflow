from .get_insight import logger, pb
from .pipeline import pipeline
from utils.general_utils import extract_urls
import re
import asyncio

# 正则表达式模式
item_pattern = re.compile(r'<item>(.*?)</item>', re.DOTALL)  # 匹配 <item> 标签内容
url_pattern = re.compile(r'<url><!\[CDATA\[(.*?)]]></url>')  # 匹配 <url> 标签内容
summary_pattern = re.compile(r'<summary><!\[CDATA\[(.*?)]]></summary>', re.DOTALL)  # 匹配 <summary> 标签内容

# 获取已存在的 URL 集合
existing_urls = {url['url'] for url in pb.read(collection_name='articles', fields=['url']) if url['url']}

async def message_manager(_input: dict):
    """
    消息管理器，根据输入类型处理不同的任务。

    :param _input: dict - 输入数据，包含以下字段：
        - user_id: str - 用户 ID
        - type: str - 消息类型（publicMsg, text, url）
        - content: str - 消息内容
        - addition: str - 附加信息（可选）
    """
    source = _input['user_id']
    logger.debug(f"Received new task, user: {source}, Addition info: {_input.get('addition', '')}")

    if _input['type'] == 'publicMsg':
        # 处理 publicMsg 类型的消息
        items = item_pattern.findall(_input["content"])
        # 遍历所有 <item> 标签内容，提取 <url> 和 <summary>
        for item in items:
            url_match = url_pattern.search(item)
            url = url_match.group(1) if url_match else None
            if not url:
                logger.warning(f"Cannot find URL in \n{item}")
                continue

            # URL 处理：将 http 替换为 https，并移除 chksm 参数后的部分
            url = url.replace('http://', 'https://')
            cut_off_point = url.find('chksm=')
            if cut_off_point != -1:
                url = url[:cut_off_point - 1]

            if url in existing_urls:
                logger.debug(f"{url} has been crawled, skipping...")
                continue

            summary_match = summary_pattern.search(item)
            summary = summary_match.group(1) if summary_match else None
            cache = {'source': source, 'abstract': summary}
            await pipeline(url, cache)

    elif _input['type'] == 'text':
        # 处理 text 类型的消息
        urls = extract_urls(_input['content'])
        if not urls:
            logger.debug(f"Cannot find any URL in\n{_input['content']}\nSkipping...")
            # TODO: 添加从文本中提取信息的处理逻辑
            return

        # 并发处理所有未爬取的 URL
        await asyncio.gather(*[pipeline(url) for url in urls if url not in existing_urls])

    elif _input['type'] == 'url':
        # 处理 url 类型的消息（主要用于微信分享的 mp_article_card）
        item = re.search(r'<url>(.*?)&amp;chksm=', _input["content"], re.DOTALL)
        if not item:
            logger.debug("shareUrlOpen not found")
            item = re.search(r'<shareUrlOriginal>(.*?)&amp;chksm=', _input["content"], re.DOTALL)
            if not item:
                logger.debug("shareUrlOriginal not found")
                item = re.search(r'<shareUrlOpen>(.*?)&amp;chksm=', _input["content"], re.DOTALL)
                if not item:
                    logger.warning(f"Cannot find URL in \n{_input['content']}")
                    return

        # 提取 URL 并移除多余的 amp; 符号
        extract_url = item.group(1).replace('amp;', '')

        # 提取摘要信息
        summary_match = re.search(r'<des>(.*?)</des>', _input["content"], re.DOTALL)
        summary = summary_match.group(1) if summary_match else None
        cache = {'source': source, 'abstract': summary}
        await pipeline(extract_url, cache)

    else:
        # 未知类型，直接返回
        logger.warning(f"Unknown input type: {_input['type']}")
        return