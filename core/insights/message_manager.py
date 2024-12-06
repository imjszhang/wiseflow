from .get_info import logger,pb
from .pipeline import pipeline
from utils.general_utils import extract_urls
import re
import asyncio


item_pattern = re.compile(r'<item>(.*?)</item>', re.DOTALL)
url_pattern = re.compile(r'<url><!\[CDATA\[(.*?)]]></url>')
summary_pattern = re.compile(r'<summary><!\[CDATA\[(.*?)]]></summary>', re.DOTALL)
existing_urls = {url['url'] for url in pb.read(collection_name='articles', fields=['url']) if url['url']}

async def message_manager(_input: dict):
    source = _input['user_id']
    logger.debug(f"received new task, user: {source}, Addition info: {_input['addition']}")
    if _input['type'] == 'publicMsg':
        items = item_pattern.findall(_input["content"])
        # Iterate through all < item > content, extracting < url > and < summary >
        for item in items:
            url_match = url_pattern.search(item)
            url = url_match.group(1) if url_match else None
            if not url:
                logger.warning(f"can not find url in \n{item}")
                continue
            # URL processing, http is replaced by https, and the part after chksm is removed.
            url = url.replace('http://', 'https://')
            cut_off_point = url.find('chksm=')
            if cut_off_point != -1:
                url = url[:cut_off_point-1]
            if url in existing_urls:
                logger.debug(f"{url} has been crawled, skip")
                continue
            summary_match = summary_pattern.search(item)
            summary = summary_match.group(1) if summary_match else None
            cache = {'source': source, 'abstract': summary}
            await pipeline(url, cache)

    elif _input['type'] == 'text':
        urls = extract_urls(_input['content'])
        if not urls:
            logger.debug(f"can not find any url in\n{_input['content']}\npass...")
            # todo get info from text process
            return
        await asyncio.gather(*[pipeline(url) for url in urls if url not in existing_urls])

    elif _input['type'] == 'url':
        # this is remained for wechat shared mp_article_card
        item = re.search(r'<url>(.*?)&amp;chksm=', _input["content"], re.DOTALL)
        if not item:
            logger.debug("shareUrlOpen not find")
            item = re.search(r'<shareUrlOriginal>(.*?)&amp;chksm=', _input["content"], re.DOTALL)
            if not item:
                logger.debug("shareUrlOriginal not find")
                item = re.search(r'<shareUrlOpen>(.*?)&amp;chksm=', _input["content"], re.DOTALL)
                if not item:
                    logger.warning(f"cannot find url in \n{_input['content']}")
                    return
        extract_url = item.group(1).replace('amp;', '')
        summary_match = re.search(r'<des>(.*?)</des>', _input["content"], re.DOTALL)
        summary = summary_match.group(1) if summary_match else None
        cache = {'source': source, 'abstract': summary}
        await pipeline(extract_url, cache)
    else:
        return