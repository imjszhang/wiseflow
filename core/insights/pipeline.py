from typing import Dict
from utils.general_utils import compare_phrase_with_list
from .get_insight import pb, logger, get_insights, insight_rewrite
from scrapers.general_crawler import general_crawler
from datetime import datetime, timedelta
from config import CONFIG
import os
import json

# 项目目录
project_dir = CONFIG["PROJECT_DIR"]

# 文件扩展名列表（用于判断是否为文件）
extensions = ('.pdf', '.docx', '.xlsx', '.doc', '.ppt', '.pptx', '.xls', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.bmp',
              '.tiff', '.mp4', '.avi', '.wmv', '.mkv', '.flv', '.wav', '.mp3', '.avi', '.mov', '.wmv', '.mpeg', '.mpg',
              '.3gp', '.ogg', '.webm', '.m4a', '.aac', '.flac', '.wma', '.amr', '.ogg', '.m4v', '.m3u8', '.m3u', '.ts',
              '.mts')

# 文章过期天数
expiration_days = 10

# 已存在的 URL 集合
existing_urls = {url['url'] for url in pb.read(collection_name='articles', fields=['url']) if url['url']}

async def pipeline(url: str, cache: Dict[str, str] = {}):
    """
    爬取和处理文章的主流程。

    :param url: str - 目标 URL
    :param cache: dict - 缓存数据，用于补充文章信息
    """
    working_list = {url}  # 待处理的 URL 列表
    while working_list:
        url = working_list.pop()
        existing_urls.add(url)

        # 跳过文件类型的 URL
        if any(url.endswith(ext) for ext in extensions):
            logger.info(f"{url} is a file, skipping...")
            continue

        logger.debug(f"Start processing {url}")

        # 获取文章内容
        flag, result = await general_crawler(url, logger)
        if flag == 1:
            # 如果返回的是 URL 列表，将新 URL 添加到待处理列表
            logger.info('Received new URL list, adding to work list...')
            new_urls = result - existing_urls
            working_list.update(new_urls)
            continue
        elif flag <= 0:
            logger.error("Failed to fetch article, aborting pipeline...")
            continue

        # 检查文章是否过期
        expiration = datetime.now() - timedelta(days=expiration_days)
        expiration_date = expiration.strftime('%Y-%m-%d')
        article_date = int(result['publish_time'])
        if article_date < int(expiration_date.replace('-', '')):
            logger.info(f"Publish date is {article_date}, too old, skipping...")
            continue

        # 将缓存数据补充到文章结果中
        for k, v in cache.items():
            if v:
                result[k] = v

        # 添加文章到数据库
        logger.debug(f"Article: {result['title']}")
        article_id = pb.add(collection_name='articles', body=result)
        if not article_id:
            logger.error('Failed to add article, writing to cache file...')
            with open(os.path.join(project_dir, 'cache_articles.json'), 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            continue

        # 提取洞察信息
        insights = get_insights(f"title: {result['title']}\n\ncontent: {result['content']}")
        if not insights:
            continue

        # 洞察后处理
        article_tags = set()
        old_insights = pb.read(collection_name='insights', filter=f"updated>'{expiration_date}'",
                               fields=['id', 'tag', 'content', 'articles'])
        for insight in insights:
            article_tags.add(insight['tag'])
            insight['articles'] = [article_id]
            old_insight_dict = {i['content']: i for i in old_insights if i['tag'] == insight['tag']}

            # 检查是否有相似的洞察
            similar_insights = compare_phrase_with_list(insight['content'], list(old_insight_dict.keys()), 0.65)
            if similar_insights:
                # 合并相似洞察并重写内容
                to_rewrite = similar_insights + [insight['content']]
                new_insight_content = insight_rewrite(to_rewrite)
                if not new_insight_content:
                    continue
                insight['content'] = new_insight_content

                # 合并相关文章并删除旧洞察
                for old_insight in similar_insights:
                    insight['articles'].extend(old_insight_dict[old_insight]['articles'])
                    if not pb.delete(collection_name='insights', id=old_insight_dict[old_insight]['id']):
                        logger.error('Failed to delete old insight')
                    old_insights.remove(old_insight_dict[old_insight])

            # 添加新的洞察到数据库
            insight['id'] = pb.add(collection_name='insights', body=insight)
            if not insight['id']:
                logger.error('Failed to add insight, writing to cache file...')
                with open(os.path.join(project_dir, 'cache_insights.json'), 'a', encoding='utf-8') as f:
                    json.dump(insight, f, ensure_ascii=False, indent=4)

        # 更新文章的标签信息
        update_result = pb.update(collection_name='articles', id=article_id, body={'tag': list(article_tags)})
        if not update_result:
            logger.error(f'Failed to update article - article_id: {article_id}')
            result['tag'] = list(article_tags)
            with open(os.path.join(project_dir, 'cache_articles.json'), 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)