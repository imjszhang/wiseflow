from typing import Dict
from core.utils.general_utils import compare_phrase_with_list
from datetime import datetime, timedelta
from config import CONFIG
import os
import json
from core.scrapers.general_crawler import general_crawler
# 从 __init__.py 导入共享组件
from core.insights import insight_extractor

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
try:
    existing_urls = {
        url['url'] for url in insight_extractor.pb_client.collection('articles').get_full_list(
            query_params={"fields": "url"}
        ) if url.get('url')
    }
except Exception as e:
    insight_extractor.logger.error(f"Failed to fetch existing URLs: {e}")
    existing_urls = set()


class InsightPipeline:
    async def pipeline(self, url: str, cache: Dict[str, str] = {}):
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
                insight_extractor.logger.info(f"{url} is a file, skipping...")
                continue

            insight_extractor.logger.debug(f"Start processing {url}")

            # 获取文章内容
            flag, result = await general_crawler(url, insight_extractor.logger)
            if flag == 1:
                # 如果返回的是 URL 列表，将新 URL 添加到待处理列表
                insight_extractor.logger.info('Received new URL list, adding to work list...')
                new_urls = result - existing_urls
                working_list.update(new_urls)
                continue
            elif flag <= 0:
                insight_extractor.logger.error("Failed to fetch article, aborting pipeline...")
                continue

            # 检查文章是否过期
            expiration = datetime.now() - timedelta(days=expiration_days)
            expiration_date = expiration.strftime('%Y-%m-%d')
            article_date = int(result['publish_time'])
            if article_date < int(expiration_date.replace('-', '')):
                insight_extractor.logger.info(f"Publish date is {article_date}, too old, skipping...")
                continue

            # 将缓存数据补充到文章结果中
            for k, v in cache.items():
                if v:
                    result[k] = v

            # 添加文章到数据库
            insight_extractor.logger.debug(f"Article: {result['title']}")
            try:
                article_record = insight_extractor.pb_client.collection('articles').create(result)
                article_id = article_record.id
            except Exception as e:
                insight_extractor.logger.error(f"Failed to add article: {e}")
                with open(os.path.join(project_dir, 'cache_articles.json'), 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                continue

            # 提取洞察信息
            insights = insight_extractor.get_insights(f"title: {result['title']}\n\ncontent: {result['content']}")
            if not insights:
                continue

            # 洞察后处理
            article_tags = set()
            try:
                old_insights = insight_extractor.pb_client.collection('insights').get_full_list(
                    query_params={
                        "filter": f"updated>'{expiration_date}'",
                        "fields": "id,tag,content,articles"
                    }
                )
                old_insight_dict = {i['content']: i for i in old_insights}
            except Exception as e:
                insight_extractor.logger.error(f"Failed to fetch old insights: {e}")
                old_insight_dict = {}

            for insight in insights:
                article_tags.add(insight['tag'])
                insight['articles'] = [article_id]

                # 检查是否有相似的洞察
                similar_insights = compare_phrase_with_list(insight['content'], list(old_insight_dict.keys()), 0.65)
                if similar_insights:
                    # 合并相似洞察并重写内容
                    to_rewrite = similar_insights + [insight['content']]
                    new_insight_content = insight_extractor.insight_rewrite(to_rewrite)
                    if not new_insight_content:
                        continue
                    insight['content'] = new_insight_content

                    # 合并相关文章并删除旧洞察
                    for old_insight in similar_insights:
                        insight['articles'].extend(old_insight_dict[old_insight]['articles'])
                        try:
                            insight_extractor.pb_client.collection('insights').delete(
                                old_insight_dict[old_insight]['id']
                            )
                        except Exception as e:
                            insight_extractor.logger.error(f"Failed to delete old insight: {e}")
                        old_insights.remove(old_insight_dict[old_insight])

                # 添加新的洞察到数据库
                try:
                    new_insight_record = insight_extractor.pb_client.collection('insights').create(insight)
                    insight['id'] = new_insight_record.id
                except Exception as e:
                    insight_extractor.logger.error(f"Failed to add insight: {e}")
                    with open(os.path.join(project_dir, 'cache_insights.json'), 'a', encoding='utf-8') as f:
                        json.dump(insight, f, ensure_ascii=False, indent=4)

            # 更新文章的标签信息
            try:
                insight_extractor.pb_client.collection('articles').update(
                    article_id,
                    {"tag": list(article_tags)}
                )
            except Exception as e:
                insight_extractor.logger.error(f"Failed to update article - article_id: {article_id}, error: {e}")
                result['tag'] = list(article_tags)
                with open(os.path.join(project_dir, 'cache_articles.json'), 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)


# 创建单例实例
pipeline = InsightPipeline()

# 明确声明模块级别的导出
__all__ = ['pipeline', 'InsightPipeline']