import asyncio
from core.utils.pb_api import PbTalker
from loguru import logger
import os
from core.utils.general_utils import get_logger_level
from dotenv import load_dotenv
import httpx

# 加载 .env 文件
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.env')
load_dotenv(dotenv_path=env_path)

# 初始化日志
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

# 初始化 PbTalker
pb = PbTalker(logger)

# /report 接口的 URL
REPORT_API_URL = "http://localhost:8077/report"  # 替换为实际的接口地址

# 延迟时间（秒）
REQUEST_DELAY = 3  # 每次请求之间的延迟时间
MAX_RETRIES = 1  # 最大重试次数


async def fetch_insight_ids_by_tags(tags):
    """
    根据标签从 Pb 数据库中查询 insight_id 列表。

    :param tags: List[str] - 用户选择的标签列表
    :return: List[str] - 符合条件的 insight_id 列表
    """
    if not tags:
        print("Error: 标签列表不能为空！")
        return []

    # 从 tags 表中查询符合条件的 tag_id 列表
    tag_filter = " || ".join([f'name="{tag}"' for tag in tags])
    tag_records = pb.read('tags', fields=['id', 'name'], filter=tag_filter)
    if not tag_records:
        print(f"Error: 未找到符合条件的标签: {tags}")
        return []

    # 提取 tag_id 列表
    tag_ids = [tag['id'] for tag in tag_records]
    print(f"根据标签 {tags} 查询到的 tag_id 列表: {tag_ids}")

    # 根据 tag_id 列表从 insights 表中查询 insight_id 列表
    tag_id_filter = " OR ".join([f'tag="{tag_id}"' for tag_id in tag_ids])
    insights = pb.read('insights', fields=['id', 'tag'], filter=tag_id_filter)
    if not insights:
        print(f"Error: 未找到符合条件的 insights 对应标签: {tags}")
        return []

    # 提取 insight_id 列表
    insight_ids = [insight['id'] for insight in insights]
    print(f"根据标签 {tags} 查询到的 insight_id 列表: {insight_ids}")
    return insight_ids


async def fetch_insight_data(insight_ids):
    """
    从 Pb 数据库中获取 insight 的相关信息。

    :param insight_ids: List[str] - insight ID 列表
    :return: dict - 包含 insight_id 和对应信息的字典
    """
    insights_data = {}
    for insight_id in insight_ids:
        # 从 Pb 数据库中获取 insight 信息
        insight = pb.read('insights', filter=f'id="{insight_id}"')
        if not insight:
            print(f"Error: Insight {insight_id} not found in Pb database.")
            continue

        # 获取 insight 的文章列表
        article_ids = insight[0].get('articles', [])
        if not article_ids:
            print(f"Error: Insight {insight_id} has no associated articles.")
            continue

        # 获取文章的详细信息
        articles = []
        for article_id in article_ids:
            article = pb.read('articles', fields=['title', 'abstract', 'content', 'url', 'publish_time'], filter=f'id="{article_id}"')
            if article:
                articles.append(article[0])

        # 保存 insight 和文章信息
        insights_data[insight_id] = {
            "content": insight[0].get('content', ''),
            "articles": articles
        }

    return insights_data


async def send_to_report_api(insight_ids, toc, comment=""):
    """
    调用 /report 接口批量生成报告。

    :param insight_ids: List[str] - insight ID 列表
    :param toc: List[str] - 报告的目录结构
    :param comment: str - 报告的附加评论
    """
    async with httpx.AsyncClient() as client:
        for insight_id in insight_ids:
            payload = {
                "insight_id": insight_id,
                "toc": toc,
                "comment": comment
            }
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    response = await client.post(REPORT_API_URL, json=payload)
                    if response.status_code == 200:
                        print(f"成功生成报告: {insight_id}")
                        break
                    else:
                        print(f"生成报告失败: {insight_id}，状态码: {response.status_code}, 响应: {response.text}")
                        retries += 1
                except Exception as e:
                    print(f"调用 /report 接口时出错: {e}")
                    retries += 1
                
                if retries < MAX_RETRIES:
                    print(f"重试 {retries}/{MAX_RETRIES}...")
                    await asyncio.sleep(REQUEST_DELAY)  # 延迟后重试
                else:
                    print(f"报告生成失败（达到最大重试次数）: {insight_id}")
            
            # 每次请求之间添加延迟
            await asyncio.sleep(REQUEST_DELAY)


# 主函数
if __name__ == "__main__":
    tags = [
        "OpenAI动态",
        "威胁情报",
        "数据泄露"
    ]  # 替换为用户选择的标签

    toc = [
        "参考情报",
        "基本内容",
        "相关发声情况",
        "应对策略"
    ]

    comment = ""

    async def main():
        """
        主流程：
        1. 根据用户选择的标签获取符合条件的 insight_id 列表。
        2. 调用 /report 接口生成报告。
        """
        # 获取符合条件的 insight_ids
        insight_ids = await fetch_insight_ids_by_tags(tags)
        if not insight_ids:
            print("未找到符合条件的 insight_ids，流程终止。")
            return

        # 批量生成报告
        await send_to_report_api(insight_ids, toc, comment)

    asyncio.run(main())