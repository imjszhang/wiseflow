# -*- coding: utf-8 -*-
# 微信公众号文章爬虫 (mp_crawler)
# 该脚本专门用于爬取微信公众号文章及其目录页面，提取文章的标题、作者、发布时间、摘要、内容和图片等信息。

from typing import Union
import httpx
from bs4 import BeautifulSoup
from datetime import datetime
import re
import asyncio

# HTTP 请求头
header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/604.1 Edg/112.0.100.0'
}

async def mp_crawler(url: str, logger) -> tuple[int, Union[set, dict]]:
    """
    爬取微信公众号文章或文章目录页面。

    :param url: str - 目标微信公众号文章或目录的 URL
    :param logger: Logger - 日志记录器
    :return: tuple[int, Union[set, dict]] - 返回状态码和数据
             - -5: URL 不是微信公众号的链接
             - -7: 爬取失败或解析失败
             - 0: 内容解析失败
             - 1: 文章目录页面，返回文章 URL 集合
             - 11: 成功，返回文章详情字典
    """
    # 检查 URL 是否为微信公众号链接
    if not url.startswith('https://mp.weixin.qq.com') and not url.startswith('http://mp.weixin.qq.com'):
        logger.warning(f'{url} is not a mp url, you should not use this function')
        return -5, {}

    # 确保 URL 使用 HTTPS 协议
    url = url.replace("http://", "https://", 1)

    # 使用 httpx 异步客户端获取页面内容
    async with httpx.AsyncClient() as client:
        for retry in range(2):
            try:
                response = await client.get(url, headers=header, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                if retry < 1:
                    logger.info(f"{e}\nwaiting 1min")
                    await asyncio.sleep(60)
                else:
                    logger.warning(e)
                    return -7, {}

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 如果是文章目录页面
        if url.startswith('https://mp.weixin.qq.com/mp/appmsgalbum'):
            # 提取文章链接
            urls = {li.attrs['data-link'].replace("http://", "https://", 1) for li in soup.find_all('li', class_='album__list-item')}
            simple_urls = set()
            for url in urls:
                cut_off_point = url.find('chksm=')
                if cut_off_point != -1:
                    url = url[:cut_off_point - 1]
                simple_urls.add(url)
            return 1, simple_urls

        # 提取文章的发布时间
        pattern = r"var createTime = '(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}'"
        match = re.search(pattern, response.text)
        if match:
            date_only = match.group(1)
            publish_time = date_only.replace('-', '')
        else:
            publish_time = datetime.strftime(datetime.today(), "%Y%m%d")

        # 提取文章的标题、作者和摘要
        try:
            meta_description = soup.find('meta', attrs={'name': 'description'})
            summary = meta_description['content'].strip() if meta_description else ''
            rich_media_title = soup.find('h1', id='activity-name').text.strip() \
                if soup.find('h1', id='activity-name') \
                else soup.find('h1', class_='rich_media_title').text.strip()
            profile_nickname = soup.find('div', class_='wx_follow_nickname').text.strip()
        except Exception as e:
            logger.warning(f"not mp format: {url}\n{e}")
            return -7, {}

        if not rich_media_title or not profile_nickname:
            logger.warning(f"failed to analysis {url}, no title or profile_nickname")
            return -7, {}

        # 提取文章内容和图片
        texts = []
        images = set()
        content_area = soup.find('div', id='js_content')
        if content_area:
            # 提取文本内容
            for section in content_area.find_all(['section', 'p'], recursive=False):
                text = section.get_text(separator=' ', strip=True)
                if text and text not in texts:
                    texts.append(text)

            # 提取图片链接
            for img in content_area.find_all('img', class_='rich_pages wxw-img'):
                img_src = img.get('data-src') or img.get('src')
                if img_src:
                    images.add(img_src)

            cleaned_texts = [t for t in texts if t.strip()]
            content = '\n'.join(cleaned_texts)
        else:
            logger.warning(f"failed to analysis contents {url}")
            return 0, {}

        if content:
            content = f"[from {profile_nickname}]{content}"
        else:
            content = f"[from {profile_nickname}]{summary}"

        # 提取 meta 标签中的图片链接
        og_image = soup.find('meta', property='og:image')
        twitter_image = soup.find('meta', property='twitter:image')
        if og_image:
            images.add(og_image['content'])
        if twitter_image:
            images.add(twitter_image['content'])

        # 生成摘要
        if rich_media_title == summary or not summary:
            abstract = ''
        else:
            abstract = f"[from {profile_nickname}]{rich_media_title}——{summary}"

    # 返回文章详情
    return 11, {
        'title': rich_media_title,
        'author': profile_nickname,
        'publish_time': publish_time,
        'abstract': abstract,
        'content': content,
        'images': list(images),
        'url': url,
    }