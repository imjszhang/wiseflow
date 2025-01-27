# -*- coding: utf-8 -*-
# General Crawler
# This script is designed to extract content from web pages, including articles and article lists.
# It uses multiple methods (e.g., GNE, LLMs) to parse and extract information from HTML content.

from gne import GeneralNewsExtractor
import httpx
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from core.llms.openai_wrapper import openai_llm
from core.llms.dify_wrapper import dify_llm
from bs4.element import Comment
from core.utils.general_utils import extract_and_convert_dates
import asyncio
import json_repair
import os
from typing import Union
from requests.compat import urljoin
from core.scrapers import scraper_map

# Dynamically select LLM provider based on environment variable
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "dify").lower()  # Default to "dify"

def select_llm():
    """
    Select the LLM provider based on the environment variable.

    :return: The selected LLM function (either OpenAI or Dify).
    :raises ValueError: If the LLM provider is unsupported.
    """
    if LLM_PROVIDER == "openai":
        return openai_llm
    elif LLM_PROVIDER == "dify":
        return dify_llm
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

# Initialize the LLM
llm = select_llm()

# Default model for HTML parsing
model = os.environ.get('HTML_PARSE_MODEL', 'gpt-4o-mini-2024-07-18')

# HTTP headers for requests
header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/604.1 Edg/112.0.100.0'
}

# Initialize the GNE extractor
extractor = GeneralNewsExtractor()

def tag_visible(element: Comment) -> bool:
    """
    Check if an HTML element is visible.

    :param element: The HTML element to check.
    :return: True if the element is visible, False otherwise.
    """
    if element.parent.name in ["style", "script", "head", "title", "meta", "[document]"]:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_soup(soup: BeautifulSoup) -> str:
    """
    Extract visible text from a BeautifulSoup object.

    :param soup: The BeautifulSoup object.
    :return: The extracted visible text.
    """
    res = []
    texts = soup.find_all(string=True)
    visible_texts = filter(tag_visible, texts)
    for v in visible_texts:
        res.append(v)
    text = "\n".join(res)
    return text.strip()

# System prompt for LLM
sys_info = '''Your task is to operate as an HTML content extractor, focusing on parsing a provided HTML segment. Your objective is to retrieve the following details directly from the raw text within the HTML, without summarizing or altering the content:

- The document's title
- The complete main content, as it appears in the HTML, comprising all textual elements considered part of the core article body
- The publication time in its original format found within the HTML

Ensure your response fits the following JSON structure, accurately reflecting the extracted data without modification:

```json
{
  "title": "The Document's Exact Title",
  "content": "All the unaltered primary text content from the article",
  "publish_time": "Original Publication Time as per HTML"
}
```

It is essential that your output adheres strictly to this format, with each field filled based on the untouched information extracted directly from the HTML source.'''

async def general_crawler(url: str, logger) -> tuple[int, Union[set, dict]]:
    """
    General crawler to extract article information or article list from a given URL.

    :param url: The URL to crawl.
    :param logger: Logger instance for logging.
    :return: A tuple containing a flag and the extracted data.
             Flags:
             - -7: Error during HTML fetch process.
             - 0: Error during content parsing process.
             - 1: The URL is likely an article list page (returns a set of article URLs).
             - 11: Success (returns a dictionary with article details).
    """
    # 0. Check if there's a specific scraper for this domain
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    base_url = f"{parsed_url.scheme}://{domain}"
    if domain in scraper_map:
        return await scraper_map[domain](url, logger)

    # 1. Fetch the HTML content using httpx
    async with httpx.AsyncClient() as client:
        for retry in range(2):
            try:
                response = await client.get(url, headers=header, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                if retry < 1:
                    logger.info(f"Cannot reach {url}\n{e}\nWaiting 1 minute...")
                    await asyncio.sleep(60)
                else:
                    logger.error(e)
                    return -7, {}

    # 2. Parse the HTML content
        page_source = response.text
        if not page_source:
            try:
                page_source = response.content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    page_source = response.content.decode('gbk')
                except Exception as e:
                    logger.error(f"Cannot decode HTML: {e}")
                    return -7, {}

        soup = BeautifulSoup(page_source, "html.parser")

        # Check if the URL is an article list page
        if len(url) < 50:
            urls = set()
            for link in soup.find_all("a", href=True):
                absolute_url = urljoin(base_url, link["href"])
                format_url = urlparse(absolute_url)
                if not format_url.netloc or format_url.netloc != domain:
                    continue
                absolute_url = f"{format_url.scheme}://{format_url.netloc}{format_url.path}{format_url.params}{format_url.query}"
                if absolute_url != url:
                    urls.add(absolute_url)

            if len(urls) > 24:
                logger.info(f"{url} is more like an article list page, found {len(urls)} URLs with the same domain.")
                return 1, urls

    # 3. Extract content using GNE
    try:
        result = extractor.extract(page_source)
        if 'meta' in result:
            del result['meta']

        if len(result['title']) < 4 or len(result['content']) < 24:
            logger.info(f"GNE extraction not good: {result}")
            result = None
    except Exception as e:
        logger.info(f"GNE extraction error: {e}")
        result = None

    # 4. Use LLM to analyze the HTML if GNE fails
    if not result:
        html_text = text_from_soup(soup)
        html_lines = [line.strip() for line in html_text.split('\n') if line.strip()]
        html_text = "\n".join(html_lines)
        if len(html_text) > 29999:
            logger.info(f"{url} content too long for LLM parsing.")
            return 0, {}

        messages = [
            {"role": "system", "content": sys_info},
            {"role": "user", "content": html_text}
        ]

        try:
            if LLM_PROVIDER == "openai":
                llm_output = llm(messages, model=model, logger=logger, temperature=0.01)
            elif LLM_PROVIDER == "dify":
                inputs = {'system': sys_info}
                response = llm(html_text, 'html_parser', inputs=inputs, logger=logger)
                llm_output = response['answer']
            else:
                raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

            result = json_repair.repair_json(llm_output, return_objects=True)
            if not isinstance(result, dict) or 'title' not in result or 'content' not in result:
                return 0, {}

        except Exception as e:
            logger.error(f"Error during LLM call: {e}")
            return 0, {}

    # 5. Post-process the extracted data
    date_str = extract_and_convert_dates(result.get('publish_time', ''))
    result['publish_time'] = date_str if date_str else datetime.strftime(datetime.today(), "%Y%m%d")
    from_site = domain.replace('www.', '').split('.')[0]
    result['content'] = f"[from {from_site}] {result['content']}"

    try:
        meta_description = soup.find("meta", {"name": "description"})
        result['abstract'] = f"[from {from_site}] {meta_description['content'].strip()}" if meta_description else ''
    except Exception:
        result['abstract'] = ''

    result['url'] = url
    return 11, result