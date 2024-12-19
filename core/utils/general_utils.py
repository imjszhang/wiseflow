from urllib.parse import urlparse
import os
import re
import jieba


def isURL(string: str) -> bool:
    """
    检查字符串是否为有效的 URL。

    :param string: str - 待检测的字符串
    :return: bool - 如果是有效的 URL，返回 True；否则返回 False
    """
    if string.startswith("www."):
        string = f"https://{string}"
    result = urlparse(string)
    return result.scheme != '' and result.netloc != ''


def extract_urls(text: str) -> set:
    """
    从文本中提取所有 URL。

    :param text: str - 包含 URL 的文本
    :return: set - 提取的 URL 集合
    """
    # 匹配 http, https 和 www 开头的 URL
    url_pattern = re.compile(r'((?:https?://|www\.)[-A-Za-z0-9+&@#/%?=~_|!:,.;]*[-A-Za-z0-9+&@#/%=~_|])')
    urls = re.findall(url_pattern, text)
    cleaned_urls = set()
    for url in urls:
        if url.startswith("www."):
            url = f"https://{url}"
        parsed_url = urlparse(url)
        if not parsed_url.netloc:
            continue
        # 移除 URL 中的 hash fragment
        if not parsed_url.scheme:
            cleaned_urls.add(f"https://{parsed_url.netloc}{parsed_url.path}{parsed_url.params}{parsed_url.query}")
        else:
            cleaned_urls.add(
                f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}{parsed_url.params}{parsed_url.query}")
    return cleaned_urls


def isChinesePunctuation(char: str) -> bool:
    """
    检查字符是否为中文标点符号。

    :param char: str - 待检测的字符
    :return: bool - 如果是中文标点符号，返回 True；否则返回 False
    """
    chinese_punctuations = set(range(0x3000, 0x303F)) | set(range(0xFF00, 0xFFEF))
    return ord(char) in chinese_punctuations


def is_chinese(string: str) -> bool:
    """
    检测字符串是否主要由中文字符组成。

    :param string: str - 待检测的字符串
    :return: bool - 如果大部分是中文字符，返回 True；否则返回 False
    """
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    non_chinese_count = len(pattern.findall(string))
    return (non_chinese_count / len(string)) < 0.68


def extract_and_convert_dates(input_string: str) -> str:
    """
    从字符串中提取日期并转换为标准格式 YYYYMMDD。

    :param input_string: str - 包含日期的字符串
    :return: str - 提取的日期（格式为 YYYYMMDD），如果未找到则返回 None
    """
    if not isinstance(input_string, str):
        return None

    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        r'(\d{4})/(\d{2})/(\d{2})',  # YYYY/MM/DD
        r'(\d{4})\.(\d{2})\.(\d{2})',  # YYYY.MM.DD
        r'(\d{4})\\(\d{2})\\(\d{2})',  # YYYY\MM\DD
        r'(\d{4})(\d{2})(\d{2})'  # YYYYMMDD
    ]

    matches = []
    for pattern in patterns:
        matches = re.findall(pattern, input_string)
        if matches:
            break
    if matches:
        return ''.join(matches[0])
    return None


def get_logger_level() -> str:
    """
    获取日志级别。

    :return: str - 日志级别（CRITICAL, DEBUG, INFO, WARNING, ERROR）
    """
    level_map = {
        'silly': 'CRITICAL',
        'verbose': 'DEBUG',
        'info': 'INFO',
        'warn': 'WARNING',
        'error': 'ERROR',
    }
    level: str = os.environ.get('WS_LOG', 'info').lower()
    if level not in level_map:
        raise ValueError(
            'WiseFlow LOG should support the values of `silly`, '
            '`verbose`, `info`, `warn`, `error`'
        )
    return level_map.get(level, 'info')


def compare_phrase_with_list(target_phrase: str, phrase_list: list, threshold: float) -> list:
    """
    比较目标短语与短语列表中每个短语的相似性。

    :param target_phrase: str - 目标短语
    :param phrase_list: list - 短语列表
    :param threshold: float - 相似性阈值
    :return: list - 满足相似性条件的短语列表
    """
    if not target_phrase:
        return []  # 如果目标短语为空，直接返回空列表

    # 分词处理
    target_tokens = set(jieba.lcut(target_phrase))
    tokenized_phrases = {phrase: set(jieba.lcut(phrase)) for phrase in phrase_list}

    # 计算相似性
    similar_phrases = [phrase for phrase, tokens in tokenized_phrases.items()
                       if len(target_tokens & tokens) / min(len(target_tokens), len(tokens)) > threshold]

    return similar_phrases