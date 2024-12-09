import pandas as pd
import httpx
import re
import sys
import asyncio
from typing import List

# 提取 URL 的正则表达式
URL_PATTERN = re.compile(r'https?://[^\s]+')

# /feed 接口的 URL
FEED_API_URL = "http://localhost:8077/feed"  # 替换为实际的接口地址

# 延迟时间（秒）
REQUEST_DELAY = 3  # 每次请求之间的延迟时间
MAX_RETRIES = 1  # 最大重试次数

def extract_urls_from_csv(file_path: str, column_name: str) -> List[str]:
    """
    从指定的 CSV 文件中提取所有链接。
    
    :param file_path: CSV 文件路径
    :param column_name: 包含链接的列名
    :return: 提取的链接列表
    """
    try:
        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns:
            raise ValueError(f"列名 '{column_name}' 不存在于 CSV 文件中。")
        
        # 提取指定列中的所有链接
        urls = []
        for content in df[column_name].dropna():
            urls.extend(URL_PATTERN.findall(str(content)))
        
        return urls
    except Exception as e:
        print(f"读取 CSV 文件时出错: {e}")
        sys.exit(1)

async def send_to_feed_api(urls: List[str], user_id: str = "script_user"):
    """
    调用 /feed 接口解析链接。
    
    :param urls: 链接列表
    :param user_id: 用户 ID，默认为 "script_user"
    """
    async with httpx.AsyncClient() as client:
        for url in urls:
            payload = {
                "user_id": user_id,
                "type": "text",
                "content": url,
                "addition": None
            }
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    response = await client.post(FEED_API_URL, json=payload)
                    if response.status_code == 200:
                        print(f"成功解析链接: {url}")
                        break
                    else:
                        print(f"解析链接失败: {url}，状态码: {response.status_code}, 响应: {response.text}")
                        retries += 1
                except Exception as e:
                    print(f"调用 /feed 接口时出错: {e}")
                    retries += 1
                
                if retries < MAX_RETRIES:
                    print(f"重试 {retries}/{MAX_RETRIES}...")
                    await asyncio.sleep(REQUEST_DELAY)  # 延迟后重试
                else:
                    print(f"链接解析失败（达到最大重试次数）: {url}")
            
            # 每次请求之间添加延迟
            await asyncio.sleep(REQUEST_DELAY)

def main():
    """
    主函数，读取 CSV 文件并调用 /feed 接口。
    """
    if len(sys.argv) < 3:
        print("用法: python get_insights_from_csv.py <csv_file_path> <column_name>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    column_name = sys.argv[2]
    
    # 提取链接
    urls = extract_urls_from_csv(csv_file_path, column_name)
    if not urls:
        print("未在指定的列中找到任何链接。")
        sys.exit(0)
    
    print(f"找到 {len(urls)} 个链接，开始解析...")
    
    # 调用 /feed 接口
    asyncio.run(send_to_feed_api(urls))

if __name__ == "__main__":
    main()