import os
import requests
import aiohttp
import asyncio
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# 从环境变量中获取 API 基础 URL 和密钥
base_url = os.getenv('DIFY_API_BASE')
token = os.getenv('DIFY_API_KEY')

if not base_url or not token:
    raise ValueError("DIFY_API_BASE and DIFY_API_KEY must be set")

# 设置请求头
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

def dify_llm(query: str, user: str = "wiseflow", conversation_id: str = None, inputs: dict = None, response_mode: str = "blocking", logger=None, **kwargs) -> dict:
    """
    调用 Dify API 发送对话消息并返回结果（同步版本）。

    :param query: 用户输入的提问内容
    :param user: 用户标识
    :param conversation_id: 会话 ID（可选）
    :param inputs: 传入的变量值（可选）
    :param response_mode: 响应模式，默认为 "blocking"
    :param logger: 日志记录器（可选）
    :param kwargs: 其他可选参数
    :return: 包含回答和元数据的字典
    """
    if logger:
        logger.debug(f'query: {query}')
        logger.debug(f'user: {user}')
        logger.debug(f'conversation_id: {conversation_id}')
        logger.debug(f'inputs: {inputs}')
        logger.debug(f'response_mode: {response_mode}')
        logger.debug(f'kwargs: {kwargs}')

    url = f"{base_url}/chat-messages"
    payload = {
        "query": query,
        "user": user,
        "response_mode": response_mode,
        "inputs": inputs or {}
    }

    if conversation_id:
        payload["conversation_id"] = conversation_id

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=100)
        response.raise_for_status()
        data = response.json()

        if logger:
            logger.debug(f'API response: {data}')

        # 解析返回结果
        if "answer" in data:
            result = {
                "answer": data["answer"],
                "conversation_id": data.get("conversation_id"),
                "message_id": data.get("message_id"),
                "metadata": data.get("metadata", {})
            }
            return result
        else:
            if logger:
                logger.error("No 'answer' field in the response.")
            return {"error": "No answer returned from the API."}
    except requests.exceptions.RequestException as e:
        if logger:
            logger.error(f'dify_llm error: {e}')
        return {"error": str(e)}
    except Exception as e:
        if logger:
            logger.error(f'Unexpected error: {e}')
        return {"error": str(e)}

async def dify_llm_async(query: str, user: str = "wiseflow", conversation_id: str = None, inputs: dict = None, response_mode: str = "blocking", logger=None, **kwargs) -> dict:
    """
    异步调用 Dify API 发送对话消息并返回结果。

    :param query: 用户输入的提问内容
    :param user: 用户标识
    :param conversation_id: 会话 ID（可选）
    :param inputs: 传入的变量值（可选）
    :param response_mode: 响应模式，默认为 "blocking"
    :param logger: 日志记录器（可选）
    :param kwargs: 其他可选参数
    :return: 包含回答和元数据的字典
    """
    if logger:
        logger.debug(f'query: {query}')
        logger.debug(f'user: {user}')
        logger.debug(f'conversation_id: {conversation_id}')
        logger.debug(f'inputs: {inputs}')
        logger.debug(f'response_mode: {response_mode}')
        logger.debug(f'kwargs: {kwargs}')

    url = f"{base_url}/chat-messages"
    payload = {
        "query": query,
        "user": user,
        "response_mode": response_mode,
        "inputs": inputs or {}
    }

    if conversation_id:
        payload["conversation_id"] = conversation_id

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload, timeout=100) as response:
                response.raise_for_status()
                data = await response.json()

                if logger:
                    logger.debug(f'API response: {data}')

                # 解析返回结果
                if "answer" in data:
                    result = {
                        "answer": data["answer"],
                        "conversation_id": data.get("conversation_id"),
                        "message_id": data.get("message_id"),
                        "metadata": data.get("metadata", {})
                    }
                    return result
                else:
                    if logger:
                        logger.error("No 'answer' field in the response.")
                    return {"error": "No answer returned from the API."}
        except aiohttp.ClientError as e:
            if logger:
                logger.error(f'dify_llm_async error: {e}')
            return {"error": str(e)}
        except Exception as e:
            if logger:
                logger.error(f'Unexpected error: {e}')
            return {"error": str(e)}

# 示例用法
if __name__ == "__main__":
    # 同步调用示例
    logger = logging.getLogger("dify_llm")
    logging.basicConfig(level=logging.DEBUG)

    query = "What are the specifications of the iPhone 13 Pro Max?"
    user = "user123"
    response = dify_llm(query=query, user=user, logger=logger)

    if "error" in response:
        print(f"Error: {response['error']}")
    else:
        print(f"Answer: {response['answer']}")
        print(f"Conversation ID: {response['conversation_id']}")
        print(f"Message ID: {response['message_id']}")
        print(f"Metadata: {response['metadata']}")

    # 异步调用示例
    async def main():
        async_logger = logging.getLogger("dify_llm_async")
        logging.basicConfig(level=logging.DEBUG)

        async_query = "What are the specifications of the iPhone 13 Pro Max?"
        async_user = "user123"
        async_response = await dify_llm_async(query=async_query, user=async_user, logger=async_logger)

        if "error" in async_response:
            print(f"Error: {async_response['error']}")
        else:
            print(f"Answer: {async_response['answer']}")
            print(f"Conversation ID: {async_response['conversation_id']}")
            print(f"Message ID: {async_response['message_id']}")
            print(f"Metadata: {async_response['metadata']}")

    asyncio.run(main())