import os
import requests
import json
from typing import List, Tuple
import aiohttp
from dify_api_config import DIFY_API_BASE_URL 

def send_chat_message(base_url: str, api_key: str, query: str, user: str, conversation_id: str, inputs={}, files: List[dict] = [], response_mode="streaming") -> Tuple[str, str]:
    url = f"{base_url}/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "response_mode": response_mode,
        "user": user,
        "conversation_id": conversation_id or "",
        "inputs": inputs,
        "files": files
    }

    try:
        response = requests.post(url, headers=headers, json=payload, stream=(response_mode == "streaming"))

        if response.status_code != 200:
            raise requests.HTTPError(f"Request failed with status code {response.status_code}: {response.text}")

        if response_mode == "blocking":
            return handle_blocking_response(response, conversation_id)
        elif response_mode == "streaming":
            return handle_streaming_response(response, conversation_id)

    except Exception as e:
        print(f"Error in send_chat_message: {e}")
        return "", conversation_id

def handle_blocking_response(response: requests.Response, conversation_id: str) -> Tuple[str, str]:
    try:
        response_json = response.json()
        result = response_json.get("answer", "")
        conversation_id = response_json.get("conversation_id", conversation_id)  # Use the passed conversation_id if not present in response
        return result, conversation_id
    except json.JSONDecodeError:
        raise requests.HTTPError("Failed to parse response JSON")

def handle_streaming_response(response: requests.Response, conversation_id: str) -> Tuple[str, str]:
    result = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data:"):
                data = decoded_line[5:].strip()
                try:
                    chunk = json.loads(data)
                    answer = chunk.get("answer", "")
                    conversation_id = chunk.get("conversation_id", conversation_id)  # Use the passed conversation_id if not present in response
                    result += answer
                except json.JSONDecodeError:
                    continue
    return result, conversation_id


def call_dify_app(api_key,content,conversation_id, inputs,files,response_mode="blocking"):
    base_url = DIFY_API_BASE_URL
    user = "kaichi_api"
    if not inputs:
        inputs = "[]"
    if not files:
        files = "[]"
    if not conversation_id: conversation_id = ""
    inputs = json.loads(inputs)
    files = json.loads(files)  
    result,conversation_id = send_chat_message(base_url, api_key, content, user=user,conversation_id=conversation_id, inputs=inputs, files=files,response_mode=response_mode)  
    return result,conversation_id

async def send_chat_message_async(base_url: str, api_key: str, query: str, user: str, conversation_id: str, inputs={}, files: List[dict] = [], response_mode="streaming") -> Tuple[str, str]:
    url = f"{base_url}/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "response_mode": response_mode,
        "user": user,
        "conversation_id": conversation_id or "",
        "inputs": inputs,
        "files": files
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    raise aiohttp.ClientError(f"Request failed with status code {response.status}: {await response.text()}")

                if response_mode == "blocking":
                    return await handle_blocking_response_async(response, conversation_id)
                elif response_mode == "streaming":
                    return await handle_streaming_response_async(response, conversation_id)

        except Exception as e:
            print(f"Error in send_chat_message_async: {e}")
            return "", conversation_id

async def handle_blocking_response_async(response: aiohttp.ClientResponse, conversation_id: str) -> Tuple[str, str]:
    try:
        response_json = await response.json()
        result = response_json.get("answer", "")
        conversation_id = response_json.get("conversation_id", conversation_id)  # Use the passed conversation_id if not present in response
        return result, conversation_id
    except aiohttp.ClientError:
        raise aiohttp.ClientError("Failed to parse response JSON")

async def handle_streaming_response_async(response: aiohttp.ClientResponse, conversation_id: str) -> Tuple[str, str]:
    result = ""
    async for line in response.content:
        decoded_line = line.decode('utf-8')
        if decoded_line.startswith("data:"):
            data = decoded_line[5:].strip()
            try:
                chunk = json.loads(data)
                answer = chunk.get("answer", "")
                conversation_id = chunk.get("conversation_id", conversation_id)  # Use the passed conversation_id if not present in response
                result += answer
            except json.JSONDecodeError:
                continue
    return result, conversation_id

async def call_dify_app_async(api_key, content, conversation_id, inputs, files, response_mode="blocking"):
    base_url = DIFY_API_BASE_URL
    user = "dify_api"
    if not inputs:
        inputs = "[]"
    if not files:
        files = "[]"
    if not conversation_id: conversation_id = ""
    inputs = json.loads(inputs)
    files = json.loads(files)
    result, conversation_id = await send_chat_message_async(base_url, api_key, content, user=user, conversation_id=conversation_id, inputs=inputs, files=files, response_mode=response_mode)
    return result, conversation_id