import os
import requests
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)
DIFY_BASE_URL = os.getenv('DIFY_API_BASE')
DIFY_DATASETS_API_KEY = os.getenv('DIFY_DATASETS_API_KEY')

class DifyDatasetsAPI:
    def __init__(self, api_key=DIFY_DATASETS_API_KEY, base_url=DIFY_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def create_dataset(self, name):
        url = f"{self.base_url}/datasets"
        data = {
            "name": name
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def list_datasets(self, page=1, limit=20):
        url = f"{self.base_url}/datasets?page={page}&limit={limit}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def create_document_by_text(self, dataset_id, name, text, indexing_technique="high_quality", process_rule=None):
        url = f"{self.base_url}/datasets/{dataset_id}/document/create_by_text"
        data = {
            "name": name,
            "text": text,
            "indexing_technique": indexing_technique,
            "process_rule": process_rule or {"mode": "automatic"}
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def create_document_by_file(self, dataset_id, file_path, data):
        url = f"{self.base_url}/datasets/{dataset_id}/document/create_by_file"
        files = {'file': open(file_path, 'rb')}
        response = requests.post(url, headers={'Authorization': f'Bearer {self.api_key}'}, files=files, data={'data': data})
        return response.json()

    def update_document_by_text(self, dataset_id, document_id, name=None, text=None, process_rule=None):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/update_by_text"
        data = {}
        if name:
            data["name"] = name
        if text:
            data["text"] = text
        if process_rule:
            data["process_rule"] = process_rule
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def update_document_by_file(self, dataset_id, document_id, file_path, data):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/update_by_file"
        files = {'file': open(file_path, 'rb')}
        response = requests.post(url, headers={'Authorization': f'Bearer {self.api_key}'}, files=files, data={'data': data})
        return response.json()

    def get_indexing_status(self, dataset_id, batch):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{batch}/indexing-status"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def delete_document(self, dataset_id, document_id):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}"
        response = requests.delete(url, headers=self.headers)
        return response.json()

    def list_documents(self, dataset_id, keyword=None, page=1, limit=20):
        url = f"{self.base_url}/datasets/{dataset_id}/documents?page={page}&limit={limit}"
        if keyword:
            url += f"&keyword={keyword}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def create_segment(self, dataset_id, document_id, segments):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/segments"
        data = {
            "segments": segments
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()

    def list_segments(self, dataset_id, document_id, keyword=None, status="completed"):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/segments?status={status}"
        if keyword:
            url += f"&keyword={keyword}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def delete_segment(self, dataset_id, document_id, segment_id):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}"
        response = requests.delete(url, headers=self.headers)
        return response.json()

    def update_segment(self, dataset_id, document_id, segment_id, segment_data):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}"
        data = {
            "segment": segment_data
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()
    



class AsyncDifyDatasetsAPI:
    def __init__(self, api_key=DIFY_DATASETS_API_KEY, base_url=DIFY_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    async def create_dataset(self, name):
        url = f"{self.base_url}/datasets"
        data = {
            "name": name
        }
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    async def list_datasets(self, page=1, limit=20):
        url = f"{self.base_url}/datasets?page={page}&limit={limit}"
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                return await response.json()

    async def create_document_by_text(self, dataset_id, name, text, indexing_technique="high_quality", process_rule=None):
        url = f"{self.base_url}/datasets/{dataset_id}/document/create_by_text"
        data = {
            "name": name,
            "text": text,
            "indexing_technique": indexing_technique,
            "process_rule": process_rule or {"mode": "automatic"}
        }
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    async def create_document_by_file(self, dataset_id, file_path, data):
        url = f"{self.base_url}/datasets/{dataset_id}/document/create_by_file"
        async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {self.api_key}'}) as session:
            with open(file_path, 'rb') as file:
                files = {'file': file}
                async with session.post(url, data={'data': data}, files=files) as response:
                    return await response.json()

    async def update_document_by_text(self, dataset_id, document_id, name=None, text=None, process_rule=None):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/update_by_text"
        data = {}
        if name:
            data["name"] = name
        if text:
            data["text"] = text
        if process_rule:
            data["process_rule"] = process_rule
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    async def update_document_by_file(self, dataset_id, document_id, file_path, data):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/update_by_file"
        async with aiohttp.ClientSession(headers={'Authorization': f'Bearer {self.api_key}'}) as session:
            with open(file_path, 'rb') as file:
                files = {'file': file}
                async with session.post(url, data={'data': data}, files=files) as response:
                    return await response.json()

    async def get_indexing_status(self, dataset_id, batch):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{batch}/indexing-status"
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                return await response.json()

    async def delete_document(self, dataset_id, document_id):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}"
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.delete(url) as response:
                return await response.json()

    async def list_documents(self, dataset_id, keyword=None, page=1, limit=20):
        url = f"{self.base_url}/datasets/{dataset_id}/documents?page={page}&limit={limit}"
        if keyword:
            url += f"&keyword={keyword}"
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                return await response.json()

    async def create_segment(self, dataset_id, document_id, segments):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/segments"
        data = {
            "segments": segments
        }
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(url, json=data) as response:
                return await response.json()

    async def list_segments(self, dataset_id, document_id, keyword=None, status="completed"):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/segments?status={status}"
        if keyword:
            url += f"&keyword={keyword}"
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url) as response:
                return await response.json()

    async def delete_segment(self, dataset_id, document_id, segment_id):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}"
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.delete(url) as response:
                return await response.json()

    async def update_segment(self, dataset_id, document_id, segment_id, segment_data):
        url = f"{self.base_url}/datasets/{dataset_id}/documents/{document_id}/segments/{segment_id}"
        data = {
            "segment": segment_data
        }
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(url, json=data) as response:
                return await response.json()