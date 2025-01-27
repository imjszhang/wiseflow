from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Literal, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from core.insights import message_manager
from core.reports import ReportService
import os
import httpx
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Request(BaseModel):
    """
    数据模型：Request
    用于接收用户的请求数据。

    属性：
    - user_id (str): 用户的唯一标识符。
    - type (Literal): 消息的类型，支持以下类型：
        - "text": 文本消息
        - "publicMsg": 公共消息
        - "file": 文件
        - "image": 图片
        - "video": 视频
        - "location": 位置
        - "chathistory": 聊天记录
        - "site": 网站
        - "attachment": 附件
        - "url": 链接
    - content (str): 消息的主要内容。
    - addition (Optional[str]): 可选的附加信息。
    """
    user_id: str
    type: Literal["text", "publicMsg", "file", "image", "video", "location", "chathistory", "site", "attachment", "url"]
    content: str
    addition: Optional[str] = None


class InvalidInputException(HTTPException):
    """
    自定义异常类：InvalidInputException
    用于处理无效输入的异常。

    参数：
    - detail (str): 异常的详细信息。
    """
    def __init__(self, detail: str):
        super().__init__(status_code=442, detail=detail)


class TranslateRequest(BaseModel):
    """
    数据模型：TranslateRequest
    用于接收翻译请求的数据。

    属性：
    - article_ids (list[str]): 需要翻译的文章 ID 列表。
    """
    article_ids: list[str]


class ReportRequest(BaseModel):
    """
    数据模型：ReportRequest
    用于接收生成报告的请求数据。

    属性：
    - insight_id (str): 需要生成报告的洞察 ID。
    - toc (list[str]): 报告的目录结构。第一个元素是标题，其余是段落标题。
        - 第一个元素必须存在，可以是空字符，LLM 会自动生成标题。
    - comment (str): 报告的附加评论，默认为空字符串。
    """
    insight_id: str
    toc: list[str] = [""]  # The first element is a headline, and the rest are paragraph headings. The first element must exist, can be a null character, and llm will automatically make headings.
    comment: str = ""


# 创建 FastAPI 应用实例
app = FastAPI(
    title="WiseFlow Union Backend",
    description="From Wiseflow Team.",
    version="0.3.0",
    openapi_url="/openapi.json"
)

# 添加 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """
    根路径接口
    返回服务的基本信息。

    返回：
    - msg (str): 服务的欢迎信息和版本号。
    """
    msg = "Hello, this is Wise Union Backend, version 0.3.1"
    return {"msg": msg}


@app.post("/feed")
async def call_to_feed(background_tasks: BackgroundTasks, request: Request):
    """
    接口：/feed
    用于接收用户的消息，并将其传递给后台任务进行处理。

    参数：
    - background_tasks (BackgroundTasks): FastAPI 的后台任务管理器。
    - request (Request): 用户的请求数据，包含消息的类型、内容等。

    示例请求：
    1. 文本消息：
    ```json
    {
        "user_id": "user123",
        "type": "text",
        "content": "Check out this link: https://example.com",
        "addition": "Some additional info"
    }
    ```

    2. 公共消息：
    ```json
    {
        "user_id": "user456",
        "type": "publicMsg",
        "content": "<item><url><![CDATA[https://example.com]]></url><summary><![CDATA[This is a summary]]></summary></item>",
        "addition": null
    }
    ```

    3. URL 消息：
    ```json
    {
        "user_id": "user789",
        "type": "url",
        "content": "<url>https://example.com&amp;chksm=12345</url>",
        "addition": "Shared from WeChat"
    }
    ```

    返回：
    - msg (str): 确认消息已成功接收。
    """
    background_tasks.add_task(message_manager, _input=request.model_dump())
    return {"msg": "received well"}


# 实例化报告服务
rs = ReportService()

@app.post("/report")
async def report(request: ReportRequest):
    """
    接口：/report
    用于生成报告。

    参数：
    - request (ReportRequest): 报告请求数据，包含洞察 ID、目录结构和评论。

    返回：
    - 生成的报告内容（由 ReportService 提供）。
    """
    return await rs.report(request.insight_id, request.toc, request.comment)


async def verify_admin():
    pb_api_base = os.getenv("PB_API_BASE")
    pb_api_auth = os.getenv("PB_API_AUTH")
    
    try:
        async with httpx.AsyncClient() as client:
            # 打印请求信息
            logger.debug(f"Attempting to connect to: {pb_api_base}/api/admins/auth-refresh")
            logger.debug(f"Using Authorization header: {pb_api_auth[:10]}...")
            
            response = await client.post(
                f"{pb_api_base}/api/admins/auth-refresh",
                headers={"Authorization": pb_api_auth}
            )
            
            # 打印响应信息
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            
            if response.status_code != 200:
                raise Exception(f"验证失败: {response.text}")
            
            return True
    except Exception as e:
        logger.error(f"验证过程中发生错误: {str(e)}")
        raise e