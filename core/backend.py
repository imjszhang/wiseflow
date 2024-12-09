from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Literal, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from insights import message_manager
from reports import ReportService


class Request(BaseModel):
    """
    Input model
    input = {'user_id': str, 'type': str, 'content':str， 'addition': Optional[str]}
    Type is one of "text", "publicMsg", "site" and "url"；
    """
    user_id: str
    type: Literal["text", "publicMsg", "file", "image", "video", "location", "chathistory", "site", "attachment", "url"]
    content: str
    addition: Optional[str] = None

class InvalidInputException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=442, detail=detail)


class TranslateRequest(BaseModel):
    article_ids: list[str]


class ReportRequest(BaseModel):
    insight_id: str
    toc: list[str] = [""]  # The first element is a headline, and the rest are paragraph headings. The first element must exist, can be a null character, and llm will automatically make headings.
    comment: str = ""


app = FastAPI(
    title="WiseFlow Union Backend",
    description="From Wiseflow Team.",
    version="0.3.0",
    openapi_url="/openapi.json"
)

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
def read_root():
    msg = "Hello, this is Wise Union Backend, version 0.3.1"
    return {"msg": msg}



@app.post("/feed")
async def call_to_feed(background_tasks: BackgroundTasks, request: Request):
    """
    以下是一些可能的请求示例：
    文本消息

    JSON

    {

    "user_id": "user123",

    "type": "text",

    "content": "Check out this link: https://example.com",

    "addition": "Some additional info"

    }

    公共消息

    JSON

    {

    "user_id": "user456",

    "type": "publicMsg",

    "content": "<item><url><![CDATA[https://example.com]]></url><summary><![CDATA[This is a summary]]></summary></item>",

    "addition": null

    }

    URL 消息

    JSON

    {

    "user_id": "user789",

    "type": "url",

    "content": "<url>https://example.com&amp;chksm=12345</url>",

    "addition": "Shared from WeChat"

    }
    """
    background_tasks.add_task(message_manager, _input=request.model_dump())
    return {"msg": "received well"}


rs = ReportService()

@app.post("/report")
async def report(request: ReportRequest):
    return await rs.report(request.insight_id, request.toc, request.comment)
