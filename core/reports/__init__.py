from .get_report import pb, project_dir, logger, get_report

import os
import uuid
class ReportService:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", "")
        # 1. base initialization
        self.cache_url = os.path.join(self.project_dir, 'backend_service')
        os.makedirs(self.cache_url, exist_ok=True)

        # 2. load the llm
        # self.llm = LocalLlmWrapper()
        self.memory = {}
        # self.scholar = Scholar(initial_file_dir=os.path.join(self.project_dir, "files"), use_gpu=use_gpu)
        logger.info('backend service init success.')

    async def report(self, insight_id: str, topics: list[str], comment: str) -> dict:
        logger.debug(f'got new report request insight_id {insight_id}')
        insight = pb.read('insights', filter=f'id="{insight_id}"')
        if not insight:
            logger.error(f'insight {insight_id} not found')
            return self.build_out(-2, 'insight not found')

        article_ids = insight[0]['articles']
        if not article_ids:
            logger.error(f'insight {insight_id} has no articles')
            return self.build_out(-2, 'can not find articles for insight')

        article_list = [pb.read('articles', fields=['title', 'abstract', 'content', 'url', 'publish_time'], filter=f'id="{_id}"')
                        for _id in article_ids]
        article_list = [_article[0] for _article in article_list if _article]

        if not article_list:
            logger.debug(f'{insight_id} has no valid articles')
            return self.build_out(-2, f'{insight_id} has no valid articles')

        content = insight[0]['content']
        if insight_id in self.memory:
            memory = self.memory[insight_id]
        else:
            memory = ''

        docx_file = os.path.join(self.cache_url, f'{insight_id}_{uuid.uuid4()}.docx')
        flag, memory = await get_report(content, article_list, memory, topics, comment, docx_file)
        self.memory[insight_id] = memory

        if flag:
            file = open(docx_file, 'rb')
            message = pb.upload('insights', insight_id, 'docx', f'{insight_id}.docx', file)
            file.close()
            if message:
                logger.debug(f'report success finish and update to: {message}')
                return self.build_out(11, message)
            else:
                logger.error(f'{insight_id} report generate successfully, however failed to update to pb.')
                return self.build_out(-2, 'report generate successfully, however failed to update to pb.')
        else:
            logger.error(f'{insight_id} failed to generate report, finish.')
            return self.build_out(-11, 'report generate failed.')

    def build_out(self, flag: int, answer: str = "") -> dict:
        return {"flag": flag, "result": [{"type": "text", "answer": answer}]}

