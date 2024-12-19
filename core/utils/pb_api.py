import os
from pocketbase import PocketBase  # Client also works the same
from pocketbase.client import FileUpload
from typing import BinaryIO, Optional, List, Dict


class PbTalker:
    """
    PocketBase 客户端封装类，用于与 PocketBase 后端交互。
    提供了常用的 CRUD 操作和文件上传功能。
    """

    def __init__(self, logger) -> None:
        """
        初始化 PocketBase 客户端。

        :param logger: 日志记录器实例
        """
        # 获取 PocketBase API 基础 URL
        url = os.environ.get('PB_API_BASE', "http://127.0.0.1:8090")
        self.logger = logger
        self.logger.debug(f"Initializing PocketBase client: {url}")
        self.client = PocketBase(url)

        # 获取认证信息
        auth = os.environ.get('PB_API_AUTH', '')
        if not auth or "|" not in auth:
            self.logger.warning("Invalid email|password found. Operating without authentication. "
                                "Ensure collection rules allow unauthenticated access.")
        else:
            email, password = auth.split('|')
            try:
                # 尝试以管理员身份认证
                admin_data = self.client.admins.auth_with_password(email, password)
                if admin_data:
                    self.logger.info(f"PocketBase authenticated as admin - {email}")
            except:
                # 如果管理员认证失败，尝试以普通用户身份认证
                user_data = self.client.collection("users").auth_with_password(email, password)
                if user_data:
                    self.logger.info(f"PocketBase authenticated as user - {email}")
                else:
                    raise Exception("PocketBase authentication failed")

    def read(self, collection_name: str, fields: Optional[List[str]] = None, filter: str = '', skiptotal: bool = True) -> list:
        """
        从指定集合中读取数据。

        :param collection_name: str - 集合名称
        :param fields: Optional[List[str]] - 要返回的字段列表
        :param filter: str - 过滤条件
        :param skiptotal: bool - 是否跳过总计数
        :return: list - 查询结果列表
        """
        results = []
        for i in range(1, 10):  # 分页读取，最多读取 10 页
            try:
                res = self.client.collection(collection_name).get_list(
                    i, 500,  # 每页最多 500 条记录
                    {
                        "filter": filter,
                        "fields": ','.join(fields) if fields else '',
                        "skiptotal": skiptotal
                    }
                )
            except Exception as e:
                self.logger.error(f"PocketBase get list failed: {e}")
                continue
            if not res.items:
                break
            for _res in res.items:
                attributes = vars(_res)
                results.append(attributes)
        return results

    def add(self, collection_name: str, body: Dict) -> str:
        """
        向指定集合中添加新记录。

        :param collection_name: str - 集合名称
        :param body: Dict - 要添加的记录数据
        :return: str - 新记录的 ID，如果失败返回空字符串
        """
        try:
            res = self.client.collection(collection_name).create(body)
        except Exception as e:
            self.logger.error(f"PocketBase create failed: {e}")
            return ''
        return res.id

    def update(self, collection_name: str, id: str, body: Dict) -> str:
        """
        更新指定集合中的记录。

        :param collection_name: str - 集合名称
        :param id: str - 要更新的记录 ID
        :param body: Dict - 更新的数据
        :return: str - 更新后的记录 ID，如果失败返回空字符串
        """
        try:
            res = self.client.collection(collection_name).update(id, body)
        except Exception as e:
            self.logger.error(f"PocketBase update failed: {e}")
            return ''
        return res.id

    def delete(self, collection_name: str, id: str) -> bool:
        """
        删除指定集合中的记录。

        :param collection_name: str - 集合名称
        :param id: str - 要删除的记录 ID
        :return: bool - 删除成功返回 True，否则返回 False
        """
        try:
            res = self.client.collection(collection_name).delete(id)
        except Exception as e:
            self.logger.error(f"PocketBase delete failed: {e}")
            return False
        return bool(res)

    def upload(self, collection_name: str, id: str, key: str, file_name: str, file: BinaryIO) -> str:
        """
        上传文件到指定集合中的记录。

        :param collection_name: str - 集合名称
        :param id: str - 记录 ID
        :param key: str - 文件字段的键名
        :param file_name: str - 文件名
        :param file: BinaryIO - 文件对象
        :return: str - 更新后的记录 ID，如果失败返回空字符串
        """
        try:
            res = self.client.collection(collection_name).update(id, {key: FileUpload((file_name, file))})
        except Exception as e:
            self.logger.error(f"PocketBase file upload failed: {e}")
            return ''
        return res.id

    def view(self, collection_name: str, item_id: str, fields: Optional[List[str]] = None) -> Dict:
        """
        查看指定集合中的单条记录。

        :param collection_name: str - 集合名称
        :param item_id: str - 记录 ID
        :param fields: Optional[List[str]] - 要返回的字段列表
        :return: Dict - 记录数据，如果失败返回空字典
        """
        try:
            res = self.client.collection(collection_name).get_one(
                item_id,
                {"fields": ','.join(fields) if fields else ''}
            )
            return vars(res)
        except Exception as e:
            self.logger.error(f"PocketBase view item failed: {e}")
            return {}