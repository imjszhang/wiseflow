import os
from dotenv import load_dotenv
import asyncio
from core.insights import pipeline, pb, logger
# 加载 .env 文件
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.env')
load_dotenv(dotenv_path=env_path)

# 全局计数器，用于记录任务执行的次数
counter = 1


async def process_site(site, counter):
    """
    处理单个站点，根据站点的配置决定是否执行任务。

    :param site: dict - 包含站点信息的字典，需包含 'per_hours' 和 'url' 字段
    :param counter: int - 当前任务执行的计数器
    """
    # 检查站点是否配置了 'per_hours' 和 'url' 字段
    if not site['per_hours'] or not site['url']:
        return

    # 如果当前计数器是站点配置的 'per_hours' 的倍数，则执行任务
    if counter % site['per_hours'] == 0:
        logger.info(f"applying {site['url']}")
        # 调用 pipeline 函数处理站点的 URL
        await pipeline(site['url'].rstrip('/'))


async def schedule_pipeline(interval):
    """
    定时调度任务，根据指定的时间间隔循环执行。

    :param interval: int - 调度任务的时间间隔（秒）
    """
    global counter
    while True:
        # 从数据库中读取已激活的站点信息
        sites = pb.read('sites', filter='activated=True')
        logger.info(f'task execute loop {counter}')

        # 并发处理所有站点
        await asyncio.gather(*[process_site(site, counter) for site in sites])

        # 增加计数器
        counter += 1
        logger.info(f'task execute loop finished, work after {interval} seconds')

        # 等待指定的时间间隔后再执行下一轮任务
        await asyncio.sleep(interval)


async def main():
    """
    主函数，初始化调度任务。

    该函数设置调度的时间间隔（以小时为单位），并启动调度任务。
    """
    # 设置调度时间间隔（小时）
    interval_hours = 1
    # 将时间间隔转换为秒
    interval_seconds = interval_hours * 60 * 60
    # 启动调度任务
    await schedule_pipeline(interval_seconds)


# 启动主函数
asyncio.run(main())