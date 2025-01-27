# 导入所有需要对外暴露的组件
from core.insights.get_insight import InsightExtractor, insight_extractor
from core.insights.pipeline import pipeline, InsightPipeline
from core.insights.message_manager import message_manager, MessageManager


# 明确声明可以被导入的内容
__all__ = [
    # 工具类实例
    'insight_extractor',
    'pipeline',
    'message_manager',
    
    # 主要功能函数（通过 InsightExtractor 实例调用）
    'insight_extractor.get_insights',
    'insight_extractor.insight_rewrite',
    
    # 类定义（如果其他地方需要继承或类型注解）
    'InsightExtractor',
    'InsightPipeline',
    'MessageManager'
]

# 版本信息（可选）
__version__ = '1.0.0'