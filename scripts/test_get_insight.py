import os
import sys
from loguru import logger

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.insights.get_insight import InsightExtractor

def test_insight_extraction():
    """Test the insight extraction functionality"""
    
    # Initialize logger
    logger.add("insight_test.log", rotation="100 MB")
    
    try:
        # Initialize InsightExtractor
        logger.info("Initializing InsightExtractor...")
        extractor = InsightExtractor()
        
        # Test article content
        test_article = """
        [from 新华社] 近日，国家发展改革委等部门联合印发《关于推进人工智能进校园的指导意见》。
        《意见》提出，到2025年，基本建立人工智能进校园的政策制度体系和实施机制，形成一批可复制可推广的经验做法。

        专家表示，人工智能技术在教育领域的应用，可以实现因材施教、个性化学习，提高教育教学质量和效率。
        同时，也要注意防范人工智能应用中的数据安全、隐私保护等风险。

        教育部相关负责人表示，将加强人工智能教育资源建设，支持开发适合不同学段的人工智能教育课程和教材，
        培养具有人工智能素养的创新型人才。同时，也将加强教师培训，提升教师运用人工智能技术的能力。
        """
        
        # Test get_insights method
        logger.info("Testing get_insights method...")
        insights = extractor.get_insights(test_article)
        
        if insights:
            logger.info("✅ Successfully extracted insights")
            logger.info(f"Number of insights found: {len(insights)}")
            
            # Display extracted insights
            for i, insight in enumerate(insights, 1):
                logger.info(f"\nInsight {i}:")
                logger.info(f"Tag ID: {insight['tag']}")
                logger.info(f"Content: {insight['content']}")
        else:
            logger.warning("⚠️ No insights were extracted from the test article")
        
        # Test insight rewrite functionality
        if insights:
            logger.info("\nTesting insight_rewrite method...")
            contents = [insight['content'] for insight in insights]
            rewritten = extractor.insight_rewrite(contents)
            
            if rewritten:
                logger.info("✅ Successfully rewrote insights")
                logger.info(f"Rewritten content:\n{rewritten}")
            else:
                logger.warning("⚠️ Insight rewrite returned empty result")
        
        logger.info("🎉 All insight extraction tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during insight extraction test: {str(e)}")
        return False

def main():
    # First test PocketBase connection
    logger.info("Testing PocketBase connection...")
    from scripts.test_pb_connection import test_pb_connection
    
    if not test_pb_connection():
        logger.error("❌ PocketBase connection test failed. Stopping further tests.")
        return False
    
    # Then test insight extraction
    logger.info("\nStarting insight extraction tests...")
    return test_insight_extraction()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 