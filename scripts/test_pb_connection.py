import os
import sys
from loguru import logger
from pocketbase import PocketBase
from pocketbase.client import ClientResponseError

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG

def test_pb_connection():
    """Test PocketBase connection and basic operations"""
    
    # Initialize logger
    logger.add("pb_connection_test.log", rotation="100 MB")
    
    try:
        # Get PocketBase URL
        pb_url = CONFIG.get("POCKETBASE_URL")
        if not pb_url:
            logger.error("POCKETBASE_URL is not defined in configuration")
            return False
            
        logger.info(f"Testing connection to PocketBase at: {pb_url}")
        
        # Initialize PocketBase client
        pb_client = PocketBase(pb_url)
        
        # Test authentication
        admin_email = CONFIG.get("POCKETBASE_ADMIN_EMAIL")
        admin_password = CONFIG.get("POCKETBASE_ADMIN_PASSWORD")
        
        if admin_email and admin_password:
            try:
                auth_data = pb_client.admins.auth_with_password(admin_email, admin_password)
                logger.info("✅ Authentication successful!")
                logger.info(f"Authenticated as: {auth_data.admin.email}")
            except ClientResponseError as e:
                logger.error(f"❌ Authentication failed: {e.message}")
                return False
        else:
            logger.warning("⚠️ Admin credentials not provided, skipping authentication test")
        
        # Test fetching tags collection
        try:
            response = pb_client.collection('tags').get_list(
                page=1,
                per_page=10,
                query_params={
                    "filter": "activated=True"
                }
            )
            logger.info("✅ Successfully fetched tags collection")
            logger.info(f"Found {response.total_items} activated tags")
            
            # Display some tag details
            for item in response.items:
                logger.info(f"Tag: {item.name} (ID: {item.id})")
                
        except ClientResponseError as e:
            logger.error(f"❌ Failed to fetch tags: {e.message}")
            return False
            
        logger.info("🎉 All PocketBase connection tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during PocketBase connection test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_pb_connection()
    sys.exit(0 if success else 1) 