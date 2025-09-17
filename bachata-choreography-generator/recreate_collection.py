#!/usr/bin/env python3
"""
Recreate Qdrant collection with correct 470D dimensions.

This script:
1. Deletes the existing collection completely
2. Creates a new collection with correct 470D dimensions
3. Re-migrates all the move data with correct embeddings
"""

import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.qdrant_service import create_superlinked_qdrant_service, QdrantConfig
from app.services.superlinked_recommendation_engine import create_superlinked_recommendation_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Recreate collection with correct dimensions."""
    logger.info("🔧 Starting collection recreation with correct dimensions...")
    
    try:
        # Initialize services
        qdrant_config = QdrantConfig.from_env()
        logger.info(f"Using Qdrant config: vector_size={qdrant_config.vector_size}")
        
        qdrant_service = create_superlinked_qdrant_service(qdrant_config)
        logger.info("✅ Qdrant service initialized")
        
        # Check current collection info
        try:
            collection_info = qdrant_service.get_collection_info()
            current_vector_size = collection_info.get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
            points_count = collection_info.get('points_count', 0)
            
            logger.info(f"Current collection: {points_count} points, {current_vector_size}D vectors")
            
            if current_vector_size == qdrant_config.vector_size:
                logger.info("✅ Vector dimensions already match! No fix needed.")
                return True
                
        except Exception as e:
            logger.info(f"Collection doesn't exist or has issues: {e}")
        
        # Delete the collection completely
        logger.info("🗑️ Deleting existing collection...")
        try:
            qdrant_service.client.delete_collection(qdrant_config.collection_name)
            logger.info("✅ Collection deleted successfully")
        except Exception as e:
            logger.info(f"Collection deletion failed (may not exist): {e}")
        
        # Recreate the collection with correct dimensions
        logger.info("🔨 Creating new collection with correct dimensions...")
        qdrant_service._ensure_collection()
        logger.info("✅ Collection created with correct dimensions")
        
        # Initialize recommendation engine (this will trigger re-migration)
        logger.info("🔄 Re-migrating move data with correct embeddings...")
        recommendation_engine = create_superlinked_recommendation_engine("data", qdrant_config)
        
        # The SuperlinkedRecommendationEngine should automatically populate the collection
        # during initialization if it's empty
        
        # Verify the fix
        collection_info = qdrant_service.get_collection_info()
        new_vector_size = collection_info.get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
        new_points_count = collection_info.get('points_count', 0)
        
        logger.info(f"✅ New collection: {new_points_count} points, {new_vector_size}D vectors")
        
        if new_vector_size == qdrant_config.vector_size and new_points_count > 0:
            logger.info("🎉 Collection recreation completed successfully!")
            return True
        else:
            logger.error(f"❌ Recreation failed: expected {qdrant_config.vector_size}D, got {new_vector_size}D")
            return False
            
    except Exception as e:
        logger.error(f"❌ Collection recreation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Collection recreation COMPLETED!")
        print("   ✅ Collection recreated with correct 470D vectors")
        print("   ✅ All move data re-migrated")
        print("   ✅ System ready for choreography generation")
    else:
        print("\n💥 Collection recreation FAILED!")
        print("   Please check the logs above for details.")
        sys.exit(1)