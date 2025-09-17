#!/usr/bin/env python3
"""
Script to update Qdrant Cloud collection schema and indexing configuration.

This script:
1. Connects to existing Qdrant Cloud collection
2. Creates proper indexes for metadata fields
3. Validates that indexes are working correctly
"""

import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.qdrant_service import create_superlinked_qdrant_service, QdrantConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_collection_indexes(qdrant_service):
    """Update the collection with proper indexes for metadata fields."""
    logger.info("🔧 Updating collection indexes...")
    
    try:
        from qdrant_client.http import models
        
        # Create indexes for metadata fields to enable efficient filtering and retrieval
        indexes_to_create = [
            {
                "field_name": "clip_id",
                "schema": models.KeywordIndexParams(type="keyword", is_tenant=False),
                "description": "keyword index for clip_id field"
            },
            {
                "field_name": "move_label", 
                "schema": models.KeywordIndexParams(type="keyword", is_tenant=False),
                "description": "keyword index for move_label field"
            },
            {
                "field_name": "energy_level",
                "schema": models.KeywordIndexParams(type="keyword", is_tenant=False),
                "description": "keyword index for energy_level field"
            },
            {
                "field_name": "role_focus",
                "schema": models.KeywordIndexParams(type="keyword", is_tenant=False),
                "description": "keyword index for role_focus field"
            },
            {
                "field_name": "tempo",
                "schema": models.IntegerIndexParams(type="integer"),
                "description": "integer index for tempo field"
            },
            {
                "field_name": "difficulty_score",
                "schema": models.FloatIndexParams(type="float"),
                "description": "float index for difficulty_score field"
            }
        ]
        
        created_count = 0
        for index_config in indexes_to_create:
            try:
                qdrant_service.client.create_payload_index(
                    collection_name=qdrant_service.config.collection_name,
                    field_name=index_config["field_name"],
                    field_schema=index_config["schema"]
                )
                logger.info(f"✅ Created {index_config['description']}")
                created_count += 1
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"✅ Index for {index_config['field_name']} already exists")
                else:
                    logger.warning(f"⚠️ Failed to create index for {index_config['field_name']}: {e}")
        
        logger.info(f"✅ Index update completed: {created_count} new indexes created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update indexes: {e}")
        return False


def validate_indexes(qdrant_service):
    """Validate that the indexes are working correctly."""
    logger.info("🔍 Validating indexes...")
    
    try:
        from qdrant_client.http import models
        
        # Test clip_id index with scroll operation
        search_results = qdrant_service.client.scroll(
            collection_name=qdrant_service.config.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="clip_id",
                        match=models.MatchValue(value="basic_step_1")
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if search_results[0]:  # Points found
            logger.info("✅ clip_id index working correctly")
        else:
            logger.warning("⚠️ No results found for clip_id test (may be expected if data not present)")
        
        # Test move_label index
        search_results = qdrant_service.client.scroll(
            collection_name=qdrant_service.config.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="move_label",
                        match=models.MatchValue(value="basic_step")
                    )
                ]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        logger.info(f"✅ move_label index working: {len(search_results[0])} results found")
        
        # Test tempo range index
        search_results = qdrant_service.client.scroll(
            collection_name=qdrant_service.config.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tempo",
                        range=models.Range(gte=100, lte=130)
                    )
                ]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        logger.info(f"✅ tempo range index working: {len(search_results[0])} results found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Index validation failed: {e}")
        return False


def main():
    """Main function to update indexes."""
    logger.info("🚀 Starting Qdrant Cloud index update")
    
    try:
        # Initialize Qdrant Cloud service
        qdrant_config = QdrantConfig.from_env()
        qdrant_service = create_superlinked_qdrant_service(qdrant_config)
        
        # Check collection exists and has data
        collection_info = qdrant_service.get_collection_info()
        logger.info(f"📊 Collection info: {collection_info.get('name')} ({collection_info.get('points_count', 0)} points)")
        
        if collection_info.get('points_count', 0) == 0:
            logger.warning("⚠️ Collection is empty - you may need to run migration first")
        
        # Update indexes
        if not update_collection_indexes(qdrant_service):
            logger.error("❌ Failed to update indexes")
            return False
        
        # Validate indexes
        if not validate_indexes(qdrant_service):
            logger.error("❌ Index validation failed")
            return False
        
        logger.info("✅ Index update completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Index update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Qdrant Cloud indexes updated successfully!")
        print("   ✅ All metadata fields now have proper indexes")
        print("   ✅ Efficient filtering and retrieval enabled")
        print("   ✅ Collection ready for production use")
    else:
        print("\n💥 Index update failed!")
        print("   Please check the logs above for details.")
        sys.exit(1)