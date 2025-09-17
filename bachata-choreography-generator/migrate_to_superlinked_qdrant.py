#!/usr/bin/env python3
"""
Migration script to convert existing 40 move annotations to Superlinked embedding format
and populate Qdrant vector database.

This script:
1. Loads existing move annotations from bachata_annotations.json
2. Generates Superlinked unified embeddings for all moves
3. Migrates embeddings to Qdrant vector database
4. Validates the migration by performing test searches
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.superlinked_embedding_service import create_superlinked_service
from app.services.qdrant_service import create_superlinked_qdrant_service, QdrantConfig
from app.services.superlinked_recommendation_engine import create_superlinked_recommendation_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_directory(data_dir: Path) -> bool:
    """Validate that the data directory contains required files."""
    annotations_file = data_dir / "bachata_annotations.json"
    
    if not annotations_file.exists():
        logger.error(f"Annotations file not found: {annotations_file}")
        return False
    
    logger.info(f"Found annotations file: {annotations_file}")
    return True


def test_superlinked_embeddings(embedding_service) -> bool:
    """Test that Superlinked embeddings are working correctly."""
    try:
        # Test embedding generation
        stats = embedding_service.get_stats()
        logger.info(f"Superlinked service stats: {stats['total_moves']} moves, {len(stats['embedding_spaces'])} embedding spaces")
        
        # Test semantic search
        semantic_results = embedding_service.search_semantic("basic steps for beginners", limit=3)
        logger.info(f"Semantic search test: {len(semantic_results)} results")
        
        # Test tempo search
        tempo_results = embedding_service.search_tempo(120.0, limit=3)
        logger.info(f"Tempo search test: {len(tempo_results)} results")
        
        # Test multi-factor search
        multi_results = embedding_service.search_moves(
            description="intermediate moves",
            target_tempo=115.0,
            difficulty_level="intermediate",
            energy_level="medium",
            limit=3
        )
        logger.info(f"Multi-factor search test: {len(multi_results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"Superlinked embeddings test failed: {e}")
        return False


def test_qdrant_integration(qdrant_service) -> bool:
    """Test that Qdrant integration is working correctly."""
    try:
        # Health check
        health_status = qdrant_service.health_check()
        if not health_status.get("qdrant_available", False):
            logger.error(f"Qdrant health check failed: {health_status}")
            return False
        
        # Collection info
        collection_info = qdrant_service.get_collection_info()
        logger.info(f"Qdrant collection info: {collection_info}")
        
        # Statistics
        stats = qdrant_service.get_statistics()
        logger.info(f"Qdrant statistics: {stats.total_points} points, {stats.collection_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Qdrant integration test failed: {e}")
        return False


def test_unified_search(recommendation_engine) -> bool:
    """Test the unified recommendation engine with Qdrant."""
    try:
        # Create mock music features for testing
        import numpy as np
        from app.services.music_analyzer import MusicFeatures
        
        mock_music_features = MusicFeatures(
            tempo=120.0,
            beat_positions=np.array([0.5, 1.0, 1.5, 2.0]),
            duration=180.0,
            rms_energy=np.array([0.1, 0.2, 0.15, 0.18]),
            spectral_centroid=np.array([1500, 1600, 1550, 1580]),
            percussive_component=np.array([0.1, 0.12, 0.11, 0.13]),
            energy_profile=np.array([0.5, 0.7, 0.6, 0.65])
        )
        
        # Test unified vector recommendations
        logger.info("Testing unified vector recommendations...")
        recommendations = recommendation_engine.recommend_moves(
            music_features=mock_music_features,
            target_difficulty="intermediate",
            target_energy="medium",
            top_k=5
        )
        
        logger.info(f"Unified search returned {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations[:3]):
            logger.info(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")
        
        # Test natural language queries
        logger.info("Testing natural language queries...")
        nl_recommendations = recommendation_engine.recommend_with_natural_language(
            "energetic intermediate moves for 125 BPM song",
            mock_music_features,
            top_k=3
        )
        
        logger.info(f"Natural language search returned {len(nl_recommendations)} recommendations:")
        for i, rec in enumerate(nl_recommendations):
            logger.info(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")
        
        # Test performance stats
        perf_stats = recommendation_engine.get_performance_stats()
        logger.info(f"Performance stats: {perf_stats['performance']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Unified search test failed: {e}")
        return False


def main():
    """Main migration function."""
    logger.info("Starting Superlinked + Qdrant migration")
    
    # Configuration
    data_dir = Path("data")
    qdrant_config = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="superlinked_bachata_moves",
        vector_size=512  # Will be updated based on actual embedding size
    )
    
    # Validate data directory
    if not validate_data_directory(data_dir):
        logger.error("Data directory validation failed")
        return False
    
    try:
        # Step 1: Initialize Superlinked embedding service
        logger.info("Step 1: Initializing Superlinked embedding service...")
        embedding_service = create_superlinked_service(str(data_dir))
        
        # Update vector size based on actual embedding dimension
        qdrant_config.vector_size = embedding_service.total_dimension
        logger.info(f"Updated Qdrant vector size to {qdrant_config.vector_size}D")
        
        # Test Superlinked embeddings
        if not test_superlinked_embeddings(embedding_service):
            logger.error("Superlinked embeddings test failed")
            return False
        
        # Step 2: Initialize Qdrant service
        logger.info("Step 2: Initializing Qdrant service...")
        qdrant_service = create_superlinked_qdrant_service(qdrant_config)
        
        # Test Qdrant integration
        if not test_qdrant_integration(qdrant_service):
            logger.error("Qdrant integration test failed")
            return False
        
        # Step 3: Prepare move data for migration
        logger.info("Step 3: Preparing move data with Superlinked embeddings...")
        start_time = time.time()
        
        moves_data = embedding_service.prepare_move_data_for_indexing()
        preparation_time = time.time() - start_time
        
        logger.info(f"Prepared {len(moves_data)} moves with embeddings in {preparation_time:.2f}s")
        
        # Step 4: Migrate to Qdrant
        logger.info("Step 4: Migrating embeddings to Qdrant...")
        migration_start = time.time()
        
        migration_summary = qdrant_service.migrate_superlinked_embeddings(moves_data)
        migration_time = time.time() - migration_start
        
        logger.info(f"Migration completed in {migration_time:.2f}s")
        logger.info(f"Migration summary: {migration_summary}")
        
        if migration_summary.get("failed_migrations", 0) > 0:
            logger.warning(f"Some migrations failed: {migration_summary}")
        
        # Step 5: Initialize unified recommendation engine
        logger.info("Step 5: Initializing unified recommendation engine...")
        recommendation_engine = create_superlinked_recommendation_engine(str(data_dir), qdrant_config)
        
        # Step 6: Test unified search functionality
        logger.info("Step 6: Testing unified search functionality...")
        if not test_unified_search(recommendation_engine):
            logger.error("Unified search test failed")
            return False
        
        # Step 7: Final validation
        logger.info("Step 7: Final validation...")
        
        # Check final collection state
        final_collection_info = qdrant_service.get_collection_info()
        final_stats = qdrant_service.get_statistics()
        
        logger.info("Migration completed successfully!")
        logger.info(f"Final collection state:")
        logger.info(f"  - Points: {final_collection_info.get('points_count', 0)}")
        logger.info(f"  - Size: {final_stats.collection_size_mb:.2f} MB")
        logger.info(f"  - Vector dimension: {final_collection_info.get('vector_size', 0)}")
        
        # Performance summary
        engine_stats = recommendation_engine.get_performance_stats()
        logger.info(f"Engine performance:")
        logger.info(f"  - Qdrant available: {recommendation_engine.is_qdrant_available()}")
        logger.info(f"  - Total searches: {engine_stats['performance']['unified_searches']}")
        logger.info(f"  - Qdrant searches: {engine_stats['performance']['qdrant_searches']}")
        logger.info(f"  - Fallback searches: {engine_stats['performance']['fallback_searches']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Superlinked + Qdrant migration completed successfully!")
        print("\n🎉 Migration successful! The system is now using:")
        print("   - Superlinked unified embeddings (512D)")
        print("   - Qdrant vector database for optimized search")
        print("   - Preserved linear relationships for tempo/difficulty")
        print("   - Natural language query support")
        print("\nYou can now use the choreography generator with the new unified approach.")
    else:
        logger.error("❌ Migration failed!")
        print("\n💥 Migration failed! Please check the logs above for details.")
        print("   Make sure Qdrant Cloud credentials are configured in .env file")
        sys.exit(1)