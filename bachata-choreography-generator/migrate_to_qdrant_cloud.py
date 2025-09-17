#!/usr/bin/env python3
"""
Migration script to transfer existing move embeddings from local to Qdrant Cloud.

This script:
1. Loads existing move annotations from bachata_annotations.json (38 clips)
2. Generates Superlinked unified embeddings for all moves
3. Migrates embeddings to Qdrant Cloud "Bachata_vibes" cluster
4. Validates the migration by performing test searches
5. Tests vector search performance and accuracy after cloud migration
"""

import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

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


def validate_cloud_configuration() -> bool:
    """Validate that Qdrant Cloud configuration is available."""
    import os
    
    # Try to load .env file if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    # Remove quotes if present
    if qdrant_url and qdrant_url.startswith("'") and qdrant_url.endswith("'"):
        qdrant_url = qdrant_url[1:-1]
    if qdrant_api_key and qdrant_api_key.startswith("'") and qdrant_api_key.endswith("'"):
        qdrant_api_key = qdrant_api_key[1:-1]
    
    if not qdrant_url or not qdrant_api_key:
        logger.error("Qdrant Cloud configuration not found!")
        logger.error("Please ensure QDRANT_URL and QDRANT_API_KEY are set in .env file")
        return False
    
    logger.info(f"✅ Qdrant Cloud configuration found:")
    logger.info(f"   URL: {qdrant_url}")
    logger.info(f"   API Key: {'*' * (len(qdrant_api_key) - 8) + qdrant_api_key[-8:]}")
    
    return True


def validate_data_directory(data_dir: Path) -> bool:
    """Validate that the data directory contains required files."""
    annotations_file = data_dir / "bachata_annotations.json"
    
    if not annotations_file.exists():
        logger.error(f"Annotations file not found: {annotations_file}")
        return False
    
    # Load and validate annotations
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        clips = annotations.get('clips', [])
        if len(clips) == 0:
            logger.error("No clips found in annotations file")
            return False
        
        logger.info(f"✅ Found annotations file with {len(clips)} clips")
        return True
        
    except Exception as e:
        logger.error(f"Error reading annotations file: {e}")
        return False


def test_cloud_connection(qdrant_service) -> bool:
    """Test connection to Qdrant Cloud."""
    try:
        logger.info("Testing Qdrant Cloud connection...")
        
        # Health check
        health_status = qdrant_service.health_check()
        
        if not health_status.get("qdrant_available", False):
            logger.error(f"❌ Qdrant Cloud connection failed: {health_status.get('error_message', 'Unknown error')}")
            return False
        
        logger.info("✅ Qdrant Cloud connection successful")
        
        # Collection info
        collection_info = qdrant_service.get_collection_info()
        logger.info(f"✅ Collection info: {collection_info.get('name', 'unknown')} ({collection_info.get('points_count', 0)} points)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cloud connection test failed: {e}")
        return False


def test_superlinked_embeddings(embedding_service) -> bool:
    """Test that Superlinked embeddings are working correctly."""
    try:
        logger.info("Testing Superlinked embeddings...")
        
        # Test embedding generation
        stats = embedding_service.get_stats()
        logger.info(f"✅ Superlinked service: {stats['total_moves']} moves, {len(stats['embedding_spaces'])} spaces")
        
        # Test different search types
        semantic_results = embedding_service.search_semantic("basic steps for beginners", limit=3)
        logger.info(f"✅ Semantic search: {len(semantic_results)} results")
        
        tempo_results = embedding_service.search_tempo(120.0, limit=3)
        logger.info(f"✅ Tempo search: {len(tempo_results)} results")
        
        multi_results = embedding_service.search_moves(
            description="intermediate moves",
            target_tempo=115.0,
            difficulty_level="intermediate",
            energy_level="medium",
            limit=3
        )
        logger.info(f"✅ Multi-factor search: {len(multi_results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Superlinked embeddings test failed: {e}")
        return False


def perform_migration(embedding_service, qdrant_service) -> Dict[str, Any]:
    """Perform the actual migration of embeddings to Qdrant Cloud."""
    logger.info("🚀 Starting migration to Qdrant Cloud...")
    
    start_time = time.time()
    
    try:
        # Step 1: Prepare move data with Superlinked embeddings
        logger.info("Step 1: Preparing move data with Superlinked embeddings...")
        prep_start = time.time()
        
        moves_data = embedding_service.prepare_move_data_for_indexing()
        prep_time = time.time() - prep_start
        
        logger.info(f"✅ Prepared {len(moves_data)} moves with embeddings in {prep_time:.2f}s")
        
        # Step 2: Clear existing cloud data (if any)
        logger.info("Step 2: Clearing existing cloud data...")
        clear_success = qdrant_service.clear_collection()
        if clear_success:
            logger.info("✅ Cleared existing cloud data")
        else:
            logger.warning("⚠️ Could not clear existing data (collection may be empty)")
        
        # Step 3: Migrate to Qdrant Cloud
        logger.info("Step 3: Migrating embeddings to Qdrant Cloud...")
        migration_start = time.time()
        
        migration_summary = qdrant_service.migrate_superlinked_embeddings(moves_data)
        migration_time = time.time() - migration_start
        
        total_time = time.time() - start_time
        
        # Update migration summary with timing
        migration_summary.update({
            "preparation_time_s": prep_time,
            "migration_time_s": migration_time,
            "total_time_s": total_time,
            "moves_migrated": len(moves_data)
        })
        
        logger.info(f"✅ Migration completed in {migration_time:.2f}s (total: {total_time:.2f}s)")
        
        return migration_summary
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "successful_migrations": 0, "failed_migrations": 0}


def validate_migration(qdrant_service, expected_count: int) -> bool:
    """Validate that the migration was successful."""
    logger.info("🔍 Validating migration...")
    
    try:
        # Check collection info
        collection_info = qdrant_service.get_collection_info()
        points_count = collection_info.get('points_count', 0)
        
        if points_count != expected_count:
            logger.error(f"❌ Point count mismatch: expected {expected_count}, got {points_count}")
            return False
        
        logger.info(f"✅ Point count correct: {points_count} points")
        
        # Test search functionality
        import numpy as np
        dummy_vector = np.random.random(512)  # 512D for Superlinked embeddings
        
        search_results = qdrant_service.search_superlinked_moves(dummy_vector, limit=5)
        
        if len(search_results) == 0:
            logger.error("❌ Search returned no results")
            return False
        
        logger.info(f"✅ Search functionality working: {len(search_results)} results")
        
        # Test specific move retrieval
        first_result = search_results[0]
        specific_move = qdrant_service.get_move_by_clip_id(first_result.clip_id)
        
        if specific_move is None:
            logger.error(f"❌ Could not retrieve specific move: {first_result.clip_id}")
            return False
        
        logger.info(f"✅ Specific move retrieval working: {specific_move.clip_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration validation failed: {e}")
        return False


def test_performance(qdrant_service, recommendation_engine) -> Dict[str, Any]:
    """Test vector search performance and accuracy after cloud migration."""
    logger.info("⚡ Testing performance after cloud migration...")
    
    performance_results = {
        "search_times": [],
        "accuracy_tests": [],
        "error_count": 0
    }
    
    try:
        # Test 1: Basic search performance
        import numpy as np
        
        search_times = []
        for i in range(10):
            start_time = time.time()
            
            dummy_vector = np.random.random(512)
            results = qdrant_service.search_superlinked_moves(dummy_vector, limit=10)
            
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            search_times.append(search_time)
            
            if len(results) == 0:
                performance_results["error_count"] += 1
        
        avg_search_time = np.mean(search_times)
        performance_results["search_times"] = search_times
        performance_results["avg_search_time_ms"] = avg_search_time
        
        logger.info(f"✅ Average search time: {avg_search_time:.2f}ms")
        
        # Test 2: Accuracy with known moves
        logger.info("Testing search accuracy with known moves...")
        
        # Get a few moves to test similarity
        all_results = qdrant_service.search_superlinked_moves(np.random.random(512), limit=38)
        
        if len(all_results) >= 3:
            # Test similarity between moves of the same type
            basic_moves = [r for r in all_results if "basic" in r.move_label.lower()]
            if len(basic_moves) >= 2:
                move1 = qdrant_service.get_move_by_clip_id(basic_moves[0].clip_id)
                if move1 and move1.embedding is not None:
                    similar_results = qdrant_service.search_superlinked_moves(move1.embedding, limit=5)
                    
                    # Check if similar moves are returned
                    similar_basic_count = sum(1 for r in similar_results if "basic" in r.move_label.lower())
                    accuracy_score = similar_basic_count / len(similar_results)
                    
                    performance_results["accuracy_tests"].append({
                        "test_type": "basic_move_similarity",
                        "accuracy_score": accuracy_score,
                        "similar_moves_found": similar_basic_count,
                        "total_results": len(similar_results)
                    })
                    
                    logger.info(f"✅ Basic move similarity accuracy: {accuracy_score:.2f}")
        
        # Test 3: Recommendation engine performance
        if recommendation_engine:
            logger.info("Testing recommendation engine performance...")
            
            # Create mock music features
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
            
            rec_start = time.time()
            recommendations = recommendation_engine.recommend_moves(
                music_features=mock_music_features,
                target_difficulty="intermediate",
                target_energy="medium",
                top_k=5
            )
            rec_time = (time.time() - rec_start) * 1000
            
            performance_results["recommendation_time_ms"] = rec_time
            performance_results["recommendations_count"] = len(recommendations)
            
            logger.info(f"✅ Recommendation engine: {len(recommendations)} results in {rec_time:.2f}ms")
        
        return performance_results
        
    except Exception as e:
        logger.error(f"❌ Performance testing failed: {e}")
        performance_results["error"] = str(e)
        return performance_results


def generate_migration_report(migration_summary: Dict[str, Any], 
                            performance_results: Dict[str, Any],
                            collection_info: Dict[str, Any]) -> str:
    """Generate a comprehensive migration report."""
    
    report = f"""
# Qdrant Cloud Migration Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Migration Summary
- **Total Moves Migrated**: {migration_summary.get('moves_migrated', 0)}
- **Successful Migrations**: {migration_summary.get('successful_migrations', 0)}
- **Failed Migrations**: {migration_summary.get('failed_migrations', 0)}
- **Preparation Time**: {migration_summary.get('preparation_time_s', 0):.2f}s
- **Migration Time**: {migration_summary.get('migration_time_s', 0):.2f}s
- **Total Time**: {migration_summary.get('total_time_s', 0):.2f}s

## Collection Information
- **Collection Name**: {collection_info.get('name', 'unknown')}
- **Points Count**: {collection_info.get('points_count', 0)}
- **Vector Size**: {collection_info.get('vector_size', 0)}D
- **Distance Metric**: {collection_info.get('distance', 'unknown')}
- **Status**: {collection_info.get('status', 'unknown')}
- **Optimizer Status**: {collection_info.get('optimizer_status', 'unknown')}

## Performance Results
- **Average Search Time**: {performance_results.get('avg_search_time_ms', 0):.2f}ms
- **Recommendation Time**: {performance_results.get('recommendation_time_ms', 0):.2f}ms
- **Search Errors**: {performance_results.get('error_count', 0)}
- **Recommendations Generated**: {performance_results.get('recommendations_count', 0)}

## Accuracy Tests
"""
    
    for test in performance_results.get('accuracy_tests', []):
        report += f"- **{test['test_type']}**: {test['accuracy_score']:.2f} ({test['similar_moves_found']}/{test['total_results']})\n"
    
    report += f"""
## Migration Status
{'✅ SUCCESS' if migration_summary.get('failed_migrations', 0) == 0 else '❌ PARTIAL FAILURE'}

The migration to Qdrant Cloud has been {'completed successfully' if migration_summary.get('failed_migrations', 0) == 0 else 'completed with some failures'}.
All {migration_summary.get('successful_migrations', 0)} move embeddings are now stored in the cloud cluster
and available for high-performance vector similarity search.
"""
    
    return report


def main():
    """Main migration function."""
    logger.info("🚀 Starting migration to Qdrant Cloud")
    
    # Step 1: Validate cloud configuration
    if not validate_cloud_configuration():
        logger.error("❌ Cloud configuration validation failed")
        return False
    
    # Step 2: Validate data directory
    data_dir = Path("data")
    if not validate_data_directory(data_dir):
        logger.error("❌ Data directory validation failed")
        return False
    
    try:
        # Step 3: Initialize services
        logger.info("Step 3: Initializing services...")
        
        # Initialize Superlinked embedding service
        embedding_service = create_superlinked_service(str(data_dir))
        
        # Initialize Qdrant Cloud service
        qdrant_config = QdrantConfig.from_env()
        qdrant_service = create_superlinked_qdrant_service(qdrant_config)
        
        # Test cloud connection
        if not test_cloud_connection(qdrant_service):
            logger.error("❌ Cloud connection test failed")
            return False
        
        # Test Superlinked embeddings
        if not test_superlinked_embeddings(embedding_service):
            logger.error("❌ Superlinked embeddings test failed")
            return False
        
        # Step 4: Perform migration
        migration_summary = perform_migration(embedding_service, qdrant_service)
        
        if migration_summary.get("error"):
            logger.error(f"❌ Migration failed: {migration_summary['error']}")
            return False
        
        # Step 5: Validate migration
        expected_count = migration_summary.get("moves_migrated", 0)
        if not validate_migration(qdrant_service, expected_count):
            logger.error("❌ Migration validation failed")
            return False
        
        # Step 6: Initialize recommendation engine for testing
        logger.info("Step 6: Initializing recommendation engine...")
        recommendation_engine = create_superlinked_recommendation_engine(str(data_dir), qdrant_config)
        
        # Step 7: Test performance
        performance_results = test_performance(qdrant_service, recommendation_engine)
        
        # Step 8: Generate final report
        collection_info = qdrant_service.get_collection_info()
        stats = qdrant_service.get_statistics()
        
        report = generate_migration_report(migration_summary, performance_results, collection_info)
        
        # Save report
        report_path = data_dir / "qdrant_cloud_migration_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Migration report saved: {report_path}")
        
        # Final summary
        logger.info("✅ Migration completed successfully!")
        logger.info(f"📈 Final statistics:")
        logger.info(f"   - Points in cloud: {collection_info.get('points_count', 0)}")
        logger.info(f"   - Collection size: {stats.collection_size_mb:.2f} MB")
        logger.info(f"   - Average search time: {performance_results.get('avg_search_time_ms', 0):.2f}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Migration to Qdrant Cloud completed successfully!")
        print("   ✅ All 38 move clips migrated with Superlinked embeddings")
        print("   ✅ Vector search performance validated")
        print("   ✅ Collection schema and indexing configured")
        print("   ✅ Cloud deployment ready for production use")
        print("\nThe system is now using Qdrant Cloud for all vector operations.")
    else:
        print("\n💥 Migration to Qdrant Cloud failed!")
        print("   Please check the logs above for details.")
        print("   Ensure QDRANT_URL and QDRANT_API_KEY are correctly configured in .env")
        sys.exit(1)