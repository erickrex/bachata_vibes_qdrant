#!/usr/bin/env python3
"""
Test script for Qdrant integration in RecommendationEngine.
Tests the new Qdrant-based similarity search functionality.
"""

import sys
import logging
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.recommendation_engine import RecommendationEngine, RecommendationRequest
from app.services.qdrant_service import QdrantConfig, create_qdrant_service
from app.services.music_analyzer import MusicAnalyzer
from app.services.move_analyzer import MoveAnalyzer
from app.services.feature_fusion import FeatureFusion
from app.services.annotation_interface import AnnotationInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_qdrant_health():
    """Test Qdrant service health and connectivity."""
    logger.info("=== Testing Qdrant Health ===")
    
    try:
        # Create Qdrant service using environment-based configuration for cloud deployment
        config = QdrantConfig.from_env()
        # Override collection name for testing
        config.collection_name = "test_bachata_moves"
        
        qdrant_service = create_qdrant_service(config)
        
        # Perform health check
        health_status = qdrant_service.health_check()
        logger.info(f"Qdrant health status: {health_status}")
        
        if health_status.get("qdrant_available", False):
            logger.info("✅ Qdrant is available and healthy")
            
            # Get collection info
            collection_info = qdrant_service.get_collection_info()
            logger.info(f"Collection info: {collection_info}")
            
            return True
        else:
            logger.warning("❌ Qdrant is not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Qdrant health check failed: {e}")
        return False


def test_recommendation_engine_initialization():
    """Test RecommendationEngine initialization with Qdrant."""
    logger.info("=== Testing RecommendationEngine Initialization ===")
    
    try:
        # Test with Qdrant enabled
        config = QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_bachata_moves"
        )
        
        engine = RecommendationEngine(use_qdrant=True, qdrant_config=config)
        
        logger.info(f"✅ RecommendationEngine initialized")
        logger.info(f"Qdrant available: {engine.is_qdrant_available()}")
        
        # Get performance stats
        stats = engine.get_performance_comparison()
        logger.info(f"Initial performance stats: {stats}")
        
        return engine
        
    except Exception as e:
        logger.error(f"❌ RecommendationEngine initialization failed: {e}")
        return None


def test_move_candidates_loading():
    """Test loading move candidates from annotations."""
    logger.info("=== Testing Move Candidates Loading ===")
    
    try:
        # Load annotations
        annotation_interface = AnnotationInterface("data")
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips
        
        logger.info(f"Loaded {len(annotations)} annotations")
        
        if not annotations:
            logger.warning("❌ No annotations found - cannot test move candidates")
            return []
        
        # Initialize analyzers
        music_analyzer = MusicAnalyzer()
        move_analyzer = MoveAnalyzer()
        feature_fusion = FeatureFusion()
        
        # Create move candidates from first few annotations
        move_candidates = []
        test_annotations = annotations[:5]  # Test with first 5 moves
        
        logger.info(f"Creating move candidates from {len(test_annotations)} annotations...")
        
        for annotation in test_annotations:
            try:
                # Analyze the move
                analysis_result = move_analyzer.analyze_move_clip(annotation.video_path)
                
                # Create dummy music features for embedding
                dummy_audio_path = "data/songs/Amor.mp3"  # Use existing song
                if Path(dummy_audio_path).exists():
                    music_features = music_analyzer.analyze_audio(dummy_audio_path)
                    
                    # Create multimodal embedding
                    multimodal_embedding = feature_fusion.create_multimodal_embedding(
                        music_features, analysis_result
                    )
                    
                    # Create move candidate
                    from app.services.recommendation_engine import MoveCandidate
                    candidate = MoveCandidate(
                        move_id=annotation.clip_id,
                        video_path=annotation.video_path,
                        move_label=annotation.move_label,
                        analysis_result=analysis_result,
                        multimodal_embedding=multimodal_embedding,
                        energy_level=annotation.energy_level,
                        difficulty=annotation.difficulty,
                        estimated_tempo=annotation.estimated_tempo,
                        lead_follow_roles=annotation.lead_follow_roles
                    )
                    
                    move_candidates.append(candidate)
                    logger.info(f"✅ Created candidate for {annotation.move_label}")
                    
                else:
                    logger.warning(f"❌ Test audio file not found: {dummy_audio_path}")
                    break
                    
            except Exception as e:
                logger.warning(f"❌ Failed to create candidate for {annotation.move_label}: {e}")
                continue
        
        logger.info(f"✅ Created {len(move_candidates)} move candidates")
        return move_candidates
        
    except Exception as e:
        logger.error(f"❌ Move candidates loading failed: {e}")
        return []


def test_qdrant_population(engine, move_candidates):
    """Test populating Qdrant with move embeddings."""
    logger.info("=== Testing Qdrant Population ===")
    
    if not engine or not engine.is_qdrant_available():
        logger.warning("❌ Qdrant not available, skipping population test")
        return False
    
    if not move_candidates:
        logger.warning("❌ No move candidates available, skipping population test")
        return False
    
    try:
        # Clear existing data
        if engine.qdrant_service:
            engine.qdrant_service.clear_collection()
            logger.info("Cleared existing Qdrant collection")
        
        # Populate with move candidates
        summary = engine.populate_qdrant_from_candidates(move_candidates)
        
        logger.info(f"Population summary: {summary}")
        
        if "error" not in summary:
            successful = summary.get("successful_migrations", 0)
            logger.info(f"✅ Successfully populated Qdrant with {successful} embeddings")
            return True
        else:
            logger.error(f"❌ Qdrant population failed: {summary['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Qdrant population test failed: {e}")
        return False


def test_qdrant_vs_memory_search(engine, move_candidates):
    """Test and compare Qdrant vs in-memory search performance."""
    logger.info("=== Testing Qdrant vs Memory Search Performance ===")
    
    if not move_candidates:
        logger.warning("❌ No move candidates available, skipping search test")
        return
    
    try:
        # Create a test music features and embedding
        music_analyzer = MusicAnalyzer()
        feature_fusion = FeatureFusion()
        
        # Use existing song for test
        test_audio_path = "data/songs/Amor.mp3"
        if not Path(test_audio_path).exists():
            logger.warning(f"❌ Test audio file not found: {test_audio_path}")
            return
        
        music_features = music_analyzer.analyze_audio(test_audio_path)
        music_embedding = feature_fusion.create_multimodal_embedding(
            music_features, move_candidates[0].analysis_result
        )
        
        # Create recommendation request
        request = RecommendationRequest(
            music_features=music_features,
            music_embedding=music_embedding,
            target_difficulty="intermediate",
            target_energy="medium",
            tempo_tolerance=10.0
        )
        
        # Test Qdrant search (if available)
        if engine.is_qdrant_available():
            logger.info("Testing Qdrant-based search...")
            qdrant_recommendations = engine.recommend_moves(request, None, top_k=10)
            logger.info(f"✅ Qdrant search returned {len(qdrant_recommendations)} recommendations")
            
            # Show top 3 results
            for i, rec in enumerate(qdrant_recommendations[:3]):
                logger.info(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.overall_score:.3f})")
        
        # Test in-memory search (force fallback)
        logger.info("Testing in-memory search...")
        engine.qdrant_available = False  # Temporarily disable Qdrant
        memory_recommendations = engine.recommend_moves(request, move_candidates, top_k=10)
        logger.info(f"✅ In-memory search returned {len(memory_recommendations)} recommendations")
        
        # Show top 3 results
        for i, rec in enumerate(memory_recommendations[:3]):
            logger.info(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.overall_score:.3f})")
        
        # Re-enable Qdrant
        engine.qdrant_available = engine.is_qdrant_available()
        
        # Get performance comparison
        perf_stats = engine.get_performance_comparison()
        logger.info(f"Performance comparison: {perf_stats}")
        
        # Log performance summary
        qdrant_time = perf_stats.get('avg_qdrant_time_ms', 0)
        memory_time = perf_stats.get('avg_memory_time_ms', 0)
        
        if qdrant_time > 0 and memory_time > 0:
            if qdrant_time < memory_time:
                speedup = memory_time / qdrant_time
                logger.info(f"🚀 Qdrant is {speedup:.2f}x faster ({qdrant_time:.2f}ms vs {memory_time:.2f}ms)")
            else:
                slowdown = qdrant_time / memory_time
                logger.info(f"🐌 In-memory is {slowdown:.2f}x faster ({memory_time:.2f}ms vs {qdrant_time:.2f}ms)")
        
    except Exception as e:
        logger.error(f"❌ Search performance test failed: {e}")


def main():
    """Run all Qdrant integration tests."""
    logger.info("🚀 Starting Qdrant Integration Tests")
    
    # Test 1: Qdrant Health
    qdrant_healthy = test_qdrant_health()
    
    # Test 2: RecommendationEngine Initialization
    engine = test_recommendation_engine_initialization()
    
    # Test 3: Load Move Candidates
    move_candidates = test_move_candidates_loading()
    
    # Test 4: Populate Qdrant
    if qdrant_healthy and engine:
        population_success = test_qdrant_population(engine, move_candidates)
        
        # Test 5: Performance Comparison
        if population_success:
            test_qdrant_vs_memory_search(engine, move_candidates)
    
    logger.info("🏁 Qdrant Integration Tests Complete")
    
    # Final summary
    logger.info("=== Test Summary ===")
    logger.info(f"Qdrant Health: {'✅' if qdrant_healthy else '❌'}")
    logger.info(f"Engine Init: {'✅' if engine else '❌'}")
    logger.info(f"Move Candidates: {'✅' if move_candidates else '❌'} ({len(move_candidates)} loaded)")
    
    if engine:
        final_stats = engine.get_performance_comparison()
        logger.info(f"Final Performance Stats: {final_stats}")


if __name__ == "__main__":
    main()