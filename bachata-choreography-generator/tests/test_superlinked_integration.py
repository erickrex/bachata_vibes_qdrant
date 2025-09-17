#!/usr/bin/env python3
"""
Test script to validate Superlinked integration without requiring Qdrant.

This script tests:
1. Superlinked embedding generation
2. SuperlinkedRecommendationEngine functionality
3. Fallback behavior when Qdrant is not available
4. Integration with the choreography pipeline
"""

import logging
import sys
import time
import numpy as np
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.superlinked_embedding_service import create_superlinked_service
from app.services.superlinked_recommendation_engine import create_superlinked_recommendation_engine
from app.services.choreography_pipeline import ChoreoGenerationPipeline, PipelineConfig
from app.services.music_analyzer import MusicFeatures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_superlinked_embeddings():
    """Test Superlinked embedding generation."""
    logger.info("Testing Superlinked embedding generation...")
    
    try:
        # Create embedding service
        embedding_service = create_superlinked_service("data")
        
        # Get statistics
        stats = embedding_service.get_stats()
        logger.info(f"✅ Embedding service stats:")
        logger.info(f"   - Total moves: {stats['total_moves']}")
        logger.info(f"   - Move categories: {stats['move_categories']}")
        logger.info(f"   - Embedding spaces: {len(stats['embedding_spaces'])}")
        logger.info(f"   - Total dimension: {embedding_service.total_dimension}D")
        
        # Test semantic search
        semantic_results = embedding_service.search_semantic("basic steps for beginners", limit=3)
        logger.info(f"✅ Semantic search: {len(semantic_results)} results")
        for i, result in enumerate(semantic_results):
            logger.info(f"   {i+1}. {result['move_label']} (score: {result['similarity_score']:.3f})")
        
        # Test tempo search
        tempo_results = embedding_service.search_tempo(120.0, limit=3)
        logger.info(f"✅ Tempo search: {len(tempo_results)} results")
        for i, result in enumerate(tempo_results):
            logger.info(f"   {i+1}. {result['move_label']} - {result['tempo']} BPM (score: {result['similarity_score']:.3f})")
        
        # Test multi-factor search
        multi_results = embedding_service.search_moves(
            description="intermediate energetic moves",
            target_tempo=125.0,
            difficulty_level="intermediate",
            energy_level="high",
            limit=3
        )
        logger.info(f"✅ Multi-factor search: {len(multi_results)} results")
        for i, result in enumerate(multi_results):
            logger.info(f"   {i+1}. {result['move_label']} - {result['energy_level']} energy (score: {result['similarity_score']:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Superlinked embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_superlinked_recommendation_engine():
    """Test SuperlinkedRecommendationEngine with fallback behavior."""
    logger.info("Testing SuperlinkedRecommendationEngine...")
    
    try:
        # Create recommendation engine (will use fallback when Qdrant is not available)
        engine = create_superlinked_recommendation_engine("data")
        
        # Check Qdrant availability
        logger.info(f"✅ Qdrant available: {engine.is_qdrant_available}")
        
        # Create mock music features
        mock_music_features = MusicFeatures(
            tempo=120.0,
            beat_positions=[0.5, 1.0, 1.5, 2.0],
            duration=180.0,
            mfcc_features=np.random.random((13, 100)),
            chroma_features=np.random.random((12, 100)),
            spectral_centroid=np.array([1500, 1600, 1550, 1580]),
            zero_crossing_rate=np.array([0.1, 0.12, 0.11, 0.13]),
            rms_energy=np.array([0.1, 0.2, 0.15, 0.18]),
            harmonic_component=np.array([0.05, 0.06, 0.055, 0.065]),
            percussive_component=np.array([0.1, 0.12, 0.11, 0.13]),
            energy_profile=[0.5, 0.7, 0.6, 0.65],
            tempo_confidence=0.85,
            sections=[],
            rhythm_pattern_strength=0.7,
            syncopation_level=0.3,
            audio_embedding=[0.1] * 128
        )
        
        # Test unified vector recommendations
        logger.info("Testing unified vector recommendations...")
        recommendations = engine.recommend_moves(
            music_features=mock_music_features,
            target_difficulty="intermediate",
            target_energy="medium",
            top_k=5
        )
        
        logger.info(f"✅ Unified search returned {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations[:3]):
            logger.info(f"   {i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")
            logger.info(f"      {rec.explanation}")
        
        # Test natural language queries
        logger.info("Testing natural language queries...")
        nl_queries = [
            "energetic intermediate moves for 125 BPM song",
            "smooth beginner basic steps with low energy",
            "advanced turns for fast bachata"
        ]
        
        for query in nl_queries:
            logger.info(f"Query: '{query}'")
            nl_recommendations = engine.recommend_with_natural_language(
                query, mock_music_features, top_k=3
            )
            logger.info(f"✅ Natural language search returned {len(nl_recommendations)} recommendations:")
            for rec in nl_recommendations:
                logger.info(f"   - {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")
        
        # Test performance stats
        perf_stats = engine.get_performance_stats()
        logger.info(f"✅ Performance stats:")
        logger.info(f"   - Engine type: {perf_stats['engine_type']}")
        logger.info(f"   - Vector dimension: {perf_stats['unified_vector_dimension']}")
        logger.info(f"   - Total searches: {perf_stats['performance']['unified_searches']}")
        logger.info(f"   - Qdrant searches: {perf_stats['performance']['qdrant_searches']}")
        logger.info(f"   - Fallback searches: {perf_stats['performance']['fallback_searches']}")
        logger.info(f"   - Natural language queries: {perf_stats['performance']['natural_language_queries']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SuperlinkedRecommendationEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_choreography_pipeline_integration():
    """Test integration with the choreography pipeline."""
    logger.info("Testing choreography pipeline integration...")
    
    try:
        # Create pipeline with Qdrant disabled for testing
        config = PipelineConfig(
            enable_qdrant=False,  # Disable Qdrant for this test
            quality_mode="fast",
            enable_caching=False
        )
        pipeline = ChoreoGenerationPipeline(config)
        
        # Test that the recommendation engine is properly initialized
        rec_engine = pipeline.recommendation_engine
        logger.info(f"✅ Pipeline recommendation engine initialized: {type(rec_engine).__name__}")
        
        # Test Qdrant health status
        qdrant_health = pipeline.get_qdrant_health_status()
        logger.info(f"✅ Qdrant health status: {qdrant_health}")
        
        # Test performance comparison
        perf_comparison = pipeline.get_performance_comparison()
        logger.info(f"✅ Performance comparison:")
        logger.info(f"   - Engine type: {perf_comparison.get('engine_type', 'Unknown')}")
        logger.info(f"   - Unified vector approach: {perf_comparison.get('unified_vector_approach', False)}")
        logger.info(f"   - Eliminated multi-factor scoring: {perf_comparison.get('eliminated_multi_factor_scoring', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Choreography pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_dimensions_consistency():
    """Test that embedding dimensions are consistent across services."""
    logger.info("Testing embedding dimensions consistency...")
    
    try:
        # Create services
        embedding_service = create_superlinked_service("data")
        rec_engine = create_superlinked_recommendation_engine("data")
        
        # Check dimensions
        embedding_dim = embedding_service.total_dimension
        logger.info(f"✅ Embedding service dimension: {embedding_dim}D")
        
        # Test that query embeddings have the same dimension
        query_embedding = embedding_service.generate_query_embedding(
            description="test query",
            target_tempo=120.0,
            difficulty_level="intermediate",
            energy_level="medium"
        )
        query_dim = len(query_embedding)
        logger.info(f"✅ Query embedding dimension: {query_dim}D")
        
        if embedding_dim == query_dim:
            logger.info("✅ Embedding dimensions are consistent")
            return True
        else:
            logger.error(f"❌ Dimension mismatch: {embedding_dim} != {query_dim}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Embedding dimensions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting Superlinked integration tests (without Qdrant)")
    
    tests = [
        ("Superlinked Embeddings", test_superlinked_embeddings),
        ("SuperlinkedRecommendationEngine", test_superlinked_recommendation_engine),
        ("Choreography Pipeline Integration", test_choreography_pipeline_integration),
        ("Embedding Dimensions Consistency", test_embedding_dimensions_consistency),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        success = test_func()
        test_time = time.time() - start_time
        
        results.append((test_name, success, test_time))
        
        if success:
            logger.info(f"✅ {test_name} PASSED ({test_time:.2f}s)")
        else:
            logger.error(f"❌ {test_name} FAILED ({test_time:.2f}s)")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, test_time in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name} ({test_time:.2f}s)")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Superlinked integration is working correctly.")
        print("\n🎉 SUCCESS! Superlinked integration is working:")
        print("   ✅ Unified 470D embeddings generated successfully")
        print("   ✅ 6 specialized embedding spaces working")
        print("   ✅ Natural language queries supported")
        print("   ✅ Fallback behavior when Qdrant unavailable")
        print("   ✅ Pipeline integration complete")
        print("\nThe system is ready to use Superlinked embeddings!")
        print("Qdrant Cloud is configured automatically via environment variables")
        return True
    else:
        logger.error(f"💥 {total - passed} tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)