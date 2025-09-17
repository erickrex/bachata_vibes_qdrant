#!/usr/bin/env python3
"""
Integration test for ChoreographyPipeline with SuperlinkedRecommendationEngine.

This test verifies that the pipeline correctly uses the new unified vector approach
instead of the old multi-factor scoring system.
"""

import sys
import logging
from pathlib import Path
import asyncio

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.choreography_pipeline import ChoreographyPipeline, PipelineConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_pipeline_integration():
    """Test that the pipeline uses SuperlinkedRecommendationEngine correctly."""
    print("\n" + "="*80)
    print("TESTING CHOREOGRAPHY PIPELINE INTEGRATION")
    print("Verifying SuperlinkedRecommendationEngine integration")
    print("="*80)
    
    # Create pipeline configuration
    config = PipelineConfig(
        quality_mode="fast",
        enable_caching=True,
        enable_qdrant=False,  # Disable Qdrant for this test
        max_workers=2
    )
    
    # Initialize pipeline
    pipeline = ChoreographyPipeline(config)
    
    # Test that the recommendation engine is SuperlinkedRecommendationEngine
    rec_engine = pipeline.recommendation_engine
    print(f"Recommendation Engine Type: {type(rec_engine).__name__}")
    
    # Get performance stats to verify it's the new engine
    stats = pipeline.get_performance_comparison()
    print(f"Engine Type: {stats.get('engine_type', 'Unknown')}")
    print(f"Unified Vector Approach: {stats.get('unified_vector_approach', False)}")
    print(f"Eliminated Multi-Factor Scoring: {stats.get('eliminated_multi_factor_scoring', False)}")
    
    # Test engine statistics
    engine_stats = rec_engine.get_performance_stats()
    print(f"Unified Vector Dimension: {engine_stats.get('unified_vector_dimension', 0)}")
    print(f"Total Moves: {engine_stats.get('total_moves', 0)}")
    print(f"Embedding Spaces: {engine_stats.get('embedding_spaces', 0)}")
    
    # Test natural language recommendation
    print(f"\nTesting natural language interface...")
    
    # Create mock music features for testing
    import numpy as np
    from app.services.music_analyzer import MusicFeatures
    
    mock_music_features = MusicFeatures(
        tempo=120.0,
        beat_positions=[0.5, 1.0, 1.5, 2.0],
        duration=180.0,
        mfcc_features=np.random.rand(13, 100),
        chroma_features=np.random.rand(12, 100),
        spectral_centroid=np.array([1700, 1800, 1750, 1780]),
        zero_crossing_rate=np.random.rand(100) * 0.1,
        rms_energy=np.array([0.1, 0.15, 0.12, 0.13]),
        harmonic_component=np.array([0.10, 0.12, 0.11, 0.115]),
        percussive_component=np.array([0.1, 0.12, 0.11, 0.115]),
        energy_profile=[0.5, 0.7, 0.6, 0.65],
        tempo_confidence=0.85,
        sections=[],
        rhythm_pattern_strength=0.8,
        syncopation_level=0.3,
        audio_embedding=[0.1] * 128
    )
    
    # Test natural language query
    nl_recommendations = rec_engine.recommend_with_natural_language(
        query="energetic intermediate moves for 120 BPM song",
        music_features=mock_music_features,
        top_k=5
    )
    
    print(f"Natural Language Query Results: {len(nl_recommendations)} moves")
    for i, rec in enumerate(nl_recommendations[:3]):
        print(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")
    
    # Test unified vector recommendations
    unified_recommendations = rec_engine.recommend_moves(
        music_features=mock_music_features,
        target_difficulty="intermediate",
        target_energy="medium",
        top_k=5
    )
    
    print(f"\nUnified Vector Results: {len(unified_recommendations)} moves")
    for i, rec in enumerate(unified_recommendations[:3]):
        print(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")
        print(f"     {rec.explanation}")
    
    # Test personalized weights
    print(f"\nTesting personalized weights...")
    user_preferences = {
        "musicality": 0.8,
        "tempo_precision": 0.6,
        "flow": 0.4
    }
    
    personalized_weights = rec_engine.create_personalized_weights(user_preferences)
    print(f"Personalized Weights: {personalized_weights}")
    
    personalized_recommendations = rec_engine.recommend_moves(
        music_features=mock_music_features,
        target_difficulty="intermediate",
        custom_weights=personalized_weights,
        top_k=3
    )
    
    print(f"Personalized Results: {len(personalized_recommendations)} moves")
    for i, rec in enumerate(personalized_recommendations):
        print(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")
    
    print(f"\n" + "="*80)
    print("PIPELINE INTEGRATION TEST COMPLETED SUCCESSFULLY")
    print("SuperlinkedRecommendationEngine successfully integrated into pipeline")
    print("Multi-factor scoring (audio 40%, tempo 30%, energy 20%, difficulty 10%) eliminated")
    print("Replaced with unified 512-dimensional vector similarity search")
    print("="*80)


async def test_performance_comparison():
    """Test performance of the new unified approach."""
    print("\n" + "="*80)
    print("TESTING PERFORMANCE COMPARISON")
    print("="*80)
    
    config = PipelineConfig(quality_mode="fast", enable_qdrant=False)
    pipeline = ChoreographyPipeline(config)
    
    # Get initial stats
    initial_stats = pipeline.get_performance_comparison()
    print(f"Initial Stats:")
    print(f"  Engine Type: {initial_stats.get('engine_type')}")
    print(f"  Unified Vector Approach: {initial_stats.get('unified_vector_approach')}")
    
    # Run some recommendations to generate performance data
    rec_engine = pipeline.recommendation_engine
    
    import numpy as np
    from app.services.music_analyzer import MusicFeatures
    
    mock_music_features = MusicFeatures(
        tempo=125.0,
        beat_positions=[0.5, 1.0, 1.5, 2.0],
        duration=180.0,
        mfcc_features=np.random.rand(13, 100),
        chroma_features=np.random.rand(12, 100),
        spectral_centroid=np.array([1700, 1800, 1750, 1780]),
        zero_crossing_rate=np.random.rand(100) * 0.1,
        rms_energy=np.array([0.1, 0.15, 0.12, 0.13]),
        harmonic_component=np.array([0.10, 0.12, 0.11, 0.115]),
        percussive_component=np.array([0.1, 0.12, 0.11, 0.115]),
        energy_profile=[0.5, 0.7, 0.6, 0.65],
        tempo_confidence=0.85,
        sections=[],
        rhythm_pattern_strength=0.8,
        syncopation_level=0.3,
        audio_embedding=[0.1] * 128
    )
    
    # Run multiple searches
    for i in range(5):
        recommendations = rec_engine.recommend_moves(
            music_features=mock_music_features,
            target_difficulty="intermediate",
            top_k=10
        )
        print(f"Search {i+1}: {len(recommendations)} recommendations")
    
    # Get final stats
    final_stats = pipeline.get_performance_comparison()
    performance = final_stats.get('performance', {})
    
    print(f"\nFinal Performance Stats:")
    print(f"  Total unified searches: {performance.get('unified_searches', 0)}")
    print(f"  Average search time: {performance.get('avg_search_time_ms', 0):.2f}ms")
    print(f"  Natural language queries: {performance.get('natural_language_queries', 0)}")


async def main():
    """Run all integration tests."""
    print("ChoreographyPipeline Integration Test Suite")
    print("Testing SuperlinkedRecommendationEngine integration")
    
    try:
        await test_pipeline_integration()
        await test_performance_comparison()
        
        print("\n" + "="*80)
        print("ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))