#!/usr/bin/env python3
"""
Test script to verify RecommendationEngine fallback functionality.
Tests both Qdrant and in-memory search paths.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.recommendation_engine import (
    RecommendationEngine, RecommendationRequest, MoveCandidate
)
from app.services.qdrant_service import QdrantConfig
from app.services.feature_fusion import FeatureFusion, MultiModalEmbedding
from app.services.music_analyzer import MusicFeatures
from app.services.move_analyzer import MoveAnalysisResult, MovementDynamics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_move_candidate(move_id: str, move_label: str, difficulty: str = "intermediate") -> MoveCandidate:
    """Create a dummy move candidate for testing."""
    
    # Create dummy movement dynamics
    movement_dynamics = MovementDynamics(
        velocity_profile=np.random.random(100),
        acceleration_profile=np.random.random(100),
        spatial_coverage=0.5,
        rhythm_score=0.8,
        complexity_score=0.6,
        dominant_movement_direction="forward",
        energy_level="medium",
        footwork_area_coverage=0.4,
        upper_body_movement_range=0.3,
        rhythm_compatibility_score=0.8,
        movement_periodicity=0.7,
        transition_points=[25, 50, 75],
        movement_intensity_profile=np.random.random(100),
        spatial_distribution={"upper": 0.3, "lower": 0.7}
    )
    
    # Create dummy analysis result
    analysis_result = MoveAnalysisResult(
        video_path=f"dummy/{move_label}.mp4",
        duration=8.0,
        frame_count=240,
        fps=30.0,
        pose_features=[],
        hand_features=[],
        movement_dynamics=movement_dynamics,
        pose_embedding=np.random.random(384),
        movement_embedding=np.random.random(128),
        movement_complexity_score=0.6,
        tempo_compatibility_range=(100, 140),
        difficulty_score=0.5 if difficulty == "intermediate" else (0.3 if difficulty == "beginner" else 0.8),
        analysis_quality=0.9,
        pose_detection_rate=0.95
    )
    
    # Create dummy multimodal embedding
    multimodal_embedding = MultiModalEmbedding(
        audio_embedding=np.random.random(128),
        pose_embedding=np.random.random(384),
        combined_embedding=np.random.random(512)
    )
    
    return MoveCandidate(
        move_id=move_id,
        video_path=f"dummy/{move_label}.mp4",
        move_label=move_label,
        analysis_result=analysis_result,
        multimodal_embedding=multimodal_embedding,
        energy_level="medium",
        difficulty=difficulty,
        estimated_tempo=120.0,
        lead_follow_roles="both"
    )


def create_dummy_music_features() -> MusicFeatures:
    """Create dummy music features for testing."""
    return MusicFeatures(
        tempo=120.0,
        beat_positions=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        duration=180.0,
        mfcc_features=np.random.random((13, 100)),
        chroma_features=np.random.random((12, 100)),
        spectral_centroid=np.random.random(100),
        zero_crossing_rate=np.random.random(100),
        rms_energy=np.random.random(100),
        harmonic_component=np.random.random(1000),
        percussive_component=np.random.random(1000),
        energy_profile=np.random.random(100).tolist(),
        tempo_confidence=0.9,
        sections=[],
        rhythm_pattern_strength=0.8,
        syncopation_level=0.3,
        audio_embedding=np.random.random(128).tolist()
    )


def test_recommendation_engine_fallback():
    """Test RecommendationEngine with both Qdrant and in-memory search."""
    logger.info("=== Testing RecommendationEngine Fallback ===")
    
    # Create dummy data
    move_candidates = [
        create_dummy_move_candidate("move_1", "basic_step", "beginner"),
        create_dummy_move_candidate("move_2", "cross_body_lead", "intermediate"),
        create_dummy_move_candidate("move_3", "lady_right_turn", "intermediate"),
        create_dummy_move_candidate("move_4", "dip", "advanced"),
        create_dummy_move_candidate("move_5", "body_roll", "advanced"),
    ]
    
    music_features = create_dummy_music_features()
    
    # Create music embedding
    feature_fusion = FeatureFusion()
    music_embedding = feature_fusion.create_multimodal_embedding(
        music_features, move_candidates[0].analysis_result
    )
    
    # Test 1: Initialize with Qdrant enabled (will fallback to mock)
    logger.info("Testing with Qdrant enabled (mock fallback)...")
    
    config = QdrantConfig.from_env()  # Use cloud configuration from environment
    engine = RecommendationEngine(use_qdrant=True, qdrant_config=config)
    
    logger.info(f"Qdrant available: {engine.is_qdrant_available()}")
    
    # Create recommendation request
    request = RecommendationRequest(
        music_features=music_features,
        music_embedding=music_embedding,
        target_difficulty="intermediate",
        target_energy="medium",
        tempo_tolerance=10.0
    )
    
    # Test recommendations (should use in-memory fallback)
    recommendations = engine.recommend_moves(request, move_candidates, top_k=3)
    
    logger.info(f"✅ Generated {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations):
        logger.info(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.overall_score:.3f})")
    
    # Test 2: Initialize with Qdrant disabled
    logger.info("Testing with Qdrant disabled...")
    
    engine_no_qdrant = RecommendationEngine(use_qdrant=False)
    
    recommendations_no_qdrant = engine_no_qdrant.recommend_moves(request, move_candidates, top_k=3)
    
    logger.info(f"✅ Generated {len(recommendations_no_qdrant)} recommendations (no Qdrant)")
    for i, rec in enumerate(recommendations_no_qdrant):
        logger.info(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.overall_score:.3f})")
    
    # Test 3: Performance comparison
    logger.info("Testing performance comparison...")
    
    perf_stats = engine.get_performance_comparison()
    logger.info(f"Performance stats: {perf_stats}")
    
    perf_stats_no_qdrant = engine_no_qdrant.get_performance_comparison()
    logger.info(f"Performance stats (no Qdrant): {perf_stats_no_qdrant}")
    
    # Test 4: Test different difficulty levels
    logger.info("Testing different difficulty levels...")
    
    for difficulty in ["beginner", "intermediate", "advanced"]:
        request_diff = RecommendationRequest(
            music_features=music_features,
            music_embedding=music_embedding,
            target_difficulty=difficulty,
            target_energy="medium",
            tempo_tolerance=10.0
        )
        
        recs = engine.recommend_moves(request_diff, move_candidates, top_k=2)
        logger.info(f"  {difficulty}: {[r.move_candidate.move_label for r in recs]}")
    
    logger.info("✅ All fallback tests completed successfully!")


def main():
    """Run the recommendation engine fallback tests."""
    logger.info("🚀 Starting RecommendationEngine Fallback Tests")
    
    try:
        test_recommendation_engine_fallback()
        logger.info("🏁 All tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()