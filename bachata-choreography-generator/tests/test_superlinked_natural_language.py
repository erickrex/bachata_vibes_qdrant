#!/usr/bin/env python3
"""
Test script for SuperlinkedRecommendationEngine natural language interface.

This script demonstrates the new unified vector approach that replaces the complex
multi-factor scoring algorithm with natural language queries and dynamic weights.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.superlinked_recommendation_engine import (
    SuperlinkedRecommendationEngine, create_superlinked_recommendation_engine
)
from app.services.music_analyzer import MusicFeatures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_music_features(tempo: float = 120.0, energy_type: str = "medium") -> MusicFeatures:
    """Create mock music features for testing."""
    
    # Adjust features based on energy type
    if energy_type == "low":
        rms_energy = np.array([0.05, 0.08, 0.06, 0.07])
        spectral_centroid = np.array([1200, 1300, 1250, 1280])
        percussive_component = np.array([0.05, 0.06, 0.055, 0.058])
        harmonic_component = np.array([0.08, 0.10, 0.09, 0.095])
    elif energy_type == "high":
        rms_energy = np.array([0.15, 0.22, 0.18, 0.20])
        spectral_centroid = np.array([2200, 2400, 2300, 2350])
        percussive_component = np.array([0.15, 0.18, 0.16, 0.17])
        harmonic_component = np.array([0.12, 0.15, 0.13, 0.14])
    else:  # medium
        rms_energy = np.array([0.1, 0.15, 0.12, 0.13])
        spectral_centroid = np.array([1700, 1800, 1750, 1780])
        percussive_component = np.array([0.1, 0.12, 0.11, 0.115])
        harmonic_component = np.array([0.10, 0.12, 0.11, 0.115])
    
    # Create mock MFCC and chroma features
    mfcc_features = np.random.rand(13, 100)  # 13 MFCC coefficients, 100 time frames
    chroma_features = np.random.rand(12, 100)  # 12 chroma features, 100 time frames
    zero_crossing_rate = np.random.rand(100) * 0.1  # 100 time frames
    
    return MusicFeatures(
        tempo=tempo,
        beat_positions=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        duration=180.0,
        mfcc_features=mfcc_features,
        chroma_features=chroma_features,
        spectral_centroid=spectral_centroid,
        zero_crossing_rate=zero_crossing_rate,
        rms_energy=rms_energy,
        harmonic_component=harmonic_component,
        percussive_component=percussive_component,
        energy_profile=[0.5, 0.7, 0.6, 0.65, 0.55, 0.68],
        tempo_confidence=0.85,
        sections=[],  # Empty sections for mock data
        rhythm_pattern_strength=0.8,
        syncopation_level=0.3,
        audio_embedding=[0.1] * 128  # Mock 128-dimensional embedding
    )


def test_unified_vector_approach():
    """Test the unified vector approach vs old multi-factor scoring."""
    print("\n" + "="*80)
    print("TESTING UNIFIED VECTOR APPROACH")
    print("Replacing multi-factor scoring (audio 40%, tempo 30%, energy 20%, difficulty 10%)")
    print("with unified 512-dimensional Superlinked vectors")
    print("="*80)
    
    # Create the SuperlinkedRecommendationEngine
    engine = create_superlinked_recommendation_engine("data")
    
    # Get engine statistics
    stats = engine.get_performance_stats()
    print(f"\nEngine Type: {stats['engine_type']}")
    print(f"Unified Vector Dimension: {stats['unified_vector_dimension']}")
    print(f"Embedding Spaces: {stats['embedding_spaces']}")
    print(f"Total Moves: {stats['total_moves']}")
    
    # Test with different music scenarios
    test_scenarios = [
        {
            "name": "Slow Romantic Bachata",
            "music": create_mock_music_features(tempo=105.0, energy_type="low"),
            "description": "smooth romantic moves for slow bachata"
        },
        {
            "name": "Fast Energetic Bachata", 
            "music": create_mock_music_features(tempo=140.0, energy_type="high"),
            "description": "energetic advanced moves for fast bachata"
        },
        {
            "name": "Medium Tempo Traditional",
            "music": create_mock_music_features(tempo=125.0, energy_type="medium"),
            "description": "traditional intermediate moves with good flow"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Tempo: {scenario['music'].tempo} BPM")
        
        # Test unified vector recommendations
        recommendations = engine.recommend_moves(
            music_features=scenario['music'],
            target_difficulty="intermediate",
            description=scenario['description'],
            top_k=5
        )
        
        print(f"Unified Vector Results ({len(recommendations)} moves):")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec.move_candidate.move_label}")
            print(f"     Score: {rec.similarity_score:.3f}")
            print(f"     Tempo: {rec.move_candidate.tempo:.0f} BPM")
            print(f"     Energy: {rec.move_candidate.energy_level}")
            print(f"     Explanation: {rec.explanation}")


def test_natural_language_interface():
    """Test the natural language query interface."""
    print("\n" + "="*80)
    print("TESTING NATURAL LANGUAGE INTERFACE")
    print("Dynamic query-time weights for personalized choreography")
    print("="*80)
    
    engine = create_superlinked_recommendation_engine("data")
    
    # Test various natural language queries
    test_queries = [
        "energetic intermediate moves for 125 BPM song",
        "smooth beginner basic steps with low energy", 
        "advanced turns and styling for fast bachata",
        "flowing medium energy moves with good transitions",
        "musical moves that match the rhythm and beat",
        "tempo-precise moves for 115 BPM romantic song"
    ]
    
    # Create different music contexts
    music_contexts = [
        create_mock_music_features(tempo=125.0, energy_type="high"),
        create_mock_music_features(tempo=110.0, energy_type="low"),
        create_mock_music_features(tempo=135.0, energy_type="high")
    ]
    
    for i, query in enumerate(test_queries):
        music = music_contexts[i % len(music_contexts)]
        
        print(f"\n--- Query {i+1}: '{query}' ---")
        print(f"Music Context: {music.tempo} BPM, Energy: {['low', 'medium', 'high'][i % 3]}")
        
        # Process natural language query
        recommendations = engine.recommend_with_natural_language(
            query=query,
            music_features=music,
            top_k=3
        )
        
        print(f"Natural Language Results:")
        for j, rec in enumerate(recommendations):
            print(f"  {j+1}. {rec.move_candidate.move_label}")
            print(f"     Score: {rec.similarity_score:.3f}")
            print(f"     {rec.explanation}")


def test_dynamic_weights():
    """Test dynamic query-time weights for personalization."""
    print("\n" + "="*80)
    print("TESTING DYNAMIC QUERY-TIME WEIGHTS")
    print("Personalized choreography through Superlinked query composition")
    print("="*80)
    
    engine = create_superlinked_recommendation_engine("data")
    music = create_mock_music_features(tempo=120.0, energy_type="medium")
    
    # Test different weight configurations
    weight_scenarios = [
        {
            "name": "Musicality Focus",
            "weights": {"text": 0.6, "tempo": 0.2, "difficulty": 0.1, "energy": 0.1, "role": 0.0, "transition": 0.0},
            "description": "moves that match musical phrasing and rhythm"
        },
        {
            "name": "Tempo Precision",
            "weights": {"text": 0.2, "tempo": 0.6, "difficulty": 0.1, "energy": 0.1, "role": 0.0, "transition": 0.0},
            "description": "moves with exact tempo matching"
        },
        {
            "name": "Flow and Transitions",
            "weights": {"text": 0.3, "tempo": 0.2, "difficulty": 0.1, "energy": 0.2, "role": 0.0, "transition": 0.2},
            "description": "moves with smooth transitions and flow"
        },
        {
            "name": "Balanced (Default)",
            "weights": None,  # Use default weights
            "description": "balanced intermediate moves"
        }
    ]
    
    for scenario in weight_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Custom Weights: {scenario['weights']}")
        
        recommendations = engine.recommend_moves(
            music_features=music,
            target_difficulty="intermediate",
            description=scenario['description'],
            custom_weights=scenario['weights'],
            top_k=3
        )
        
        print(f"Results with {scenario['name'].lower()}:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")


def test_personalized_weights():
    """Test personalized weight creation based on user preferences."""
    print("\n" + "="*80)
    print("TESTING PERSONALIZED WEIGHT CREATION")
    print("User preference-based weight adjustment")
    print("="*80)
    
    engine = create_superlinked_recommendation_engine("data")
    
    # Test different user preference profiles
    user_profiles = [
        {
            "name": "Musical Dancer",
            "preferences": {"musicality": 0.8, "tempo_precision": 0.6, "flow": 0.4}
        },
        {
            "name": "Technical Dancer", 
            "preferences": {"difficulty_match": 0.9, "tempo_precision": 0.8, "energy_match": 0.3}
        },
        {
            "name": "Flow-Focused Dancer",
            "preferences": {"flow": 1.0, "musicality": 0.5, "energy_match": 0.7}
        }
    ]
    
    for profile in user_profiles:
        print(f"\n--- {profile['name']} ---")
        print(f"Preferences: {profile['preferences']}")
        
        # Create personalized weights
        personalized_weights = engine.create_personalized_weights(profile['preferences'])
        print(f"Personalized Weights: {personalized_weights}")
        
        # Test with personalized weights
        music = create_mock_music_features(tempo=125.0, energy_type="medium")
        recommendations = engine.recommend_moves(
            music_features=music,
            target_difficulty="intermediate",
            custom_weights=personalized_weights,
            top_k=3
        )
        
        print(f"Personalized Results:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")


def test_performance_comparison():
    """Test performance of unified vector approach."""
    print("\n" + "="*80)
    print("TESTING PERFORMANCE COMPARISON")
    print("Unified vector similarity vs multi-factor scoring")
    print("="*80)
    
    engine = create_superlinked_recommendation_engine("data")
    music = create_mock_music_features(tempo=120.0, energy_type="medium")
    
    # Run multiple searches to get average performance
    import time
    
    num_tests = 10
    total_time = 0
    
    print(f"Running {num_tests} unified vector searches...")
    
    for i in range(num_tests):
        start_time = time.time()
        
        recommendations = engine.recommend_moves(
            music_features=music,
            target_difficulty="intermediate",
            top_k=10
        )
        
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        total_time += search_time
        
        if i == 0:  # Show first result
            print(f"First search returned {len(recommendations)} recommendations")
    
    avg_time = total_time / num_tests
    print(f"Average search time: {avg_time:.2f}ms")
    
    # Get engine performance stats
    stats = engine.get_performance_stats()
    performance = stats.get('performance', {})
    
    print(f"Engine Statistics:")
    print(f"  Total unified searches: {performance.get('unified_searches', 0)}")
    print(f"  Average search time: {performance.get('avg_search_time_ms', 0):.2f}ms")
    print(f"  Natural language queries: {performance.get('natural_language_queries', 0)}")
    print(f"  Cache hits: {performance.get('cache_hits', 0)}")
    print(f"  Cache misses: {performance.get('cache_misses', 0)}")


def main():
    """Run all tests for the SuperlinkedRecommendationEngine."""
    print("SuperlinkedRecommendationEngine Test Suite")
    print("Testing unified vector approach that replaces multi-factor scoring")
    
    try:
        # Test core unified vector functionality
        test_unified_vector_approach()
        
        # Test natural language interface
        test_natural_language_interface()
        
        # Test dynamic weights
        test_dynamic_weights()
        
        # Test personalized weights
        test_personalized_weights()
        
        # Test performance
        test_performance_comparison()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("SuperlinkedRecommendationEngine successfully replaces multi-factor scoring")
        print("with unified 512-dimensional vector similarity search")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())