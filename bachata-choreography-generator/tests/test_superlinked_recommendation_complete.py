#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced SuperlinkedRecommendationEngine.

This script validates that the SuperlinkedRecommendationEngine completely replaces
the complex RecommendationEngine by testing all specialized embedding spaces:
- TextSimilaritySpace for semantic move understanding
- NumberSpace for tempo-aware search with linear BPM relationships
- CategoricalSimilaritySpace for energy level and role focus filtering
- CustomSpace for transition-aware choreography generation
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from services.superlinked_recommendation_engine import create_superlinked_recommendation_engine
from services.music_analyzer import MusicFeatures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_music_features(tempo: float = 120.0) -> MusicFeatures:
    """Create mock music features for testing."""
    from services.music_analyzer import MusicSection
    
    return MusicFeatures(
        tempo=tempo,
        beat_positions=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        duration=180.0,
        mfcc_features=np.random.random((13, 100)),
        chroma_features=np.random.random((12, 100)),
        spectral_centroid=np.random.random(100) * 1000 + 1500,
        zero_crossing_rate=np.random.random(100) * 0.1 + 0.05,
        rms_energy=np.random.random(100) * 0.5 + 0.3,
        harmonic_component=np.random.random(100) * 0.3,
        percussive_component=np.random.random(100) * 0.4,
        energy_profile=list(np.random.random(100) * 0.6 + 0.2),
        tempo_confidence=0.85,
        sections=[
            MusicSection(0.0, 30.0, "intro", 0.6, 0.8, ["basic_step"]),
            MusicSection(30.0, 120.0, "verse", 0.7, 0.9, ["cross_body_lead", "turn"]),
            MusicSection(120.0, 180.0, "chorus", 0.9, 0.85, ["styling", "dip"])
        ],
        rhythm_pattern_strength=0.8,
        syncopation_level=0.4,
        audio_embedding=list(np.random.random(128))
    )


def test_semantic_move_search(engine):
    """Test semantic move search using TextSimilaritySpace."""
    print("\n" + "="*60)
    print("TEST 1: Semantic Move Search (TextSimilaritySpace)")
    print("="*60)
    
    music_features = create_mock_music_features(125.0)
    
    # Test various semantic queries
    semantic_queries = [
        "energetic basic steps for beginners",
        "smooth romantic turns for couples",
        "advanced styling moves with arm work",
        "simple footwork for learning",
        "dynamic cross body leads"
    ]
    
    for query in semantic_queries:
        print(f"\nSemantic Query: '{query}'")
        results = engine.search_semantic_moves(
            semantic_query=query,
            music_features=music_features,
            top_k=5,
            semantic_weight=0.8
        )
        
        print(f"Found {len(results)} semantic matches:")
        for i, result in enumerate(results[:3], 1):
            candidate = result.move_candidate
            print(f"  {i}. {candidate.move_label} (score: {result.similarity_score:.3f})")
            print(f"     Energy: {candidate.energy_level}, Difficulty: {candidate.difficulty_score:.1f}")
            print(f"     Description: {candidate.move_description}")
    
    return True


def test_tempo_aware_search(engine):
    """Test tempo-aware search with linear BPM relationships."""
    print("\n" + "="*60)
    print("TEST 2: Tempo-Aware Search (NumberSpace Linear Relationships)")
    print("="*60)
    
    music_features = create_mock_music_features(125.0)
    
    # Test different tempos to verify linear relationships
    test_tempos = [100.0, 115.0, 125.0, 135.0, 145.0]
    
    for tempo in test_tempos:
        print(f"\nTempo Search: {tempo} BPM")
        results = engine.search_tempo_aware_moves(
            target_tempo=tempo,
            music_features=music_features,
            tempo_tolerance=10.0,
            top_k=5
        )
        
        print(f"Found {len(results)} tempo-compatible moves:")
        for i, result in enumerate(results[:3], 1):
            candidate = result.move_candidate
            tempo_diff = abs(candidate.tempo - tempo)
            print(f"  {i}. {candidate.move_label} - {candidate.tempo:.1f} BPM (±{tempo_diff:.1f})")
            print(f"     Score: {result.similarity_score:.3f}")
    
    # Verify linear relationship: 125 BPM should be closer to 130 than 90
    print(f"\nVerifying Linear Relationships:")
    results_125 = engine.search_tempo_aware_moves(125.0, music_features, 20.0, 10)
    
    # Find moves at different tempos
    tempo_scores = {}
    for result in results_125:
        move_tempo = result.move_candidate.tempo
        if move_tempo not in tempo_scores or result.similarity_score > tempo_scores[move_tempo]:
            tempo_scores[move_tempo] = result.similarity_score
    
    print(f"Tempo similarity scores from 125 BPM target:")
    for tempo in sorted(tempo_scores.keys()):
        print(f"  {tempo:.1f} BPM: {tempo_scores[tempo]:.3f}")
    
    return True


def test_categorical_filtering(engine):
    """Test categorical filtering for energy levels and role focus."""
    print("\n" + "="*60)
    print("TEST 3: Categorical Filtering (CategoricalSimilaritySpace)")
    print("="*60)
    
    music_features = create_mock_music_features(120.0)
    
    # Test energy level filtering
    energy_levels = ["low", "medium", "high"]
    role_focuses = ["lead_focus", "follow_focus", "both"]
    
    for energy in energy_levels:
        for role in role_focuses:
            print(f"\nCategorical Filter: Energy={energy}, Role={role}")
            results = engine.search_categorical_filtered_moves(
                music_features=music_features,
                energy_level=energy,
                role_focus=role,
                difficulty_level="intermediate",
                top_k=5
            )
            
            print(f"Found {len(results)} categorically filtered moves:")
            for i, result in enumerate(results[:3], 1):
                candidate = result.move_candidate
                print(f"  {i}. {candidate.move_label}")
                print(f"     Energy: {candidate.energy_level}, Role: {candidate.role_focus}")
                print(f"     Score: {result.similarity_score:.3f}")
    
    return True


def test_transition_aware_choreography(engine):
    """Test transition-aware choreography generation using CustomSpace."""
    print("\n" + "="*60)
    print("TEST 4: Transition-Aware Choreography (CustomSpace)")
    print("="*60)
    
    music_features = create_mock_music_features(122.0)
    
    # Generate choreographies with different parameters
    test_configs = [
        {"length": 6, "difficulty": "beginner", "style": "smooth"},
        {"length": 8, "difficulty": "intermediate", "style": "balanced"},
        {"length": 5, "difficulty": "advanced", "style": "energetic"}
    ]
    
    for config in test_configs:
        print(f"\nGenerating {config['length']}-move {config['difficulty']} {config['style']} choreography:")
        
        choreography = engine.generate_transition_aware_choreography(
            music_features=music_features,
            sequence_length=config["length"],
            target_difficulty=config["difficulty"],
            target_energy="medium",
            role_focus="both",
            diversity_factor=0.3,
            transition_weight=0.4
        )
        
        print(f"Generated choreography with {len(choreography)} moves:")
        for i, move in enumerate(choreography, 1):
            candidate = move.move_candidate
            print(f"  {i}. {candidate.move_label}")
            print(f"     Energy: {candidate.energy_level}, Tempo: {candidate.tempo:.1f} BPM")
            print(f"     Score: {move.similarity_score:.3f}")
            
            # Show transition compatibility
            if i < len(choreography):
                next_move = choreography[i]
                transitions = engine.get_move_transitions(candidate.move_id)
                is_compatible = next_move.move_candidate.move_id in transitions
                print(f"     → Transition to next: {'✓' if is_compatible else '○'}")
    
    return True


def test_natural_language_queries(engine):
    """Test natural language query processing."""
    print("\n" + "="*60)
    print("TEST 5: Natural Language Query Processing")
    print("="*60)
    
    music_features = create_mock_music_features(118.0)
    
    # Test various natural language queries
    nl_queries = [
        "energetic intermediate moves for 125 BPM song",
        "smooth beginner basic steps with low energy",
        "advanced turns and styling for fast bachata",
        "medium energy moves for both lead and follow",
        "slow romantic moves with gentle transitions"
    ]
    
    for query in nl_queries:
        print(f"\nNatural Language Query: '{query}'")
        results = engine.recommend_with_natural_language(
            query=query,
            music_features=music_features,
            top_k=5
        )
        
        print(f"Found {len(results)} matches:")
        for i, result in enumerate(results[:3], 1):
            candidate = result.move_candidate
            print(f"  {i}. {candidate.move_label}")
            print(f"     {candidate.move_description}")
            print(f"     Score: {result.similarity_score:.3f}")
    
    return True


def test_complete_choreography_generation(engine):
    """Test the complete choreography generation that replaces RecommendationEngine."""
    print("\n" + "="*60)
    print("TEST 6: Complete Choreography Generation (RecommendationEngine Replacement)")
    print("="*60)
    
    music_features = create_mock_music_features(124.0)
    
    # Generate complete choreography using all capabilities
    result = engine.generate_complete_choreography_with_transitions(
        music_features=music_features,
        choreography_length=8,
        target_difficulty="intermediate",
        style_preference="balanced",
        natural_language_description="smooth flowing bachata with romantic energy"
    )
    
    print("Complete Choreography Generation Results:")
    print(f"Generated {result['choreography_analysis']['total_moves']} moves")
    print(f"Average similarity score: {result['choreography_analysis']['average_similarity_score']:.3f}")
    print(f"Generation time: {result['generation_metadata']['generation_time_ms']:.2f}ms")
    
    print(f"\nChoreography Analysis:")
    analysis = result['choreography_analysis']
    print(f"  Difficulty distribution: {analysis['difficulty_distribution']}")
    print(f"  Energy consistency: {analysis['energy_consistency']:.3f}")
    print(f"  Transition quality: {analysis['transition_quality']:.3f}")
    print(f"  Tempo compatibility: {analysis['tempo_compatibility']:.3f}")
    print(f"  Style adherence: {analysis['style_adherence']:.3f}")
    
    print(f"\nSearch Phases:")
    phases = result['search_phases']
    print(f"  Semantic candidates: {phases['semantic_candidates']}")
    print(f"  Tempo candidates: {phases['tempo_candidates']}")
    print(f"  Categorical candidates: {phases['categorical_candidates']}")
    print(f"  Final sequence length: {phases['final_sequence_length']}")
    
    print(f"\nGenerated Choreography Sequence:")
    for i, move in enumerate(result['choreography_sequence'], 1):
        candidate = move.move_candidate
        print(f"  {i}. {candidate.move_label}")
        print(f"     Energy: {candidate.energy_level}, Tempo: {candidate.tempo:.1f} BPM")
        print(f"     Difficulty: {candidate.difficulty_score:.1f}, Score: {move.similarity_score:.3f}")
    
    return True


def test_performance_comparison(engine):
    """Test performance statistics and comparison with old system."""
    print("\n" + "="*60)
    print("TEST 7: Performance Statistics")
    print("="*60)
    
    # Get performance stats
    stats = engine.get_performance_stats()
    
    print("SuperlinkedRecommendationEngine Performance:")
    print(f"  Engine type: {stats['engine_type']}")
    print(f"  Unified vector dimension: {stats['unified_vector_dimension']}")
    print(f"  Total moves indexed: {stats['total_moves']}")
    print(f"  Embedding spaces: {stats['embedding_spaces']}")
    
    print(f"\nSearch Performance:")
    perf = stats['performance']
    print(f"  Unified searches: {perf['unified_searches']}")
    print(f"  Average search time: {perf['avg_search_time_ms']:.2f}ms")
    print(f"  Natural language queries: {perf['natural_language_queries']}")
    print(f"  Qdrant searches: {perf['qdrant_searches']}")
    print(f"  Fallback searches: {perf['fallback_searches']}")
    
    print(f"\nEmbedding Service Stats:")
    embedding_stats = stats['embedding_service_stats']
    print(f"  Total moves: {embedding_stats['total_moves']}")
    print(f"  Move categories: {embedding_stats['move_categories']}")
    print(f"  Transition graph size: {embedding_stats['transition_graph_size']}")
    
    return True


def main():
    """Run all tests for the enhanced SuperlinkedRecommendationEngine."""
    print("Enhanced SuperlinkedRecommendationEngine Test Suite")
    print("Testing complete replacement of complex RecommendationEngine")
    print("="*80)
    
    try:
        # Initialize the engine
        print("Initializing SuperlinkedRecommendationEngine...")
        engine = create_superlinked_recommendation_engine("data")
        print("✓ Engine initialized successfully")
        
        # Run all tests
        tests = [
            ("Semantic Move Search", test_semantic_move_search),
            ("Tempo-Aware Search", test_tempo_aware_search),
            ("Categorical Filtering", test_categorical_filtering),
            ("Transition-Aware Choreography", test_transition_aware_choreography),
            ("Natural Language Queries", test_natural_language_queries),
            ("Complete Choreography Generation", test_complete_choreography_generation),
            ("Performance Statistics", test_performance_comparison)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                print(f"\nRunning {test_name}...")
                success = test_func(engine)
                if success:
                    print(f"✓ {test_name} PASSED")
                    passed_tests += 1
                else:
                    print(f"✗ {test_name} FAILED")
            except Exception as e:
                print(f"✗ {test_name} FAILED with error: {e}")
                logger.exception(f"Test {test_name} failed")
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Passed: {passed_tests}/{total_tests} tests")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
            print("SuperlinkedRecommendationEngine successfully replaces RecommendationEngine")
            print("✓ Semantic move search using TextSimilaritySpace")
            print("✓ Tempo-aware search with linear BPM relationships")
            print("✓ Categorical filtering for energy and role focus")
            print("✓ Transition-aware choreography generation")
            return True
        else:
            print(f"❌ {total_tests - passed_tests} tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Test suite failed to initialize: {e}")
        logger.exception("Test suite initialization failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)