#!/usr/bin/env python3
"""
Test script for the diversity selection and choreography flow optimization system (Task 5.3).
Tests diversity selection algorithm, transition compatibility matrix, sequence optimization
using dynamic programming, and musical structure awareness.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.choreography_optimizer import (
    ChoreographyOptimizer, OptimizationRequest, TransitionScore, ChoreographySequence
)
from app.services.recommendation_engine import (
    RecommendationEngine, MoveCandidate, RecommendationRequest, RecommendationScore
)
from app.services.feature_fusion import FeatureFusion
from app.services.music_analyzer import MusicAnalyzer
from app.services.move_analyzer import MoveAnalyzer
from app.services.annotation_interface import AnnotationInterface


def test_choreography_optimizer_initialization():
    """Test choreography optimizer initialization and transition weights."""
    print("🚀 Testing Choreography Optimizer Initialization")
    print("-" * 55)
    
    optimizer = ChoreographyOptimizer()
    
    # Test transition weights
    expected_weights = {
        'pose_similarity': 0.25,
        'movement_flow': 0.35,
        'energy_continuity': 0.25,
        'difficulty_progression': 0.15
    }
    
    print(f"📊 Transition weights: {optimizer.transition_weights}")
    assert optimizer.transition_weights == expected_weights, f"Unexpected transition weights: {optimizer.transition_weights}"
    
    # Test weight sum
    weight_sum = sum(optimizer.transition_weights.values())
    print(f"📊 Weight sum: {weight_sum}")
    assert abs(weight_sum - 1.0) < 0.001, f"Weights don't sum to 1.0: {weight_sum}"
    
    # Test transition cache
    print(f"🗄️  Transition cache initialized: {len(optimizer.transition_cache)} entries")
    assert isinstance(optimizer.transition_cache, dict), "Transition cache should be a dictionary"
    
    print(f"✅ Choreography optimizer initialization test passed")
    return True


def test_transition_compatibility_calculation():
    """Test transition compatibility calculation between move pairs."""
    print("\n🔄 Testing Transition Compatibility Calculation")
    print("-" * 55)
    
    optimizer = ChoreographyOptimizer()
    engine = RecommendationEngine()
    music_analyzer = MusicAnalyzer()
    move_analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    
    try:
        # Get test files
        audio_files = list(Path("data/songs").glob("*.mp3"))
        if not audio_files:
            print("⚠️  No audio files found")
            return False
        
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:2]  # Use first 2 for speed
        
        if len(annotations) < 2:
            print("⚠️  Need at least 2 annotations for transition testing")
            return False
        
        # Analyze music
        test_audio = str(audio_files[0])
        music_features = music_analyzer.analyze_audio(test_audio)
        
        # Create move candidates
        candidates = []
        for i, annotation in enumerate(annotations):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            print(f"📹 Creating candidate {i+1}: {annotation.move_label}")
            
            # Analyze move
            move_result = move_analyzer.analyze_move_clip(video_path)
            
            # Create candidate
            candidate = engine.create_move_candidate(
                move_id=f"move_{i:03d}",
                video_path=video_path,
                move_label=annotation.move_label,
                analysis_result=move_result,
                music_features=music_features,
                energy_level=annotation.energy_level,
                difficulty="intermediate"
            )
            
            candidates.append(candidate)
        
        if len(candidates) < 2:
            print("⚠️  Could not create enough candidates")
            return False
        
        # Test transition compatibility calculation
        from_move = candidates[0]
        to_move = candidates[1]
        
        print(f"\n🔄 Testing transition: {from_move.move_label} → {to_move.move_label}")
        
        transition_score = optimizer._calculate_transition_compatibility(from_move, to_move)
        
        print(f"📊 Transition Results:")
        print(f"   Overall compatibility: {transition_score.compatibility_score:.3f}")
        print(f"   Pose similarity: {transition_score.pose_similarity:.3f}")
        print(f"   Movement flow: {transition_score.movement_flow:.3f}")
        print(f"   Energy continuity: {transition_score.energy_continuity:.3f}")
        print(f"   Difficulty progression: {transition_score.difficulty_progression:.3f}")
        print(f"   Is smooth: {transition_score.is_smooth}")
        print(f"   Requires pause: {transition_score.requires_pause}")
        print(f"   Transition time: {transition_score.transition_time:.1f}s")
        
        # Test score properties
        assert 0.0 <= transition_score.compatibility_score <= 1.0, f"Invalid compatibility score: {transition_score.compatibility_score}"
        assert 0.0 <= transition_score.pose_similarity <= 1.0, f"Invalid pose similarity: {transition_score.pose_similarity}"
        assert 0.0 <= transition_score.movement_flow <= 1.0, f"Invalid movement flow: {transition_score.movement_flow}"
        assert 0.0 <= transition_score.energy_continuity <= 1.0, f"Invalid energy continuity: {transition_score.energy_continuity}"
        assert 0.0 <= transition_score.difficulty_progression <= 1.0, f"Invalid difficulty progression: {transition_score.difficulty_progression}"
        assert transition_score.transition_time > 0.0, f"Invalid transition time: {transition_score.transition_time}"
        
        # Test transition characteristics logic
        if transition_score.compatibility_score > 0.7:
            assert transition_score.is_smooth, "High compatibility should be smooth"
        if transition_score.compatibility_score < 0.4:
            assert transition_score.requires_pause, "Low compatibility should require pause"
        
        print(f"✅ Transition compatibility calculation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Transition compatibility test failed: {e}")
        return False


def test_transition_matrix_building():
    """Test building transition compatibility matrix between all move pairs."""
    print("\n🗂️  Testing Transition Matrix Building")
    print("-" * 55)
    
    optimizer = ChoreographyOptimizer()
    engine = RecommendationEngine()
    music_analyzer = MusicAnalyzer()
    move_analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    fusion = FeatureFusion()
    
    try:
        # Get test files
        audio_files = list(Path("data/songs").glob("*.mp3"))
        if not audio_files:
            print("⚠️  No audio files found")
            return False
        
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:3]  # Use first 3 for speed
        
        if len(annotations) < 3:
            print("⚠️  Need at least 3 annotations for matrix testing")
            return False
        
        # Analyze music
        test_audio = str(audio_files[0])
        music_features = music_analyzer.analyze_audio(test_audio)
        music_embedding = fusion.create_multimodal_embedding(music_features, 
                                                           move_analyzer.analyze_move_clip(
                                                               os.path.join("data", annotations[0].video_path)
                                                           ))
        
        # Create recommendation request
        request = RecommendationRequest(
            music_features=music_features,
            music_embedding=music_embedding,
            target_difficulty="intermediate",
            target_energy="medium"
        )
        
        # Create move candidates and get recommendations
        candidates = []
        for i, annotation in enumerate(annotations):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            print(f"📹 Creating candidate {i+1}: {annotation.move_label}")
            
            # Analyze move
            move_result = move_analyzer.analyze_move_clip(video_path)
            
            # Create candidate
            candidate = engine.create_move_candidate(
                move_id=f"move_{i:03d}",
                video_path=video_path,
                move_label=annotation.move_label,
                analysis_result=move_result,
                music_features=music_features,
                energy_level=annotation.energy_level,
                difficulty="intermediate"
            )
            
            candidates.append(candidate)
        
        if len(candidates) < 3:
            print("⚠️  Could not create enough candidates")
            return False
        
        # Get recommendations
        recommendations = engine.recommend_moves(request, candidates, top_k=len(candidates))
        
        print(f"\n🗂️  Building transition matrix for {len(recommendations)} moves")
        
        # Build transition matrix
        transition_matrix = optimizer._build_transition_matrix(recommendations)
        
        print(f"📊 Transition Matrix Results:")
        print(f"   Total transitions: {len(transition_matrix)}")
        
        # Expected number of transitions: n * (n-1) for n moves
        expected_transitions = len(recommendations) * (len(recommendations) - 1)
        print(f"   Expected transitions: {expected_transitions}")
        assert len(transition_matrix) == expected_transitions, f"Wrong number of transitions: {len(transition_matrix)} vs {expected_transitions}"
        
        # Test matrix properties
        for (from_move, to_move), transition_score in transition_matrix.items():
            assert isinstance(from_move, str), f"From move should be string: {type(from_move)}"
            assert isinstance(to_move, str), f"To move should be string: {type(to_move)}"
            assert isinstance(transition_score, TransitionScore), f"Should be TransitionScore: {type(transition_score)}"
            assert from_move != to_move, f"Self-transition found: {from_move} -> {to_move}"
        
        # Get matrix summary
        summary = optimizer.get_transition_matrix_summary(transition_matrix)
        print(f"📊 Matrix Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        # Test summary properties
        assert summary['total_transitions'] == len(transition_matrix), "Summary count mismatch"
        assert 0.0 <= summary['mean_compatibility'] <= 1.0, f"Invalid mean compatibility: {summary['mean_compatibility']}"
        assert summary['smooth_transitions'] + summary['pause_transitions'] <= summary['total_transitions'], "Transition counts don't add up"
        
        print(f"✅ Transition matrix building test passed")
        return True
        
    except Exception as e:
        print(f"❌ Transition matrix building test failed: {e}")
        return False


def test_diversity_selection_algorithm():
    """Test diversity selection algorithm to avoid repetitive move sequences."""
    print("\n🎭 Testing Diversity Selection Algorithm")
    print("-" * 55)
    
    optimizer = ChoreographyOptimizer()
    
    # Create mock recommendation scores with different move types
    mock_moves = [
        # Basic moves (should be limited)
        type('MockScore', (), {
            'move_candidate': type('MockCandidate', (), {'move_label': 'basic_step_1'})(),
            'overall_score': 0.9
        })(),
        type('MockScore', (), {
            'move_candidate': type('MockCandidate', (), {'move_label': 'basic_step_2'})(),
            'overall_score': 0.8
        })(),
        type('MockScore', (), {
            'move_candidate': type('MockCandidate', (), {'move_label': 'basic_step_3'})(),
            'overall_score': 0.7
        })(),
        
        # Turn moves
        type('MockScore', (), {
            'move_candidate': type('MockCandidate', (), {'move_label': 'lady_right_turn_1'})(),
            'overall_score': 0.85
        })(),
        type('MockScore', (), {
            'move_candidate': type('MockCandidate', (), {'move_label': 'lady_left_turn_1'})(),
            'overall_score': 0.75
        })(),
        
        # Cross body moves
        type('MockScore', (), {
            'move_candidate': type('MockCandidate', (), {'move_label': 'cross_body_lead_1'})(),
            'overall_score': 0.8
        })(),
        
        # Styling moves
        type('MockScore', (), {
            'move_candidate': type('MockCandidate', (), {'move_label': 'arm_styling_1'})(),
            'overall_score': 0.7
        })(),
    ]
    
    print(f"🎭 Testing with {len(mock_moves)} mock moves")
    
    # Test diversity selection
    max_repetitions = 2
    min_diversity_threshold = 0.6
    
    diverse_moves = optimizer._apply_diversity_selection(
        mock_moves, max_repetitions, min_diversity_threshold
    )
    
    print(f"📊 Diversity Selection Results:")
    print(f"   Original moves: {len(mock_moves)}")
    print(f"   Selected moves: {len(diverse_moves)}")
    
    # Count move types in selection
    move_type_counts = {}
    for move in diverse_moves:
        move_type = optimizer._get_move_type(move.move_candidate.move_label)
        move_type_counts[move_type] = move_type_counts.get(move_type, 0) + 1
    
    print(f"   Move type distribution: {move_type_counts}")
    
    # Test constraints
    for move_type, count in move_type_counts.items():
        assert count <= max_repetitions, f"Move type {move_type} exceeds max repetitions: {count} > {max_repetitions}"
    
    # Test diversity score
    diversity_score = optimizer._calculate_diversity_score(diverse_moves)
    print(f"   Diversity score: {diversity_score:.3f}")
    assert diversity_score >= 0.0, f"Diversity score should be non-negative: {diversity_score}"
    
    # Test move type extraction
    test_cases = [
        ("basic_step_1", "basic"),
        ("lady_right_turn_1", "turn"),
        ("cross_body_lead_1", "cross_body"),
        ("arm_styling_1", "styling"),
        ("body_roll_1", "body_roll"),
        ("dip_1", "dip"),
        ("shadow_position_1", "shadow"),
        ("hammerlock_1", "hammerlock"),
        ("unknown_move", "other")
    ]
    
    print(f"\n🏷️  Testing move type extraction:")
    for move_label, expected_type in test_cases:
        actual_type = optimizer._get_move_type(move_label)
        print(f"   {move_label} → {actual_type}")
        assert actual_type == expected_type, f"Wrong move type for {move_label}: {actual_type} vs {expected_type}"
    
    print(f"✅ Diversity selection algorithm test passed")
    return True


def test_musical_structure_alignment():
    """Test musical structure awareness to align move complexity with song sections."""
    print("\n🎵 Testing Musical Structure Alignment")
    print("-" * 55)
    
    optimizer = ChoreographyOptimizer()
    music_analyzer = MusicAnalyzer()
    
    try:
        # Get test audio file
        audio_files = list(Path("data/songs").glob("*.mp3"))
        if not audio_files:
            print("⚠️  No audio files found")
            return False
        
        # Analyze music to get sections
        test_audio = str(audio_files[0])
        music_features = music_analyzer.analyze_audio(test_audio)
        
        print(f"🎵 Test audio: {Path(test_audio).name}")
        print(f"📊 Musical sections: {len(music_features.sections)}")
        
        # Create mock recommendation scores
        mock_candidates = [
            type('MockScore', (), {
                'move_candidate': type('MockCandidate', (), {
                    'move_label': 'basic_step',
                    'energy_level': 'low'
                })(),
                'overall_score': 0.8
            })(),
            type('MockScore', (), {
                'move_candidate': type('MockCandidate', (), {
                    'move_label': 'cross_body_lead',
                    'energy_level': 'medium'
                })(),
                'overall_score': 0.9
            })(),
            type('MockScore', (), {
                'move_candidate': type('MockCandidate', (), {
                    'move_label': 'lady_right_turn',
                    'energy_level': 'high'
                })(),
                'overall_score': 0.85
            })(),
            type('MockScore', (), {
                'move_candidate': type('MockCandidate', (), {
                    'move_label': 'arm_styling',
                    'energy_level': 'low'
                })(),
                'overall_score': 0.7
            })(),
        ]
        
        # Test musical structure alignment
        section_assignments = optimizer._align_with_musical_structure(
            music_features.sections, mock_candidates
        )
        
        print(f"📊 Section Alignment Results:")
        print(f"   Total sections: {len(section_assignments)}")
        
        for i, (section, assigned_moves) in enumerate(section_assignments):
            print(f"   Section {i+1} ({section.section_type}):")
            print(f"      Duration: {section.end_time - section.start_time:.1f}s")
            print(f"      Energy level: {section.energy_level:.2f}")
            print(f"      Assigned moves: {len(assigned_moves)}")
            print(f"      Recommended types: {section.recommended_move_types}")
            
            # Test that each section has some moves assigned
            assert len(assigned_moves) > 0, f"Section {i+1} has no assigned moves"
            
            # Test that moves are sorted by score
            scores = [move.overall_score for move in assigned_moves]
            assert scores == sorted(scores, reverse=True), f"Section {i+1} moves not sorted by score"
        
        # Test energy compatibility function
        test_energy_cases = [
            (0.3, 'low', True),      # Compatible
            (0.6, 'medium', True),   # Compatible
            (0.9, 'high', True),     # Compatible
            (0.1, 'high', False),    # Not compatible
            (0.9, 'low', False),     # Not compatible
        ]
        
        print(f"\n⚡ Testing energy compatibility:")
        for section_energy, move_energy, expected in test_energy_cases:
            result = optimizer._is_energy_compatible(section_energy, move_energy)
            print(f"   Section: {section_energy:.1f}, Move: {move_energy} → {result}")
            assert result == expected, f"Energy compatibility mismatch: {result} vs {expected}"
        
        print(f"✅ Musical structure alignment test passed")
        return True
        
    except Exception as e:
        print(f"❌ Musical structure alignment test failed: {e}")
        return False


def test_sequence_optimization_dynamic_programming():
    """Test sequence optimization using dynamic programming for smooth choreography flow."""
    print("\n🧮 Testing Sequence Optimization with Dynamic Programming")
    print("-" * 55)
    
    optimizer = ChoreographyOptimizer()
    engine = RecommendationEngine()
    music_analyzer = MusicAnalyzer()
    move_analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    fusion = FeatureFusion()
    
    try:
        # Get test files
        audio_files = list(Path("data/songs").glob("*.mp3"))
        if not audio_files:
            print("⚠️  No audio files found")
            return False
        
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:3]  # Use first 3 for speed
        
        if len(annotations) < 3:
            print("⚠️  Need at least 3 annotations for optimization testing")
            return False
        
        # Analyze music
        test_audio = str(audio_files[0])
        music_features = music_analyzer.analyze_audio(test_audio)
        music_embedding = fusion.create_multimodal_embedding(music_features, 
                                                           move_analyzer.analyze_move_clip(
                                                               os.path.join("data", annotations[0].video_path)
                                                           ))
        
        print(f"🎵 Test music: {Path(test_audio).name}")
        print(f"🎵 Duration: {music_features.duration:.1f}s")
        print(f"🎵 Sections: {len(music_features.sections)}")
        
        # Create move candidates and get recommendations
        candidates = []
        for i, annotation in enumerate(annotations):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            print(f"📹 Creating candidate {i+1}: {annotation.move_label}")
            
            # Analyze move
            move_result = move_analyzer.analyze_move_clip(video_path)
            
            # Create candidate
            candidate = engine.create_move_candidate(
                move_id=f"move_{i:03d}",
                video_path=video_path,
                move_label=annotation.move_label,
                analysis_result=move_result,
                music_features=music_features,
                energy_level=annotation.energy_level,
                difficulty="intermediate"
            )
            
            candidates.append(candidate)
        
        if len(candidates) < 3:
            print("⚠️  Could not create enough candidates")
            return False
        
        # Create recommendation request
        request = RecommendationRequest(
            music_features=music_features,
            music_embedding=music_embedding,
            target_difficulty="intermediate",
            target_energy="medium"
        )
        
        # Get recommendations
        recommendations = engine.recommend_moves(request, candidates, top_k=len(candidates))
        
        # Create optimization request
        target_duration = min(60.0, music_features.duration / 2)  # Use half the song duration or 60s max
        optimization_request = OptimizationRequest(
            music_features=music_features,
            candidate_moves=recommendations,
            target_duration=target_duration,
            diversity_weight=0.3,
            flow_weight=0.4,
            musical_alignment_weight=0.3,
            max_repetitions=2,
            min_diversity_threshold=0.5
        )
        
        print(f"\n🧮 Optimizing choreography:")
        print(f"   Target duration: {target_duration:.1f}s")
        print(f"   Candidate moves: {len(recommendations)}")
        
        # Optimize choreography
        optimized_sequence = optimizer.optimize_choreography(optimization_request)
        
        print(f"📊 Optimization Results:")
        print(f"   Selected moves: {len(optimized_sequence.moves)}")
        print(f"   Total duration: {optimized_sequence.total_duration:.1f}s")
        print(f"   Total score: {optimized_sequence.total_score:.3f}")
        print(f"   Diversity score: {optimized_sequence.diversity_score:.3f}")
        print(f"   Flow score: {optimized_sequence.flow_score:.3f}")
        print(f"   Musical alignment: {optimized_sequence.musical_alignment_score:.3f}")
        print(f"   Transitions: {len(optimized_sequence.transition_scores)}")
        print(f"   Optimization method: {optimized_sequence.optimization_method}")
        
        # Test optimization results
        assert isinstance(optimized_sequence, ChoreographySequence), "Should return ChoreographySequence"
        assert len(optimized_sequence.moves) > 0, "Should have selected moves"
        assert optimized_sequence.total_duration > 0, "Should have positive duration"
        assert 0.0 <= optimized_sequence.total_score <= 1.0, f"Invalid total score: {optimized_sequence.total_score}"
        assert 0.0 <= optimized_sequence.diversity_score <= 1.0, f"Invalid diversity score: {optimized_sequence.diversity_score}"
        assert 0.0 <= optimized_sequence.flow_score <= 1.0, f"Invalid flow score: {optimized_sequence.flow_score}"
        assert 0.0 <= optimized_sequence.musical_alignment_score <= 1.0, f"Invalid musical alignment: {optimized_sequence.musical_alignment_score}"
        
        # Test that duration is reasonable (not too far from target)
        duration_ratio = optimized_sequence.total_duration / target_duration
        assert 0.5 <= duration_ratio <= 1.5, f"Duration too far from target: {duration_ratio:.2f}"
        
        # Test transition scores
        expected_transitions = len(optimized_sequence.moves) - 1
        assert len(optimized_sequence.transition_scores) == expected_transitions, \
            f"Wrong number of transitions: {len(optimized_sequence.transition_scores)} vs {expected_transitions}"
        
        # Print move sequence
        print(f"\n📝 Move Sequence:")
        for i, move in enumerate(optimized_sequence.moves):
            print(f"   {i+1}. {move.move_label} ({move.analysis_result.duration:.1f}s)")
        
        print(f"✅ Sequence optimization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Sequence optimization test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Diversity Selection and Choreography Flow Optimization Tests (Task 5.3)")
    print("=" * 90)
    
    tests = [
        ("Choreography Optimizer Initialization", test_choreography_optimizer_initialization),
        ("Transition Compatibility Calculation", test_transition_compatibility_calculation),
        ("Transition Matrix Building", test_transition_matrix_building),
        ("Diversity Selection Algorithm", test_diversity_selection_algorithm),
        ("Musical Structure Alignment", test_musical_structure_alignment),
        ("Sequence Optimization with Dynamic Programming", test_sequence_optimization_dynamic_programming)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*90}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*90}")
        
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append(False)
    
    # Summary
    print(f"\n{'='*90}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*90}")
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"✅ Task 5.3 - Diversity Selection and Choreography Flow Optimization implementation is complete")
        print(f"✅ Diversity selection algorithm to avoid repetitive move sequences")
        print(f"✅ Transition compatibility matrix between all move pairs based on pose analysis")
        print(f"✅ Sequence optimization using dynamic programming for smooth choreography flow")
        print(f"✅ Musical structure awareness to align move complexity with song sections")
        return True
    else:
        print(f"❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)