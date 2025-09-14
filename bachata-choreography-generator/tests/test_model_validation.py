#!/usr/bin/env python3
"""
Test script for the model validation and performance testing framework (Task 5.4).
Tests cross-validation system, A/B testing framework, evaluation metrics,
and performance benchmarking.
"""

import sys
import os
import numpy as np
from pathlib import Path
import tempfile
import json

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.model_validation import (
    ModelValidationFramework, ValidationResult, CrossValidationResult, 
    ABTestResult, PerformanceBenchmark
)
from app.services.recommendation_engine import RecommendationEngine, MoveCandidate
from app.services.music_analyzer import MusicAnalyzer
from app.services.move_analyzer import MoveAnalyzer
from app.services.annotation_interface import AnnotationInterface


def test_model_validation_framework_initialization():
    """Test model validation framework initialization."""
    print("🚀 Testing Model Validation Framework Initialization")
    print("-" * 60)
    
    framework = ModelValidationFramework()
    
    # Test components initialization
    assert framework.recommendation_engine is not None, "RecommendationEngine not initialized"
    assert framework.choreography_optimizer is not None, "ChoreographyOptimizer not initialized"
    assert framework.feature_fusion is not None, "FeatureFusion not initialized"
    assert framework.music_analyzer is not None, "MusicAnalyzer not initialized"
    assert framework.move_analyzer is not None, "MoveAnalyzer not initialized"
    
    # Test quality metrics
    expected_metrics = [
        'flow_score',
        'musicality_score', 
        'difficulty_progression_score',
        'diversity_score',
        'transition_smoothness'
    ]
    
    print(f"📊 Quality metrics: {framework.quality_metrics}")
    assert framework.quality_metrics == expected_metrics, f"Unexpected quality metrics: {framework.quality_metrics}"
    
    print(f"✅ Model validation framework initialization test passed")
    return True


def test_choreography_quality_evaluation():
    """Test choreography quality evaluation metrics."""
    print("\n📊 Testing Choreography Quality Evaluation")
    print("-" * 60)
    
    framework = ModelValidationFramework()
    music_analyzer = MusicAnalyzer()
    
    try:
        # Get test audio file
        audio_files = list(Path("data/songs").glob("*.mp3"))
        if not audio_files:
            print("⚠️  No audio files found")
            return False
        
        test_audio = str(audio_files[0])
        print(f"🎵 Test audio: {Path(test_audio).name}")
        
        # Analyze music
        music_features = music_analyzer.analyze_audio(test_audio)
        
        # Create mock choreography sequence
        from app.services.choreography_optimizer import ChoreographySequence, TransitionScore
        
        # Create mock moves
        mock_moves = []
        for i in range(3):
            mock_move = type('MockMove', (), {
                'move_label': f'basic_step_{i}',
                'difficulty': 'intermediate',
                'analysis_result': type('MockResult', (), {
                    'duration': 15.0
                })()
            })()
            mock_moves.append(mock_move)
        
        # Create mock transition scores
        mock_transitions = [
            TransitionScore(
                from_move="basic_step_0",
                to_move="basic_step_1", 
                compatibility_score=0.8,
                pose_similarity=0.8,
                movement_flow=0.8,
                energy_continuity=0.8,
                difficulty_progression=0.8,
                is_smooth=True,
                requires_pause=False,
                transition_time=0.5
            ),
            TransitionScore(
                from_move="basic_step_1",
                to_move="basic_step_2",
                compatibility_score=0.9,
                pose_similarity=0.9,
                movement_flow=0.9,
                energy_continuity=0.9,
                difficulty_progression=0.9,
                is_smooth=True,
                requires_pause=False,
                transition_time=0.5
            )
        ]
        
        # Create mock choreography
        choreography = ChoreographySequence(
            moves=mock_moves,
            total_duration=45.0,
            total_score=0.8,
            diversity_score=0.7,
            flow_score=0.85,
            musical_alignment_score=0.8,
            transition_scores=mock_transitions,
            section_alignment=[(section, mock_moves) for section in music_features.sections[:1]],
            optimization_method="test",
            iterations=1
        )
        
        print(f"🎭 Mock choreography: {len(choreography.moves)} moves, {choreography.total_duration:.1f}s")
        
        # Evaluate quality
        quality_scores = framework.evaluate_choreography_quality(choreography, music_features)
        
        print(f"📊 Quality Evaluation Results:")
        for metric, score in quality_scores.items():
            print(f"   {metric}: {score:.3f}")
        
        # Test quality score properties
        for metric, score in quality_scores.items():
            assert 0.0 <= score <= 1.0, f"Quality score out of range: {metric}={score}"
        
        # Test that all expected metrics are present
        expected_metrics = [
            'flow_score', 'musicality_score', 'difficulty_progression_score',
            'diversity_score', 'transition_smoothness', 'overall_quality'
        ]
        
        for metric in expected_metrics:
            assert metric in quality_scores, f"Missing quality metric: {metric}"
        
        print(f"✅ Choreography quality evaluation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Choreography quality evaluation test failed: {e}")
        return False


def test_cross_validation_system():
    """Test cross-validation system using held-out test songs."""
    print("\n🔄 Testing Cross-Validation System")
    print("-" * 60)
    
    framework = ModelValidationFramework()
    engine = RecommendationEngine()
    annotation_interface = AnnotationInterface()
    
    try:
        # Get test songs
        audio_files = list(Path("data/songs").glob("*.mp3"))[:3]  # Use 3 for speed
        if len(audio_files) < 3:
            print("⚠️  Need at least 3 audio files for cross-validation testing")
            return False
        
        test_songs = [str(f) for f in audio_files]
        print(f"🎵 Test songs: {len(test_songs)}")
        
        # Get move candidates
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:3]  # Use 3 for speed
        
        if len(annotations) < 3:
            print("⚠️  Need at least 3 annotations for testing")
            return False
        
        # Create move candidates
        move_candidates = []
        for i, annotation in enumerate(annotations):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            # Create mock candidate (simplified for testing)
            candidate = type('MockCandidate', (), {
                'move_id': f'move_{i:03d}',
                'video_path': video_path,
                'move_label': annotation.move_label,
                'energy_level': annotation.energy_level,
                'difficulty': 'intermediate'
            })()
            
            move_candidates.append(candidate)
        
        if len(move_candidates) < 3:
            print("⚠️  Could not create enough move candidates")
            return False
        
        print(f"🕺 Move candidates: {len(move_candidates)}")
        
        # Create mock expert ratings
        expert_ratings = {
            test_songs[0]: 0.8,
            test_songs[1]: 0.7,
            test_songs[2]: 0.9
        }
        
        print(f"👨‍🏫 Expert ratings: {len(expert_ratings)} songs")
        
        # Run cross-validation
        k_folds = 3
        cv_result = framework.run_cross_validation(
            test_songs, move_candidates, k_folds, expert_ratings
        )
        
        print(f"📊 Cross-Validation Results:")
        print(f"   Mean score: {cv_result.mean_score:.3f}")
        print(f"   Std score: {cv_result.std_score:.3f}")
        print(f"   Fold scores: {[f'{s:.3f}' for s in cv_result.fold_scores]}")
        print(f"   Best fold: {cv_result.best_fold}")
        print(f"   Worst fold: {cv_result.worst_fold}")
        
        # Test cross-validation result properties
        assert isinstance(cv_result, CrossValidationResult), "Should return CrossValidationResult"
        assert len(cv_result.fold_scores) == k_folds, f"Wrong number of fold scores: {len(cv_result.fold_scores)}"
        assert 0.0 <= cv_result.mean_score <= 1.0, f"Mean score out of range: {cv_result.mean_score}"
        assert cv_result.std_score >= 0.0, f"Std score should be non-negative: {cv_result.std_score}"
        assert 0 <= cv_result.best_fold < k_folds, f"Invalid best fold: {cv_result.best_fold}"
        assert 0 <= cv_result.worst_fold < k_folds, f"Invalid worst fold: {cv_result.worst_fold}"
        
        # Test validation details
        assert len(cv_result.validation_details) == k_folds, "Should have details for each fold"
        
        print(f"✅ Cross-validation system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Cross-validation system test failed: {e}")
        return False


def test_ab_testing_framework():
    """Test A/B testing framework to compare different scoring weight combinations."""
    print("\n🧪 Testing A/B Testing Framework")
    print("-" * 60)
    
    framework = ModelValidationFramework()
    annotation_interface = AnnotationInterface()
    
    try:
        # Get test songs
        audio_files = list(Path("data/songs").glob("*.mp3"))[:2]  # Use 2 for speed
        if len(audio_files) < 2:
            print("⚠️  Need at least 2 audio files for A/B testing")
            return False
        
        test_songs = [str(f) for f in audio_files]
        print(f"🎵 Test songs: {len(test_songs)}")
        
        # Get move candidates
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:3]
        
        move_candidates = []
        for i, annotation in enumerate(annotations):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            candidate = type('MockCandidate', (), {
                'move_id': f'move_{i:03d}',
                'video_path': video_path,
                'move_label': annotation.move_label,
                'energy_level': annotation.energy_level,
                'difficulty': 'intermediate'
            })()
            
            move_candidates.append(candidate)
        
        if len(move_candidates) < 2:
            print("⚠️  Could not create enough move candidates")
            return False
        
        print(f"🕺 Move candidates: {len(move_candidates)}")
        
        # Define variant weights
        variant_a_weights = {
            'audio_similarity': 0.40,
            'tempo_matching': 0.30,
            'energy_alignment': 0.20,
            'difficulty_compatibility': 0.10
        }
        
        variant_b_weights = {
            'audio_similarity': 0.30,
            'tempo_matching': 0.40,
            'energy_alignment': 0.20,
            'difficulty_compatibility': 0.10
        }
        
        print(f"🅰️  Variant A: audio=40%, tempo=30%, energy=20%, difficulty=10%")
        print(f"🅱️  Variant B: audio=30%, tempo=40%, energy=20%, difficulty=10%")
        
        # Run A/B test
        ab_result = framework.run_ab_test(
            test_songs, move_candidates, variant_a_weights, variant_b_weights
        )
        
        print(f"📊 A/B Test Results:")
        print(f"   Variant A score: {ab_result.variant_a_score:.3f}")
        print(f"   Variant B score: {ab_result.variant_b_score:.3f}")
        print(f"   Improvement: {ab_result.improvement:.1f}%")
        print(f"   Statistical significance: {ab_result.statistical_significance:.3f}")
        print(f"   Sample size: {ab_result.sample_size}")
        
        # Test A/B result properties
        assert isinstance(ab_result, ABTestResult), "Should return ABTestResult"
        assert 0.0 <= ab_result.variant_a_score <= 1.0, f"Variant A score out of range: {ab_result.variant_a_score}"
        assert 0.0 <= ab_result.variant_b_score <= 1.0, f"Variant B score out of range: {ab_result.variant_b_score}"
        assert -100.0 <= ab_result.improvement <= 100.0, f"Improvement out of reasonable range: {ab_result.improvement}"
        assert 0.0 <= ab_result.statistical_significance <= 1.0, f"Statistical significance out of range: {ab_result.statistical_significance}"
        assert ab_result.sample_size > 0, f"Sample size should be positive: {ab_result.sample_size}"
        
        # Test test details
        assert 'variant_a_scores' in ab_result.test_details, "Missing variant A scores"
        assert 'variant_b_scores' in ab_result.test_details, "Missing variant B scores"
        assert len(ab_result.test_details['variant_a_scores']) == ab_result.sample_size, "Variant A scores count mismatch"
        assert len(ab_result.test_details['variant_b_scores']) == ab_result.sample_size, "Variant B scores count mismatch"
        
        print(f"✅ A/B testing framework test passed")
        return True
        
    except Exception as e:
        print(f"❌ A/B testing framework test failed: {e}")
        return False


def test_performance_benchmarking():
    """Test performance benchmarking for recommendation speed and accuracy."""
    print("\n⚡ Testing Performance Benchmarking")
    print("-" * 60)
    
    framework = ModelValidationFramework()
    annotation_interface = AnnotationInterface()
    
    try:
        # Get test songs (use fewer for benchmarking)
        audio_files = list(Path("data/songs").glob("*.mp3"))[:2]
        if len(audio_files) < 1:
            print("⚠️  Need at least 1 audio file for benchmarking")
            return False
        
        test_songs = [str(f) for f in audio_files]
        print(f"🎵 Test songs: {len(test_songs)}")
        
        # Get move candidates
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:2]
        
        move_candidates = []
        for i, annotation in enumerate(annotations):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            candidate = type('MockCandidate', (), {
                'move_id': f'move_{i:03d}',
                'video_path': video_path,
                'move_label': annotation.move_label,
                'energy_level': annotation.energy_level,
                'difficulty': 'intermediate'
            })()
            
            move_candidates.append(candidate)
        
        if len(move_candidates) < 1:
            print("⚠️  Could not create move candidates")
            return False
        
        print(f"🕺 Move candidates: {len(move_candidates)}")
        
        # Define operations to benchmark
        operations = ['music_analysis', 'recommendation_generation']  # Subset for speed
        
        print(f"⚡ Benchmarking operations: {operations}")
        
        # Run benchmarks
        benchmarks = framework.benchmark_performance(test_songs, move_candidates, operations)
        
        print(f"📊 Performance Benchmark Results:")
        for operation, benchmark in benchmarks.items():
            print(f"   {operation}:")
            print(f"      Mean time: {benchmark.mean_time:.3f}s")
            print(f"      Std time: {benchmark.std_time:.3f}s")
            print(f"      Min time: {benchmark.min_time:.3f}s")
            print(f"      Max time: {benchmark.max_time:.3f}s")
            print(f"      Throughput: {benchmark.throughput:.2f} ops/sec")
        
        # Test benchmark properties
        for operation, benchmark in benchmarks.items():
            assert isinstance(benchmark, PerformanceBenchmark), f"Should return PerformanceBenchmark for {operation}"
            assert benchmark.mean_time > 0, f"Mean time should be positive for {operation}: {benchmark.mean_time}"
            assert benchmark.std_time >= 0, f"Std time should be non-negative for {operation}: {benchmark.std_time}"
            assert benchmark.min_time > 0, f"Min time should be positive for {operation}: {benchmark.min_time}"
            assert benchmark.max_time >= benchmark.min_time, f"Max time should be >= min time for {operation}"
            assert benchmark.throughput >= 0, f"Throughput should be non-negative for {operation}: {benchmark.throughput}"
        
        print(f"✅ Performance benchmarking test passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmarking test failed: {e}")
        return False


def test_model_accuracy_validation():
    """Test model accuracy validation against ground truth."""
    print("\n🎯 Testing Model Accuracy Validation")
    print("-" * 60)
    
    framework = ModelValidationFramework()
    annotation_interface = AnnotationInterface()
    
    try:
        # Get test songs
        audio_files = list(Path("data/songs").glob("*.mp3"))[:1]  # Use 1 for speed
        if not audio_files:
            print("⚠️  No audio files found")
            return False
        
        test_songs = [str(f) for f in audio_files]
        print(f"🎵 Test songs: {len(test_songs)}")
        
        # Get move candidates
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:3]
        
        move_candidates = []
        for i, annotation in enumerate(annotations):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            candidate = type('MockCandidate', (), {
                'move_id': f'move_{i:03d}',
                'video_path': video_path,
                'move_label': annotation.move_label,
                'energy_level': annotation.energy_level,
                'difficulty': 'intermediate'
            })()
            
            move_candidates.append(candidate)
        
        if not move_candidates:
            print("⚠️  Could not create move candidates")
            return False
        
        print(f"🕺 Move candidates: {len(move_candidates)}")
        
        # Create mock ground truth
        ground_truth = {
            test_songs[0]: {
                'expected_quality': 0.8,
                'expected_moves': ['basic_step', 'cross_body_lead']
            }
        }
        
        print(f"📋 Ground truth: {len(ground_truth)} entries")
        
        # Run accuracy validation
        validation_result = framework.validate_model_accuracy(
            test_songs, move_candidates, ground_truth
        )
        
        print(f"📊 Model Accuracy Validation Results:")
        print(f"   Test name: {validation_result.test_name}")
        print(f"   Score: {validation_result.score:.3f}")
        print(f"   Execution time: {validation_result.execution_time:.2f}s")
        print(f"   Success: {validation_result.success}")
        if validation_result.error_message:
            print(f"   Error: {validation_result.error_message}")
        
        # Test validation result properties
        assert isinstance(validation_result, ValidationResult), "Should return ValidationResult"
        assert validation_result.test_name == "model_accuracy", f"Wrong test name: {validation_result.test_name}"
        assert validation_result.execution_time > 0, f"Execution time should be positive: {validation_result.execution_time}"
        
        if validation_result.success:
            assert 0.0 <= validation_result.score <= 1.0, f"Score out of range: {validation_result.score}"
            assert isinstance(validation_result.details, dict), "Details should be a dictionary"
        
        print(f"✅ Model accuracy validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model accuracy validation test failed: {e}")
        return False


def test_validation_results_persistence():
    """Test saving and loading validation results."""
    print("\n💾 Testing Validation Results Persistence")
    print("-" * 60)
    
    framework = ModelValidationFramework()
    
    try:
        # Create mock validation results
        test_results = {
            'cross_validation': {
                'mean_score': 0.75,
                'std_score': 0.1,
                'fold_scores': [0.7, 0.8, 0.75]
            },
            'ab_test': {
                'variant_a_score': 0.7,
                'variant_b_score': 0.8,
                'improvement': 14.3
            },
            'performance_benchmark': {
                'music_analysis': {
                    'mean_time': 2.5,
                    'throughput': 0.4
                }
            }
        }
        
        print(f"📊 Mock results: {len(test_results)} categories")
        
        # Test saving results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        framework.save_validation_results(test_results, temp_path)
        
        # Verify file was created
        assert os.path.exists(temp_path), "Results file should be created"
        
        # Test loading results
        loaded_results = framework.load_validation_results(temp_path)
        
        print(f"📥 Loaded results: {len(loaded_results)} categories")
        
        # Verify loaded results match original
        assert loaded_results == test_results, "Loaded results should match original"
        
        # Test specific values
        assert loaded_results['cross_validation']['mean_score'] == 0.75, "Mean score mismatch"
        assert loaded_results['ab_test']['improvement'] == 14.3, "Improvement mismatch"
        
        # Clean up
        os.unlink(temp_path)
        
        print(f"✅ Validation results persistence test passed")
        return True
        
    except Exception as e:
        print(f"❌ Validation results persistence test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Model Validation and Performance Testing Framework Tests (Task 5.4)")
    print("=" * 95)
    
    tests = [
        ("Model Validation Framework Initialization", test_model_validation_framework_initialization),
        ("Choreography Quality Evaluation", test_choreography_quality_evaluation),
        ("Cross-Validation System", test_cross_validation_system),
        ("A/B Testing Framework", test_ab_testing_framework),
        ("Performance Benchmarking", test_performance_benchmarking),
        ("Model Accuracy Validation", test_model_accuracy_validation),
        ("Validation Results Persistence", test_validation_results_persistence)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*95}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*95}")
        
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
    print(f"\n{'='*95}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*95}")
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"✅ Task 5.4 - Model Validation and Performance Testing Framework implementation is complete")
        print(f"✅ Cross-validation system using held-out test songs and expert choreographer ratings")
        print(f"✅ A/B testing framework to compare different scoring weight combinations")
        print(f"✅ Evaluation metrics for choreography quality (flow, musicality, difficulty progression)")
        print(f"✅ Performance benchmarking for recommendation speed and accuracy")
        return True
    else:
        print(f"❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)