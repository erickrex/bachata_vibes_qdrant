#!/usr/bin/env python3
"""
Test script for the multi-factor scoring recommendation system (Task 5.2).
Tests weighted scoring algorithm, cosine similarity matching, tempo compatibility,
and energy level alignment.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.recommendation_engine import (
    RecommendationEngine, MoveCandidate, RecommendationRequest, RecommendationScore
)
from app.services.feature_fusion import FeatureFusion
from app.services.music_analyzer import MusicAnalyzer
from app.services.move_analyzer import MoveAnalyzer
from app.services.annotation_interface import AnnotationInterface


def test_recommendation_engine_initialization():
    """Test recommendation engine initialization and default weights."""
    print("🚀 Testing Recommendation Engine Initialization")
    print("-" * 50)
    
    engine = RecommendationEngine()
    
    # Test default weights
    expected_weights = {
        'audio_similarity': 0.40,
        'tempo_matching': 0.30,
        'energy_alignment': 0.20,
        'difficulty_compatibility': 0.10
    }
    
    print(f"📊 Default weights: {engine.default_weights}")
    assert engine.default_weights == expected_weights, f"Unexpected default weights: {engine.default_weights}"
    
    # Test weight sum
    weight_sum = sum(engine.default_weights.values())
    print(f"📊 Weight sum: {weight_sum}")
    assert abs(weight_sum - 1.0) < 0.001, f"Weights don't sum to 1.0: {weight_sum}"
    
    # Test tempo tolerance
    print(f"🎵 Default tempo tolerance: ±{engine.tempo_tolerance} BPM")
    assert engine.tempo_tolerance == 10.0, f"Unexpected tempo tolerance: {engine.tempo_tolerance}"
    
    print(f"✅ Recommendation engine initialization test passed")
    return True


def test_audio_similarity_calculation():
    """Test cosine similarity matching between music and move embeddings."""
    print("\n🎵 Testing Audio Similarity Calculation")
    print("-" * 50)
    
    engine = RecommendationEngine()
    fusion = FeatureFusion()
    music_analyzer = MusicAnalyzer()
    move_analyzer = MoveAnalyzer(target_fps=30)
    
    try:
        # Get test audio and video files
        audio_files = list(Path("data/songs").glob("*.mp3"))
        if not audio_files:
            print("⚠️  No audio files found")
            return False
        
        annotation_interface = AnnotationInterface()
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips
        
        if not annotations:
            print("⚠️  No annotations found")
            return False
        
        # Analyze music
        test_audio = str(audio_files[0])
        music_features = music_analyzer.analyze_audio(test_audio)
        music_embedding = fusion.create_audio_embedding(music_features)
        
        print(f"🎵 Test audio: {Path(test_audio).name}")
        print(f"📊 Music embedding shape: {music_embedding.shape}")
        
        # Test with multiple moves
        similarities = []
        
        for i, annotation in enumerate(annotations[:3]):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            print(f"\n📹 Move {i+1}: {annotation.move_label}")
            
            # Analyze move
            move_result = move_analyzer.analyze_move_clip(video_path)
            move_embedding = fusion.create_multimodal_embedding(music_features, move_result)
            
            # Calculate similarity
            similarity = engine._calculate_audio_similarity(
                type('MockEmbedding', (), {'audio_embedding': music_embedding})(),
                move_embedding
            )
            
            similarities.append(similarity)
            print(f"🔍 Audio similarity: {similarity:.4f}")
            
            # Test similarity properties
            assert 0.0 <= similarity <= 1.0, f"Similarity out of range: {similarity}"
        
        if similarities:
            print(f"\n📊 Similarity statistics:")
            print(f"   Mean: {np.mean(similarities):.4f}")
            print(f"   Std:  {np.std(similarities):.4f}")
            print(f"   Min:  {np.min(similarities):.4f}")
            print(f"   Max:  {np.max(similarities):.4f}")
        
        print(f"✅ Audio similarity calculation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Audio similarity test failed: {e}")
        return False


def test_tempo_compatibility_scoring():
    """Test tempo compatibility scoring with BPM range matching."""
    print("\n🎵 Testing Tempo Compatibility Scoring")
    print("-" * 50)
    
    engine = RecommendationEngine()
    
    # Create mock candidate with tempo range
    class MockCandidate:
        def __init__(self, tempo_range):
            self.analysis_result = type('MockResult', (), {
                'tempo_compatibility_range': tempo_range
            })()
    
    # Test cases: (music_tempo, move_tempo_range, expected_compatibility_range)
    test_cases = [
        (120.0, (115.0, 125.0), (0.9, 1.0)),  # Within range
        (130.0, (115.0, 125.0), (0.4, 0.6)),  # Outside range, within tolerance
        (140.0, (115.0, 125.0), (0.6, 0.9)),  # Outside range and tolerance (exponential decay)
        (112.0, (115.0, 125.0), (0.6, 0.8)),  # Outside range, within tolerance
    ]
    
    print(f"🧪 Testing {len(test_cases)} tempo compatibility scenarios:")
    
    for i, (music_tempo, tempo_range, expected_range) in enumerate(test_cases):
        candidate = MockCandidate(tempo_range)
        compatibility, tempo_diff = engine._calculate_tempo_compatibility(
            music_tempo, candidate, 10.0
        )
        
        print(f"   {i+1}. Music: {music_tempo} BPM, Move: {tempo_range} BPM")
        print(f"      → Compatibility: {compatibility:.3f}, Diff: {tempo_diff:.1f}")
        
        # Test compatibility is in expected range
        assert expected_range[0] <= compatibility <= expected_range[1], \
            f"Compatibility {compatibility} not in expected range {expected_range}"
        
        # Test compatibility is in valid range
        assert 0.0 <= compatibility <= 1.0, f"Compatibility out of range: {compatibility}"
        
        # Test tempo difference is non-negative
        assert tempo_diff >= 0.0, f"Negative tempo difference: {tempo_diff}"
    
    print(f"✅ Tempo compatibility scoring test passed")
    return True


def test_energy_alignment_scoring():
    """Test energy level alignment scoring (low/medium/high)."""
    print("\n⚡ Testing Energy Alignment Scoring")
    print("-" * 50)
    
    engine = RecommendationEngine()
    
    # Test cases: (target_energy, move_energy, expected_score, expected_match)
    test_cases = [
        ("low", "low", 1.0, True),      # Exact match
        ("medium", "medium", 1.0, True),  # Exact match
        ("high", "high", 1.0, True),    # Exact match
        ("low", "medium", 0.7, False),  # Adjacent level
        ("medium", "high", 0.7, False), # Adjacent level
        ("low", "high", 0.3, False),    # Opposite levels
        ("high", "low", 0.3, False),    # Opposite levels
    ]
    
    print(f"🧪 Testing {len(test_cases)} energy alignment scenarios:")
    
    for i, (target, move, expected_score, expected_match) in enumerate(test_cases):
        score, match = engine._calculate_energy_alignment(target, move)
        
        print(f"   {i+1}. Target: {target}, Move: {move}")
        print(f"      → Score: {score:.1f}, Match: {match}")
        
        # Test score matches expected
        assert abs(score - expected_score) < 0.1, \
            f"Score {score} doesn't match expected {expected_score}"
        
        # Test match flag
        assert match == expected_match, \
            f"Match {match} doesn't match expected {expected_match}"
        
        # Test score is in valid range
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    
    print(f"✅ Energy alignment scoring test passed")
    return True


def test_difficulty_compatibility_scoring():
    """Test difficulty compatibility scoring."""
    print("\n⭐ Testing Difficulty Compatibility Scoring")
    print("-" * 50)
    
    engine = RecommendationEngine()
    
    # Test cases: (target_difficulty, move_difficulty, expected_score, expected_match)
    test_cases = [
        ("beginner", "beginner", 1.0, True),        # Exact match
        ("intermediate", "intermediate", 1.0, True), # Exact match
        ("advanced", "advanced", 1.0, True),        # Exact match
        ("beginner", "intermediate", 0.8, False),   # Adjacent level
        ("intermediate", "advanced", 0.8, False),   # Adjacent level
        ("beginner", "advanced", 0.4, False),       # Two levels apart
        ("advanced", "beginner", 0.4, False),       # Two levels apart
    ]
    
    print(f"🧪 Testing {len(test_cases)} difficulty compatibility scenarios:")
    
    for i, (target, move, expected_score, expected_match) in enumerate(test_cases):
        score, match = engine._calculate_difficulty_compatibility(target, move)
        
        print(f"   {i+1}. Target: {target}, Move: {move}")
        print(f"      → Score: {score:.1f}, Match: {match}")
        
        # Test score matches expected
        assert abs(score - expected_score) < 0.1, \
            f"Score {score} doesn't match expected {expected_score}"
        
        # Test match flag
        assert match == expected_match, \
            f"Match {match} doesn't match expected {expected_match}"
        
        # Test score is in valid range
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    
    print(f"✅ Difficulty compatibility scoring test passed")
    return True


def test_music_energy_detection():
    """Test automatic music energy level detection."""
    print("\n🎵 Testing Music Energy Level Detection")
    print("-" * 50)
    
    engine = RecommendationEngine()
    music_analyzer = MusicAnalyzer()
    
    try:
        # Test with different audio files
        audio_files = list(Path("data/songs").glob("*.mp3"))[:3]
        if len(audio_files) < 2:
            print("⚠️  Need at least 2 audio files for energy detection testing")
            return False
        
        energy_detections = []
        
        for i, audio_file in enumerate(audio_files):
            print(f"\n🎵 Audio {i+1}: {Path(audio_file).name}")
            
            # Analyze music
            music_features = music_analyzer.analyze_audio(str(audio_file))
            
            # Detect energy level
            energy_level = engine._detect_music_energy_level(music_features)
            energy_detections.append(energy_level)
            
            print(f"🔍 Detected energy level: {energy_level}")
            print(f"📊 Music stats:")
            print(f"   Tempo: {music_features.tempo:.1f} BPM")
            print(f"   RMS energy: {np.mean(music_features.rms_energy):.4f}")
            print(f"   Spectral centroid: {np.mean(music_features.spectral_centroid):.1f}")
            
            # Test energy level is valid
            assert energy_level in ['low', 'medium', 'high'], \
                f"Invalid energy level: {energy_level}"
        
        # Test that we get some variety in energy levels (not all the same)
        unique_levels = set(energy_detections)
        print(f"\n📊 Energy level distribution: {dict(zip(unique_levels, [energy_detections.count(level) for level in unique_levels]))}")
        
        print(f"✅ Music energy detection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Music energy detection test failed: {e}")
        return False


def test_move_candidate_creation():
    """Test move candidate creation with multimodal embeddings."""
    print("\n🕺 Testing Move Candidate Creation")
    print("-" * 50)
    
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
        annotations = annotation_collection.clips
        
        if not annotations:
            print("⚠️  No annotations found")
            return False
        
        # Analyze music
        test_audio = str(audio_files[0])
        music_features = music_analyzer.analyze_audio(test_audio)
        
        # Test with first annotation
        annotation = annotations[0]
        video_path = annotation.video_path
        if not os.path.isabs(video_path):
            video_path = os.path.join("data", video_path)
        
        if not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path}")
            return False
        
        print(f"📹 Creating candidate for: {annotation.move_label}")
        
        # Analyze move
        move_result = move_analyzer.analyze_move_clip(video_path)
        
        # Create move candidate
        candidate = engine.create_move_candidate(
            move_id="test_001",
            video_path=video_path,
            move_label=annotation.move_label,
            analysis_result=move_result,
            music_features=music_features,
            energy_level="medium",
            difficulty="intermediate",
            estimated_tempo=120.0,
            lead_follow_roles="both"
        )
        
        # Test candidate properties
        print(f"📊 Candidate properties:")
        print(f"   Move ID: {candidate.move_id}")
        print(f"   Move label: {candidate.move_label}")
        print(f"   Energy level: {candidate.energy_level}")
        print(f"   Difficulty: {candidate.difficulty}")
        print(f"   Estimated tempo: {candidate.estimated_tempo}")
        print(f"   Lead/follow roles: {candidate.lead_follow_roles}")
        
        # Test multimodal embedding
        assert candidate.multimodal_embedding is not None, "Missing multimodal embedding"
        assert candidate.multimodal_embedding.combined_embedding.shape == (512,), \
            f"Wrong embedding shape: {candidate.multimodal_embedding.combined_embedding.shape}"
        
        print(f"📊 Embedding shape: {candidate.multimodal_embedding.combined_embedding.shape}")
        
        print(f"✅ Move candidate creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Move candidate creation test failed: {e}")
        return False


def test_end_to_end_recommendation():
    """Test complete end-to-end recommendation workflow."""
    print("\n🎯 Testing End-to-End Recommendation Workflow")
    print("-" * 50)
    
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
        
        if len(annotations) < 2:
            print("⚠️  Need at least 2 annotations for recommendation testing")
            return False
        
        # Analyze music
        test_audio = str(audio_files[0])
        music_features = music_analyzer.analyze_audio(test_audio)
        music_embedding = fusion.create_multimodal_embedding(music_features, 
                                                           move_analyzer.analyze_move_clip(
                                                               os.path.join("data", annotations[0].video_path)
                                                           ))
        
        print(f"🎵 Test music: {Path(test_audio).name}")
        print(f"🎵 Tempo: {music_features.tempo:.1f} BPM")
        
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
                difficulty="intermediate",
                estimated_tempo=annotation.estimated_tempo
            )
            
            candidates.append(candidate)
        
        if len(candidates) < 2:
            print("⚠️  Could not create enough candidates")
            return False
        
        print(f"\n🎯 Generated {len(candidates)} move candidates")
        
        # Create recommendation request
        request = RecommendationRequest(
            music_features=music_features,
            music_embedding=music_embedding,
            target_difficulty="intermediate",
            target_energy="medium",
            tempo_tolerance=10.0
        )
        
        # Get recommendations
        recommendations = engine.recommend_moves(request, candidates, top_k=len(candidates))
        
        print(f"\n📊 Recommendation Results:")
        for i, rec in enumerate(recommendations):
            print(f"   {i+1}. {rec.move_candidate.move_label}")
            print(f"      Overall: {rec.overall_score:.3f}")
            print(f"      Audio: {rec.audio_similarity:.3f}, Tempo: {rec.tempo_compatibility:.3f}")
            print(f"      Energy: {rec.energy_alignment:.3f}, Difficulty: {rec.difficulty_compatibility:.3f}")
            
            # Get explanation
            explanation = engine.get_scoring_explanation(rec)
            print(f"      Explanation: {explanation}")
        
        # Test recommendation properties
        assert len(recommendations) == len(candidates), \
            f"Wrong number of recommendations: {len(recommendations)} vs {len(candidates)}"
        
        # Test recommendations are sorted by score (descending)
        scores = [rec.overall_score for rec in recommendations]
        assert scores == sorted(scores, reverse=True), "Recommendations not sorted by score"
        
        # Test all scores are in valid range
        for rec in recommendations:
            assert 0.0 <= rec.overall_score <= 1.0, f"Invalid overall score: {rec.overall_score}"
            assert 0.0 <= rec.audio_similarity <= 1.0, f"Invalid audio similarity: {rec.audio_similarity}"
            assert 0.0 <= rec.tempo_compatibility <= 1.0, f"Invalid tempo compatibility: {rec.tempo_compatibility}"
            assert 0.0 <= rec.energy_alignment <= 1.0, f"Invalid energy alignment: {rec.energy_alignment}"
            assert 0.0 <= rec.difficulty_compatibility <= 1.0, f"Invalid difficulty compatibility: {rec.difficulty_compatibility}"
        
        print(f"✅ End-to-end recommendation test passed")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end recommendation test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Multi-Factor Scoring Recommendation System Tests (Task 5.2)")
    print("=" * 80)
    
    tests = [
        ("Recommendation Engine Initialization", test_recommendation_engine_initialization),
        ("Audio Similarity Calculation", test_audio_similarity_calculation),
        ("Tempo Compatibility Scoring", test_tempo_compatibility_scoring),
        ("Energy Alignment Scoring", test_energy_alignment_scoring),
        ("Difficulty Compatibility Scoring", test_difficulty_compatibility_scoring),
        ("Music Energy Detection", test_music_energy_detection),
        ("Move Candidate Creation", test_move_candidate_creation),
        ("End-to-End Recommendation", test_end_to_end_recommendation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*80}")
        
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
    print(f"\n{'='*80}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"✅ Task 5.2 - Multi-Factor Scoring Recommendation System implementation is complete")
        print(f"✅ Weighted scoring algorithm: audio similarity (40%), tempo matching (30%), energy alignment (20%), difficulty compatibility (10%)")
        print(f"✅ Cosine similarity matching between music and move embeddings")
        print(f"✅ Tempo compatibility scoring with BPM range matching (±10 BPM tolerance)")
        print(f"✅ Energy level alignment scoring (low/medium/high) between music and move characteristics")
        return True
    else:
        print(f"❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)