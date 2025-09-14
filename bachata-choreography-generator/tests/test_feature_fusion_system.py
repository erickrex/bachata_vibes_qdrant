#!/usr/bin/env python3
"""
Test script for the feature fusion system (Task 5.1).
Tests the 512-dimensional combined feature vector creation and quality metrics.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.feature_fusion import FeatureFusion, MultiModalEmbedding
from app.services.music_analyzer import MusicAnalyzer
from app.services.move_analyzer import MoveAnalyzer
from app.services.annotation_interface import AnnotationInterface


def test_audio_embedding_creation():
    """Test 128-dimensional audio embedding creation."""
    print("🎵 Testing Audio Embedding Creation")
    print("-" * 40)
    
    fusion = FeatureFusion()
    music_analyzer = MusicAnalyzer()
    
    # Find a test audio file
    audio_files = list(Path("data/songs").glob("*.mp3"))
    if not audio_files:
        print("⚠️  No audio files found in data/songs/")
        return False
    
    test_audio = str(audio_files[0])
    print(f"📁 Testing with: {Path(test_audio).name}")
    
    try:
        # Analyze audio
        music_features = music_analyzer.analyze_audio(test_audio)
        print(f"✅ Music analysis completed")
        
        # Create audio embedding
        audio_embedding = fusion.create_audio_embedding(music_features)
        
        # Test embedding properties
        print(f"📊 Audio embedding shape: {audio_embedding.shape}")
        assert audio_embedding.shape == (128,), f"Expected (128,), got {audio_embedding.shape}"
        
        # Check embedding is normalized
        norm = np.linalg.norm(audio_embedding)
        print(f"📏 Embedding norm: {norm:.4f}")
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: {norm}"
        
        # Check for non-zero features
        non_zero_count = np.count_nonzero(audio_embedding)
        print(f"🔢 Non-zero features: {non_zero_count}/128 ({non_zero_count/128*100:.1f}%)")
        assert non_zero_count > 50, f"Too few non-zero features: {non_zero_count}"
        
        # Check embedding statistics
        print(f"📈 Embedding stats:")
        print(f"   Mean: {np.mean(audio_embedding):.4f}")
        print(f"   Std:  {np.std(audio_embedding):.4f}")
        print(f"   Min:  {np.min(audio_embedding):.4f}")
        print(f"   Max:  {np.max(audio_embedding):.4f}")
        
        print(f"✅ Audio embedding creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Audio embedding test failed: {e}")
        return False


def test_pose_embedding_aggregation():
    """Test pose embedding aggregation from MediaPipe landmarks."""
    print("\n🕺 Testing Pose Embedding Aggregation")
    print("-" * 40)
    
    fusion = FeatureFusion()
    move_analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    
    try:
        # Load annotations to get video paths
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips
        
        if not annotations:
            print("⚠️  No annotations found")
            return False
        
        # Test with first available video
        test_annotation = annotations[0]
        video_path = test_annotation.video_path
        
        # Resolve full path
        if not os.path.isabs(video_path):
            video_path = os.path.join("data", video_path)
        
        print(f"📹 Testing with: {Path(video_path).name}")
        
        if not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path}")
            return False
        
        # Analyze move
        move_result = move_analyzer.analyze_move_clip(video_path)
        print(f"✅ Move analysis completed")
        
        # Create pose embedding aggregation
        pose_embedding = fusion.create_pose_embedding_aggregation(move_result)
        
        # Test embedding properties
        print(f"📊 Pose embedding shape: {pose_embedding.shape}")
        assert pose_embedding.shape == (384,), f"Expected (384,), got {pose_embedding.shape}"
        
        # Check embedding is normalized
        norm = np.linalg.norm(pose_embedding)
        print(f"📏 Embedding norm: {norm:.4f}")
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: {norm}"
        
        # Check for non-zero features
        non_zero_count = np.count_nonzero(pose_embedding)
        print(f"🔢 Non-zero features: {non_zero_count}/384 ({non_zero_count/384*100:.1f}%)")
        assert non_zero_count > 100, f"Too few non-zero features: {non_zero_count}"
        
        # Check embedding statistics
        print(f"📈 Embedding stats:")
        print(f"   Mean: {np.mean(pose_embedding):.4f}")
        print(f"   Std:  {np.std(pose_embedding):.4f}")
        print(f"   Min:  {np.min(pose_embedding):.4f}")
        print(f"   Max:  {np.max(pose_embedding):.4f}")
        
        print(f"✅ Pose embedding aggregation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pose embedding test failed: {e}")
        return False


def test_multimodal_embedding_creation():
    """Test 512-dimensional multimodal embedding creation."""
    print("\n🎭 Testing Multimodal Embedding Creation")
    print("-" * 40)
    
    fusion = FeatureFusion()
    music_analyzer = MusicAnalyzer()
    move_analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    
    try:
        # Get test audio file
        audio_files = list(Path("data/songs").glob("*.mp3"))
        if not audio_files:
            print("⚠️  No audio files found")
            return False
        
        test_audio = str(audio_files[0])
        
        # Get test video file
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips
        
        if not annotations:
            print("⚠️  No annotations found")
            return False
        
        test_annotation = annotations[0]
        video_path = test_annotation.video_path
        
        if not os.path.isabs(video_path):
            video_path = os.path.join("data", video_path)
        
        if not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path}")
            return False
        
        print(f"🎵 Audio: {Path(test_audio).name}")
        print(f"📹 Video: {Path(video_path).name}")
        
        # Analyze both modalities
        music_features = music_analyzer.analyze_audio(test_audio)
        move_result = move_analyzer.analyze_move_clip(video_path)
        
        print(f"✅ Both analyses completed")
        
        # Create multimodal embedding
        multimodal_embedding = fusion.create_multimodal_embedding(music_features, move_result)
        
        # Test multimodal embedding properties
        print(f"📊 Combined embedding shape: {multimodal_embedding.combined_embedding.shape}")
        assert multimodal_embedding.combined_embedding.shape == (512,), f"Expected (512,), got shape"
        
        print(f"📊 Audio embedding shape: {multimodal_embedding.audio_embedding.shape}")
        assert multimodal_embedding.audio_embedding.shape == (128,), f"Expected (128,), got shape"
        
        print(f"📊 Pose embedding shape: {multimodal_embedding.pose_embedding.shape}")
        assert multimodal_embedding.pose_embedding.shape == (384,), f"Expected (384,), got shape"
        
        # Check combined embedding is normalized
        norm = np.linalg.norm(multimodal_embedding.combined_embedding)
        print(f"📏 Combined embedding norm: {norm:.4f}")
        assert abs(norm - 1.0) < 0.01, f"Combined embedding not normalized: {norm}"
        
        # Check for non-zero features
        non_zero_count = np.count_nonzero(multimodal_embedding.combined_embedding)
        print(f"🔢 Non-zero features: {non_zero_count}/512 ({non_zero_count/512*100:.1f}%)")
        assert non_zero_count > 200, f"Too few non-zero features: {non_zero_count}"
        
        # Test metadata
        print(f"📝 Metadata:")
        print(f"   Audio source: {Path(multimodal_embedding.audio_source or 'None').name}")
        print(f"   Move source: {Path(multimodal_embedding.move_source).name}")
        print(f"   Version: {multimodal_embedding.embedding_version}")
        
        # Check embedding statistics
        print(f"📈 Combined embedding stats:")
        print(f"   Mean: {np.mean(multimodal_embedding.combined_embedding):.4f}")
        print(f"   Std:  {np.std(multimodal_embedding.combined_embedding):.4f}")
        print(f"   Min:  {np.min(multimodal_embedding.combined_embedding):.4f}")
        print(f"   Max:  {np.max(multimodal_embedding.combined_embedding):.4f}")
        
        print(f"✅ Multimodal embedding creation test passed")
        return multimodal_embedding
        
    except Exception as e:
        print(f"❌ Multimodal embedding test failed: {e}")
        return False


def test_similarity_calculation():
    """Test similarity calculation between embeddings."""
    print("\n🔍 Testing Similarity Calculation")
    print("-" * 40)
    
    fusion = FeatureFusion()
    music_analyzer = MusicAnalyzer()
    move_analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    
    try:
        # Get test files
        audio_files = list(Path("data/songs").glob("*.mp3"))[:2]
        if len(audio_files) < 2:
            print("⚠️  Need at least 2 audio files for similarity testing")
            return False
        
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:2]
        
        if len(annotations) < 2:
            print("⚠️  Need at least 2 video annotations for similarity testing")
            return False
        
        # Create two multimodal embeddings
        embeddings = []
        
        for i, (audio_file, annotation) in enumerate(zip(audio_files, annotations)):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                print(f"⚠️  Video file not found: {video_path}")
                continue
            
            print(f"🎭 Creating embedding {i+1}:")
            print(f"   Audio: {Path(audio_file).name}")
            print(f"   Video: {Path(video_path).name}")
            
            music_features = music_analyzer.analyze_audio(str(audio_file))
            move_result = move_analyzer.analyze_move_clip(video_path)
            
            embedding = fusion.create_multimodal_embedding(music_features, move_result)
            embeddings.append(embedding)
        
        if len(embeddings) < 2:
            print("⚠️  Could not create enough embeddings for similarity testing")
            return False
        
        # Calculate similarity
        similarity_score = fusion.calculate_similarity(embeddings[0], embeddings[1])
        
        print(f"📊 Similarity Results:")
        print(f"   Overall score: {similarity_score.overall_score:.4f}")
        print(f"   Audio similarity: {similarity_score.audio_similarity:.4f}")
        print(f"   Pose similarity: {similarity_score.pose_similarity:.4f}")
        print(f"   Weights used: {similarity_score.weights}")
        
        # Test similarity properties
        assert -1.0 <= similarity_score.overall_score <= 1.0, f"Invalid overall score: {similarity_score.overall_score}"
        assert -1.0 <= similarity_score.audio_similarity <= 1.0, f"Invalid audio similarity: {similarity_score.audio_similarity}"
        assert -1.0 <= similarity_score.pose_similarity <= 1.0, f"Invalid pose similarity: {similarity_score.pose_similarity}"
        
        # Test self-similarity (should be close to 1.0)
        self_similarity = fusion.calculate_similarity(embeddings[0], embeddings[0])
        print(f"🔄 Self-similarity: {self_similarity.overall_score:.4f}")
        assert self_similarity.overall_score > 0.99, f"Self-similarity too low: {self_similarity.overall_score}"
        
        print(f"✅ Similarity calculation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Similarity calculation test failed: {e}")
        return False


def test_embedding_quality_metrics():
    """Test embedding quality using similarity metrics."""
    print("\n📊 Testing Embedding Quality Metrics")
    print("-" * 40)
    
    fusion = FeatureFusion()
    music_analyzer = MusicAnalyzer()
    move_analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    
    try:
        # Get test files
        audio_files = list(Path("data/songs").glob("*.mp3"))[:3]
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips[:3]
        
        if len(audio_files) < 2 or len(annotations) < 2:
            print("⚠️  Need at least 2 files for quality testing")
            return False
        
        # Create multiple embeddings
        embeddings = []
        labels = []
        
        for i, (audio_file, annotation) in enumerate(zip(audio_files, annotations)):
            video_path = annotation.video_path
            if not os.path.isabs(video_path):
                video_path = os.path.join("data", video_path)
            
            if not os.path.exists(video_path):
                continue
            
            print(f"🎭 Creating embedding {i+1}")
            
            music_features = music_analyzer.analyze_audio(str(audio_file))
            move_result = move_analyzer.analyze_move_clip(video_path)
            
            embedding = fusion.create_multimodal_embedding(music_features, move_result)
            embeddings.append(embedding)
            labels.append(f"{Path(audio_file).stem}_{annotation.move_label}")
        
        if len(embeddings) < 2:
            print("⚠️  Could not create enough embeddings for quality testing")
            return False
        
        # Test embedding quality
        quality_metrics = fusion.test_embedding_quality(embeddings, labels)
        
        print(f"📊 Quality Metrics:")
        for metric, value in quality_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # Test quality thresholds
        assert quality_metrics['dimensionality_utilization'] > 0.3, f"Low dimensionality utilization: {quality_metrics['dimensionality_utilization']}"
        assert quality_metrics['embedding_variance'] > 0.0, f"Negative embedding variance: {quality_metrics['embedding_variance']}"
        assert -1.0 <= quality_metrics['audio_pose_correlation'] <= 1.0, f"Invalid correlation: {quality_metrics['audio_pose_correlation']}"
        
        print(f"✅ Embedding quality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Embedding quality test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Feature Fusion System Tests (Task 5.1)")
    print("=" * 60)
    
    # We're already in the correct directory
    
    tests = [
        ("Audio Embedding Creation", test_audio_embedding_creation),
        ("Pose Embedding Aggregation", test_pose_embedding_aggregation),
        ("Multimodal Embedding Creation", test_multimodal_embedding_creation),
        ("Similarity Calculation", test_similarity_calculation),
        ("Embedding Quality Metrics", test_embedding_quality_metrics)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
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
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"✅ Task 5.1 - Feature Fusion System implementation is complete")
        print(f"✅ 512-dimensional combined feature vector (128D audio + 384D pose)")
        print(f"✅ Audio feature extraction pipeline using MFCC, Chroma, and Tonnetz")
        print(f"✅ Pose feature aggregation from MediaPipe landmarks")
        print(f"✅ Embedding quality testing with similarity metrics")
        return True
    else:
        print(f"❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)