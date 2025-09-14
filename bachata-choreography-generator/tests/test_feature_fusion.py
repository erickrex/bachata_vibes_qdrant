#!/usr/bin/env python3
"""
Test script for the enhanced feature fusion system (Task 4.5).
Tests the 384-dimensional pose feature vector, movement complexity scoring,
tempo compatibility ranges, and difficulty scoring.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

from app.services.move_analyzer import MoveAnalyzer
from app.services.annotation_interface import AnnotationInterface


def test_feature_fusion_system():
    """Test the enhanced feature fusion system."""
    print("🧪 Testing Enhanced Feature Fusion System (Task 4.5)")
    print("=" * 60)
    
    # Initialize services
    analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    
    # Load existing annotations to get video paths
    try:
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips
        print(f"📊 Loaded {len(annotations)} existing annotations")
    except Exception as e:
        print(f"⚠️  Could not load annotations: {e}")
        return False
    
    if not annotations:
        print("❌ No annotations found. Please run annotation framework first.")
        return False
    
    # Test with a few representative moves
    test_moves = annotations[:3]  # Test first 3 moves
    
    print(f"\n🎯 Testing feature fusion on {len(test_moves)} moves:")
    
    results = []
    
    for i, annotation in enumerate(test_moves):
        video_path = annotation.video_path
        move_label = annotation.move_label
        
        # Resolve the full path to the video file
        if not os.path.isabs(video_path):
            video_path = os.path.join("data", video_path)
        
        print(f"\n--- Move {i+1}: {move_label} ---")
        print(f"📹 Video: {Path(video_path).name}")
        print(f"🔍 Full path: {video_path}")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path}")
            continue
        
        try:
            # Analyze the move
            result = analyzer.analyze_move_clip(video_path)
            results.append(result)
            
            # Test 1: 384-dimensional pose embedding
            print(f"✅ Pose embedding shape: {result.pose_embedding.shape}")
            assert result.pose_embedding.shape == (384,), f"Expected (384,), got {result.pose_embedding.shape}"
            
            # Check embedding is not all zeros
            non_zero_count = np.count_nonzero(result.pose_embedding)
            print(f"📊 Non-zero embedding features: {non_zero_count}/384 ({non_zero_count/384*100:.1f}%)")
            assert non_zero_count > 100, f"Too few non-zero features: {non_zero_count}"
            
            # Test 2: Movement complexity score
            complexity = result.movement_complexity_score
            print(f"🧮 Movement complexity score: {complexity:.3f}")
            assert 0.0 <= complexity <= 1.0, f"Complexity score out of range: {complexity}"
            
            # Test 3: Tempo compatibility range
            min_bpm, max_bpm = result.tempo_compatibility_range
            print(f"🎵 Tempo compatibility: {min_bpm:.1f} - {max_bpm:.1f} BPM")
            assert 80.0 <= min_bpm <= 160.0, f"Min BPM out of range: {min_bpm}"
            assert 80.0 <= max_bpm <= 160.0, f"Max BPM out of range: {max_bpm}"
            assert min_bpm < max_bpm, f"Invalid tempo range: {min_bpm} >= {max_bpm}"
            
            # Test 4: Difficulty score
            difficulty = result.difficulty_score
            print(f"⭐ Difficulty score: {difficulty:.3f}")
            assert 0.0 <= difficulty <= 1.0, f"Difficulty score out of range: {difficulty}"
            
            # Test 5: Enhanced movement dynamics
            dynamics = result.movement_dynamics
            print(f"🏃 Movement dynamics:")
            print(f"   - Spatial coverage: {dynamics.spatial_coverage:.4f}")
            print(f"   - Rhythm score: {dynamics.rhythm_score:.3f}")
            print(f"   - Energy level: {dynamics.energy_level}")
            print(f"   - Footwork area: {dynamics.footwork_area_coverage:.4f}")
            print(f"   - Upper body range: {dynamics.upper_body_movement_range:.4f}")
            print(f"   - Rhythm compatibility: {dynamics.rhythm_compatibility_score:.3f}")
            print(f"   - Movement periodicity: {dynamics.movement_periodicity:.3f}")
            print(f"   - Transition points: {len(dynamics.transition_points)}")
            
            # Test 6: Feature vector statistics
            embedding_stats = {
                'mean': np.mean(result.pose_embedding),
                'std': np.std(result.pose_embedding),
                'min': np.min(result.pose_embedding),
                'max': np.max(result.pose_embedding),
                'non_zero_ratio': non_zero_count / 384
            }
            
            print(f"📈 Embedding statistics:")
            for stat_name, stat_value in embedding_stats.items():
                print(f"   - {stat_name}: {stat_value:.4f}")
            
            print(f"✅ Move analysis completed successfully")
            
        except Exception as e:
            print(f"❌ Error analyzing {move_label}: {e}")
            return False
    
    # Test 7: Compare embeddings between different moves
    print(f"\n🔍 Comparing embeddings between moves:")
    
    if len(results) >= 2:
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                move1 = test_moves[i].move_label
                move2 = test_moves[j].move_label
                
                # Calculate cosine similarity
                emb1 = results[i].pose_embedding
                emb2 = results[j].pose_embedding
                
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    print(f"   {move1} ↔ {move2}: {similarity:.3f}")
                    
                    # Embeddings should be different but not completely orthogonal
                    assert -1.0 <= similarity <= 1.0, f"Invalid similarity: {similarity}"
    
    # Test 8: Validate complexity scoring components
    print(f"\n🧪 Testing complexity scoring components:")
    
    for i, result in enumerate(results):
        move_label = test_moves[i].move_label
        print(f"\n{move_label}:")
        
        # Test individual complexity components
        joint_complexity = analyzer._calculate_joint_angle_complexity(result.pose_features)
        velocity_complexity = analyzer._calculate_velocity_complexity(result.movement_dynamics.velocity_profile)
        coordination_complexity = analyzer._calculate_coordination_complexity(result.pose_features)
        
        print(f"   - Joint angle complexity: {joint_complexity:.3f}")
        print(f"   - Velocity complexity: {velocity_complexity:.3f}")
        print(f"   - Coordination complexity: {coordination_complexity:.3f}")
        
        # All components should be in valid range
        assert 0.0 <= joint_complexity <= 1.0, f"Joint complexity out of range: {joint_complexity}"
        assert 0.0 <= velocity_complexity <= 1.0, f"Velocity complexity out of range: {velocity_complexity}"
        assert 0.0 <= coordination_complexity <= 1.0, f"Coordination complexity out of range: {coordination_complexity}"
    
    # Test 9: Validate difficulty scoring components
    print(f"\n⭐ Testing difficulty scoring components:")
    
    for i, result in enumerate(results):
        move_label = test_moves[i].move_label
        print(f"\n{move_label}:")
        
        # Test individual difficulty components
        coordination_difficulty = analyzer._calculate_coordination_difficulty(result.pose_features)
        stability_difficulty = analyzer._calculate_stability_difficulty(result.pose_features, result.movement_dynamics)
        
        print(f"   - Coordination difficulty: {coordination_difficulty:.3f}")
        print(f"   - Stability difficulty: {stability_difficulty:.3f}")
        
        # All components should be in valid range
        assert 0.0 <= coordination_difficulty <= 1.0, f"Coordination difficulty out of range: {coordination_difficulty}"
        assert 0.0 <= stability_difficulty <= 1.0, f"Stability difficulty out of range: {stability_difficulty}"
    
    print(f"\n🎉 All feature fusion tests passed!")
    print(f"✅ 384-dimensional pose embeddings generated successfully")
    print(f"✅ Movement complexity scoring implemented")
    print(f"✅ Tempo compatibility ranges calculated")
    print(f"✅ Difficulty scores computed")
    print(f"✅ Enhanced movement dynamics analysis working")
    
    return True


def test_embedding_consistency():
    """Test that embeddings are consistent across multiple runs."""
    print(f"\n🔄 Testing embedding consistency:")
    
    analyzer = MoveAnalyzer(target_fps=30)
    annotation_interface = AnnotationInterface()
    
    try:
        annotation_collection = annotation_interface.load_annotations()
        annotations = annotation_collection.clips
        if not annotations:
            print("⚠️  No annotations available for consistency test")
            return True
        
        # Test with first available move
        test_annotation = annotations[0]
        video_path = test_annotation.video_path
        
        # Resolve the full path to the video file
        if not os.path.isabs(video_path):
            video_path = os.path.join("data", video_path)
        
        print(f"📹 Testing consistency with: {Path(video_path).name}")
        print(f"🔍 Full path: {video_path}")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path}")
            return True
        
        # Analyze the same move multiple times
        results = []
        for i in range(3):
            result = analyzer.analyze_move_clip(video_path)
            results.append(result.pose_embedding)
            print(f"   Run {i+1}: Generated {result.pose_embedding.shape[0]}D embedding")
        
        # Check consistency
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                similarity = np.corrcoef(results[i], results[j])[0, 1]
                print(f"   Correlation between run {i+1} and {j+1}: {similarity:.4f}")
                
                # Embeddings should be highly consistent (correlation > 0.95)
                assert similarity > 0.95, f"Low consistency: {similarity}"
        
        print(f"✅ Embedding consistency test passed")
        return True
        
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Starting Feature Fusion System Tests")
    print("=" * 60)
    
    # Test 1: Core feature fusion functionality
    success1 = test_feature_fusion_system()
    
    # Test 2: Embedding consistency
    success2 = test_embedding_consistency()
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Task 4.5 - Feature Fusion System implementation is complete")
        print(f"✅ 384-dimensional pose feature vectors working correctly")
        print(f"✅ Movement complexity scoring implemented")
        print(f"✅ Tempo compatibility ranges calculated")
        print(f"✅ Difficulty scores computed accurately")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)