#!/usr/bin/env python3
"""
Test script for the MoveAnalyzer service.
Tests MediaPipe pose detection and feature extraction on sample video clips.
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.move_analyzer import MoveAnalyzer, analyze_video_directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_video_analysis():
    """Test analysis of a single video file."""
    print("\n" + "="*60)
    print("TESTING SINGLE VIDEO ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MoveAnalyzer(target_fps=30)
    
    # Find a test video file
    video_dir = Path("data/Bachata_steps")
    test_video = None
    
    # Look for a basic step video first
    for category_dir in video_dir.iterdir():
        if category_dir.is_dir() and "basic" in category_dir.name.lower():
            video_files = list(category_dir.glob("*.mp4"))
            if video_files:
                test_video = video_files[0]
                break
    
    # If no basic step found, use any video
    if not test_video:
        for category_dir in video_dir.iterdir():
            if category_dir.is_dir():
                video_files = list(category_dir.glob("*.mp4"))
                if video_files:
                    test_video = video_files[0]
                    break
    
    if not test_video:
        print("❌ No test video files found!")
        return False
    
    print(f"📹 Testing with video: {test_video}")
    
    try:
        # Analyze the video
        result = analyzer.analyze_move_clip(str(test_video))
        
        # Print results
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Video Duration: {result.duration:.2f} seconds")
        print(f"📊 Frames Analyzed: {result.frame_count}")
        print(f"📊 Target FPS: {result.fps}")
        print(f"📊 Pose Detection Rate: {result.pose_detection_rate:.2%}")
        print(f"📊 Analysis Quality: {result.analysis_quality:.2f}")
        
        # Movement dynamics
        dynamics = result.movement_dynamics
        print(f"\n🏃 Movement Dynamics:")
        print(f"   - Spatial Coverage: {dynamics.spatial_coverage:.4f}")
        print(f"   - Rhythm Score: {dynamics.rhythm_score:.2f}")
        print(f"   - Complexity Score: {dynamics.complexity_score:.2f}")
        print(f"   - Energy Level: {dynamics.energy_level}")
        print(f"   - Dominant Direction: {dynamics.dominant_movement_direction}")
        
        # Embedding dimensions
        print(f"\n🧠 Feature Embeddings:")
        print(f"   - Pose Embedding: {result.pose_embedding.shape}")
        print(f"   - Movement Embedding: {result.movement_embedding.shape}")
        
        # Sample pose features
        if result.pose_features:
            sample_pose = result.pose_features[0]
            print(f"\n🤸 Sample Pose Features:")
            print(f"   - Landmarks Shape: {sample_pose.landmarks.shape}")
            print(f"   - Joint Angles: {list(sample_pose.joint_angles.keys())}")
            print(f"   - Center of Mass: ({sample_pose.center_of_mass[0]:.3f}, {sample_pose.center_of_mass[1]:.3f})")
            print(f"   - Confidence: {sample_pose.confidence:.2f}")
        
        # Hand detection results
        hand_detections = sum(1 for hf in result.hand_features 
                            if hf.left_hand_landmarks is not None or hf.right_hand_landmarks is not None)
        print(f"\n👋 Hand Detection:")
        print(f"   - Frames with hands detected: {hand_detections}/{len(result.hand_features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_analysis():
    """Test batch analysis of multiple videos."""
    print("\n" + "="*60)
    print("TESTING BATCH ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MoveAnalyzer(target_fps=15)  # Lower FPS for faster batch processing
    
    # Test with basic_steps directory
    test_dir = Path("data/Bachata_steps/basic_steps")
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    print(f"📁 Testing batch analysis in: {test_dir}")
    
    try:
        # Analyze all videos in directory
        results = analyze_video_directory(str(test_dir), analyzer)
        
        print(f"\n✅ Batch analysis completed!")
        print(f"📊 Videos processed: {len(results)}")
        
        # Summary statistics
        if results:
            qualities = [r.analysis_quality for r in results.values()]
            detection_rates = [r.pose_detection_rate for r in results.values()]
            durations = [r.duration for r in results.values()]
            
            print(f"\n📈 Summary Statistics:")
            print(f"   - Average Quality: {np.mean(qualities):.2f} ± {np.std(qualities):.2f}")
            print(f"   - Average Detection Rate: {np.mean(detection_rates):.2%} ± {np.std(detection_rates):.2%}")
            print(f"   - Average Duration: {np.mean(durations):.1f}s ± {np.std(durations):.1f}s")
            
            # Individual results
            print(f"\n📋 Individual Results:")
            for filename, result in results.items():
                print(f"   - {filename}: Quality={result.analysis_quality:.2f}, "
                      f"Detection={result.pose_detection_rate:.1%}, "
                      f"Duration={result.duration:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction_details():
    """Test detailed feature extraction capabilities."""
    print("\n" + "="*60)
    print("TESTING DETAILED FEATURE EXTRACTION")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MoveAnalyzer(target_fps=10)  # Lower FPS for detailed analysis
    
    # Find a test video
    video_dir = Path("data/Bachata_steps")
    test_video = None
    
    for category_dir in video_dir.iterdir():
        if category_dir.is_dir():
            video_files = list(category_dir.glob("*.mp4"))
            if video_files:
                test_video = video_files[0]
                break
    
    if not test_video:
        print("❌ No test video files found!")
        return False
    
    print(f"📹 Detailed analysis of: {test_video}")
    
    try:
        result = analyzer.analyze_move_clip(str(test_video))
        
        # Analyze pose features in detail
        print(f"\n🔍 Detailed Pose Analysis:")
        
        if result.pose_features:
            # Joint angle analysis
            all_angles = {
                'left_elbow': [],
                'right_elbow': [],
                'left_knee': [],
                'right_knee': [],
                'torso_lean': []
            }
            
            for pose_feat in result.pose_features:
                for angle_name in all_angles.keys():
                    all_angles[angle_name].append(pose_feat.joint_angles.get(angle_name, 180.0))
            
            print(f"   Joint Angle Statistics:")
            for angle_name, angles in all_angles.items():
                print(f"     - {angle_name}: {np.mean(angles):.1f}° ± {np.std(angles):.1f}°")
            
            # Movement trajectory analysis
            centers = [pf.center_of_mass for pf in result.pose_features]
            x_coords = [c[0] for c in centers]
            y_coords = [c[1] for c in centers]
            
            print(f"\n   Movement Trajectory:")
            print(f"     - X movement: {np.min(x_coords):.3f} to {np.max(x_coords):.3f} (range: {np.max(x_coords)-np.min(x_coords):.3f})")
            print(f"     - Y movement: {np.min(y_coords):.3f} to {np.max(y_coords):.3f} (range: {np.max(y_coords)-np.min(y_coords):.3f})")
            
            # Confidence analysis
            confidences = [pf.confidence for pf in result.pose_features]
            print(f"\n   Detection Confidence:")
            print(f"     - Average: {np.mean(confidences):.2f}")
            print(f"     - Range: {np.min(confidences):.2f} to {np.max(confidences):.2f}")
            print(f"     - Frames with high confidence (>0.8): {sum(1 for c in confidences if c > 0.8)}/{len(confidences)}")
        
        # Hand detection analysis
        print(f"\n👋 Hand Detection Analysis:")
        left_detections = sum(1 for hf in result.hand_features if hf.left_hand_landmarks is not None)
        right_detections = sum(1 for hf in result.hand_features if hf.right_hand_landmarks is not None)
        
        print(f"   - Left hand detected: {left_detections}/{len(result.hand_features)} frames")
        print(f"   - Right hand detected: {right_detections}/{len(result.hand_features)} frames")
        
        if result.hand_features:
            left_confidences = [hf.left_hand_confidence for hf in result.hand_features if hf.left_hand_confidence > 0]
            right_confidences = [hf.right_hand_confidence for hf in result.hand_features if hf.right_hand_confidence > 0]
            
            if left_confidences:
                print(f"   - Left hand avg confidence: {np.mean(left_confidences):.2f}")
            if right_confidences:
                print(f"   - Right hand avg confidence: {np.mean(right_confidences):.2f}")
        
        # Embedding analysis
        print(f"\n🧠 Embedding Analysis:")
        print(f"   - Pose embedding stats: mean={np.mean(result.pose_embedding):.4f}, std={np.std(result.pose_embedding):.4f}")
        print(f"   - Movement embedding stats: mean={np.mean(result.movement_embedding):.4f}, std={np.std(result.movement_embedding):.4f}")
        print(f"   - Non-zero pose features: {np.count_nonzero(result.pose_embedding)}/{len(result.pose_embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting MoveAnalyzer Tests")
    print("="*60)
    
    # Check if data directory exists
    if not Path("data/Bachata_steps").exists():
        print("❌ Data directory not found! Please ensure video files are available.")
        return
    
    # Run tests
    tests = [
        ("Single Video Analysis", test_single_video_analysis),
        ("Batch Analysis", test_batch_analysis),
        ("Detailed Feature Extraction", test_feature_extraction_details),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MoveAnalyzer is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()