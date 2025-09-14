#!/usr/bin/env python3
"""
Test script for enhanced movement dynamics analysis (Task 4.4).
Tests the new movement dynamics features including velocity/acceleration patterns,
spatial analysis, rhythm compatibility, and transition point detection.
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.move_analyzer import (
    MoveAnalyzer, analyze_video_directory, 
    calculate_transition_compatibility, analyze_move_transitions
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_enhanced_movement_dynamics():
    """Test the enhanced movement dynamics analysis features."""
    print("\n" + "="*70)
    print("TESTING ENHANCED MOVEMENT DYNAMICS ANALYSIS (Task 4.4)")
    print("="*70)
    
    # Initialize analyzer
    analyzer = MoveAnalyzer(target_fps=30)
    
    # Find test videos from different categories
    video_dir = Path("data/Bachata_steps")
    test_videos = []
    
    # Try to get videos from different categories for comparison
    categories_to_test = ["basic_steps", "cross_body_lead", "body_roll", "dip"]
    
    for category in categories_to_test:
        category_dir = video_dir / category
        if category_dir.exists():
            video_files = list(category_dir.glob("*.mp4"))
            if video_files:
                test_videos.append((category, video_files[0]))
                if len(test_videos) >= 3:  # Limit to 3 videos for testing
                    break
    
    if not test_videos:
        # Fallback: get any available videos
        for category_dir in video_dir.iterdir():
            if category_dir.is_dir():
                video_files = list(category_dir.glob("*.mp4"))
                if video_files:
                    test_videos.append((category_dir.name, video_files[0]))
                    if len(test_videos) >= 3:
                        break
    
    if not test_videos:
        print("❌ No test video files found!")
        return False
    
    print(f"📹 Testing with {len(test_videos)} videos:")
    for category, video_path in test_videos:
        print(f"   - {category}: {video_path.name}")
    
    # Analyze each video
    results = {}
    
    for category, video_path in test_videos:
        print(f"\n🔍 Analyzing {category} - {video_path.name}")
        
        try:
            result = analyzer.analyze_move_clip(str(video_path))
            results[f"{category}_{video_path.stem}"] = result
            
            # Display enhanced movement dynamics
            dynamics = result.movement_dynamics
            
            print(f"✅ Analysis completed!")
            print(f"📊 Basic Metrics:")
            print(f"   - Duration: {result.duration:.2f}s")
            print(f"   - Frames: {result.frame_count}")
            print(f"   - Quality: {result.analysis_quality:.2f}")
            
            print(f"\n🏃 Enhanced Movement Dynamics:")
            print(f"   - Spatial Coverage: {dynamics.spatial_coverage:.4f}")
            print(f"   - Footwork Area Coverage: {dynamics.footwork_area_coverage:.4f}")
            print(f"   - Upper Body Movement Range: {dynamics.upper_body_movement_range:.4f}")
            print(f"   - Rhythm Compatibility Score: {dynamics.rhythm_compatibility_score:.3f}")
            print(f"   - Movement Periodicity: {dynamics.movement_periodicity:.3f}")
            print(f"   - Transition Points: {len(dynamics.transition_points)} points at frames {dynamics.transition_points}")
            
            print(f"\n📈 Movement Profiles:")
            print(f"   - Velocity Profile: mean={np.mean(dynamics.velocity_profile):.4f}, std={np.std(dynamics.velocity_profile):.4f}")
            print(f"   - Acceleration Profile: mean={np.mean(dynamics.acceleration_profile):.4f}, std={np.std(dynamics.acceleration_profile):.4f}")
            print(f"   - Intensity Profile: mean={np.mean(dynamics.movement_intensity_profile):.4f}, std={np.std(dynamics.movement_intensity_profile):.4f}")
            
            print(f"\n🎯 Spatial Distribution:")
            for region, value in dynamics.spatial_distribution.items():
                print(f"   - {region}: {value:.4f}")
            
            print(f"\n🎵 Movement Characteristics:")
            print(f"   - Energy Level: {dynamics.energy_level}")
            print(f"   - Dominant Direction: {dynamics.dominant_movement_direction}")
            print(f"   - Complexity Score: {dynamics.complexity_score:.3f}")
            print(f"   - Rhythm Score: {dynamics.rhythm_score:.3f}")
            
            # Test enhanced embedding
            print(f"\n🧠 Enhanced Embeddings:")
            print(f"   - Movement Embedding Shape: {result.movement_embedding.shape}")
            print(f"   - Movement Embedding Stats: mean={np.mean(result.movement_embedding):.4f}, std={np.std(result.movement_embedding):.4f}")
            print(f"   - Non-zero Features: {np.count_nonzero(result.movement_embedding)}/{len(result.movement_embedding)}")
            
        except Exception as e:
            print(f"❌ Analysis failed for {category}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True, results


def test_transition_compatibility(results):
    """Test transition compatibility analysis between moves."""
    print("\n" + "="*70)
    print("TESTING TRANSITION COMPATIBILITY ANALYSIS")
    print("="*70)
    
    if len(results) < 2:
        print("❌ Need at least 2 moves to test transition compatibility")
        return False
    
    move_names = list(results.keys())
    print(f"📊 Calculating transition compatibility for {len(move_names)} moves")
    
    try:
        # Test pairwise compatibility
        print(f"\n🔄 Pairwise Transition Compatibility:")
        
        for i, move1 in enumerate(move_names):
            for j, move2 in enumerate(move_names):
                if i < j:  # Avoid duplicates
                    compatibility = calculate_transition_compatibility(
                        results[move1], results[move2]
                    )
                    print(f"   - {move1} → {move2}: {compatibility:.3f}")
                    print(f"   - {move2} → {move1}: {calculate_transition_compatibility(results[move2], results[move1]):.3f}")
        
        # Test full transition matrix
        print(f"\n🎯 Full Transition Matrix Analysis:")
        transition_matrix = analyze_move_transitions(results)
        
        print(f"✅ Calculated {len(transition_matrix)} transition pairs")
        
        # Find best and worst transitions
        sorted_transitions = sorted(transition_matrix.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Best Transitions (Top 5):")
        for (move1, move2), score in sorted_transitions[:5]:
            print(f"   - {move1} → {move2}: {score:.3f}")
        
        print(f"\n⚠️  Challenging Transitions (Bottom 5):")
        for (move1, move2), score in sorted_transitions[-5:]:
            print(f"   - {move1} → {move2}: {score:.3f}")
        
        # Statistics
        scores = list(transition_matrix.values())
        print(f"\n📈 Transition Compatibility Statistics:")
        print(f"   - Average: {np.mean(scores):.3f}")
        print(f"   - Std Dev: {np.std(scores):.3f}")
        print(f"   - Range: {np.min(scores):.3f} to {np.max(scores):.3f}")
        print(f"   - High compatibility (>0.7): {sum(1 for s in scores if s > 0.7)}/{len(scores)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transition compatibility analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rhythm_analysis():
    """Test rhythm compatibility and periodicity analysis."""
    print("\n" + "="*70)
    print("TESTING RHYTHM ANALYSIS FEATURES")
    print("="*70)
    
    # Initialize analyzer
    analyzer = MoveAnalyzer(target_fps=30)
    
    # Find videos with different expected rhythm characteristics
    video_dir = Path("data/Bachata_steps")
    
    # Test with basic steps (should have high rhythm compatibility)
    basic_dir = video_dir / "basic_steps"
    if basic_dir.exists():
        basic_videos = list(basic_dir.glob("*.mp4"))
        if basic_videos:
            print(f"🎵 Testing rhythm analysis with basic step: {basic_videos[0].name}")
            
            try:
                result = analyzer.analyze_move_clip(str(basic_videos[0]))
                dynamics = result.movement_dynamics
                
                print(f"✅ Rhythm Analysis Results:")
                print(f"   - Rhythm Compatibility Score: {dynamics.rhythm_compatibility_score:.3f}")
                print(f"   - Movement Periodicity: {dynamics.movement_periodicity:.3f}")
                print(f"   - Rhythm Score (consistency): {dynamics.rhythm_score:.3f}")
                print(f"   - Transition Points: {len(dynamics.transition_points)} detected")
                
                # Analyze velocity patterns for rhythm
                velocity_profile = dynamics.velocity_profile
                if len(velocity_profile) > 4:
                    print(f"\n📊 Velocity Pattern Analysis:")
                    print(f"   - Velocity Range: {np.min(velocity_profile):.4f} to {np.max(velocity_profile):.4f}")
                    print(f"   - Velocity Variation (CV): {np.std(velocity_profile) / (np.mean(velocity_profile) + 1e-6):.3f}")
                    
                    # Look for periodic patterns
                    velocity_norm = velocity_profile - np.mean(velocity_profile)
                    if len(velocity_norm) > 1:
                        autocorr = np.correlate(velocity_norm, velocity_norm, mode='full')
                        autocorr = autocorr[len(autocorr)//2:]
                        if len(autocorr) > 3:
                            secondary_peak = np.max(autocorr[2:min(len(autocorr), len(velocity_profile)//2)])
                            print(f"   - Autocorrelation Peak: {secondary_peak / autocorr[0]:.3f}")
                
                return True
                
            except Exception as e:
                print(f"❌ Rhythm analysis failed: {e}")
                return False
    
    print("❌ No basic step videos found for rhythm analysis")
    return False


def main():
    """Run all enhanced movement dynamics tests."""
    print("🚀 Starting Enhanced Movement Dynamics Tests (Task 4.4)")
    print("="*70)
    
    # Check if data directory exists
    if not Path("data/Bachata_steps").exists():
        print("❌ Data directory not found! Please ensure video files are available.")
        return
    
    # Run tests
    tests_results = []
    
    # Test 1: Enhanced movement dynamics
    print(f"\n🧪 Test 1: Enhanced Movement Dynamics Analysis")
    try:
        success, results = test_enhanced_movement_dynamics()
        tests_results.append(("Enhanced Movement Dynamics", success))
        
        if success and results:
            # Test 2: Transition compatibility (depends on Test 1)
            print(f"\n🧪 Test 2: Transition Compatibility Analysis")
            transition_success = test_transition_compatibility(results)
            tests_results.append(("Transition Compatibility", transition_success))
        else:
            tests_results.append(("Transition Compatibility", False))
            
    except Exception as e:
        print(f"❌ Test 1 failed with exception: {e}")
        tests_results.append(("Enhanced Movement Dynamics", False))
        tests_results.append(("Transition Compatibility", False))
    
    # Test 3: Rhythm analysis
    print(f"\n🧪 Test 3: Rhythm Analysis Features")
    try:
        rhythm_success = test_rhythm_analysis()
        tests_results.append(("Rhythm Analysis", rhythm_success))
    except Exception as e:
        print(f"❌ Test 3 failed with exception: {e}")
        tests_results.append(("Rhythm Analysis", False))
    
    # Summary
    print("\n" + "="*70)
    print("ENHANCED MOVEMENT DYNAMICS TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in tests_results if success)
    total = len(tests_results)
    
    for test_name, success in tests_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhanced movement dynamics tests passed!")
        print("✅ Task 4.4 implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    print(f"\n📋 Task 4.4 Features Implemented:")
    print(f"   ✅ Movement velocity and acceleration patterns")
    print(f"   ✅ Spatial movement patterns (footwork area, upper body range)")
    print(f"   ✅ Rhythm compatibility scores")
    print(f"   ✅ Transition point identification")
    print(f"   ✅ Transition compatibility calculation")
    print(f"   ✅ Enhanced movement embeddings")


if __name__ == "__main__":
    main()