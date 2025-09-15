#!/usr/bin/env python3
"""
Validation runner script for the Bachata Choreography Generator.
Provides options to run different levels of validation testing.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path


def run_basic_validation():
    """Run basic pipeline validation."""
    print("🔧 Running Basic Pipeline Validation...")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "validate_basic_pipeline.py"
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], timeout=300)  # 5 minute timeout
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Basic validation timed out")
        return False
    except Exception as e:
        print(f"❌ Basic validation failed: {e}")
        return False


def run_end_to_end_validation(use_youtube=False, save_results=None):
    """Run full end-to-end validation."""
    print("🎵 Running End-to-End Validation...")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "validate_end_to_end.py"
    
    cmd = [sys.executable, str(script_path)]
    
    if use_youtube:
        cmd.append("--use-youtube")
    
    if save_results:
        cmd.extend(["--save-results", save_results])
    
    try:
        result = subprocess.run(cmd, timeout=600)  # 10 minute timeout
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ End-to-end validation timed out")
        return False
    except Exception as e:
        print(f"❌ End-to-end validation failed: {e}")
        return False


def run_quick_check():
    """Run a quick system check."""
    print("⚡ Quick System Check")
    print("=" * 30)
    
    checks = {
        "Python version": sys.version_info >= (3, 8),
        "Data directory": Path("data").exists(),
        "Video clips": len(list(Path("data/Bachata_steps").rglob("*.mp4"))) > 0 if Path("data/Bachata_steps").exists() else False,
        "Temp directory": True  # We'll create it
    }
    
    # Create temp directory
    Path("data/temp").mkdir(parents=True, exist_ok=True)
    
    # Check dependencies
    deps_to_check = ['numpy', 'cv2', 'librosa', 'mediapipe']
    for dep in deps_to_check:
        try:
            __import__(dep)
            checks[f"{dep} module"] = True
        except ImportError:
            checks[f"{dep} module"] = False
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        checks["FFmpeg"] = result.returncode == 0
    except:
        checks["FFmpeg"] = False
    
    # Print results
    passed = 0
    for check, status in checks.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check}")
        if status:
            passed += 1
    
    print(f"\n📊 {passed}/{len(checks)} checks passed")
    
    if passed >= len(checks) * 0.7:  # 70% pass rate
        print("🎉 System looks good for validation!")
        return True
    else:
        print("⚠️  Some issues detected. Check missing dependencies.")
        return False


def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(
        description="Validation runner for Bachata Choreography Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_validation.py --quick          # Quick system check
  python run_validation.py --basic          # Basic pipeline validation
  python run_validation.py --full           # Full end-to-end validation
  python run_validation.py --full --youtube # Use YouTube for testing
  python run_validation.py --all            # Run all validations
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                       help="Run quick system check")
    parser.add_argument("--basic", action="store_true",
                       help="Run basic pipeline validation")
    parser.add_argument("--full", action="store_true",
                       help="Run full end-to-end validation")
    parser.add_argument("--all", action="store_true",
                       help="Run all validation levels")
    parser.add_argument("--youtube", action="store_true",
                       help="Use YouTube URL for end-to-end testing")
    parser.add_argument("--save-results", type=str,
                       help="Save end-to-end results to JSON file")
    
    args = parser.parse_args()
    
    # If no specific validation requested, run quick check
    if not any([args.quick, args.basic, args.full, args.all]):
        args.quick = True
    
    start_time = time.time()
    results = {}
    
    print("🎵 BACHATA CHOREOGRAPHY GENERATOR - VALIDATION RUNNER")
    print("=" * 60)
    
    # Run requested validations
    if args.quick or args.all:
        print("\n" + "🔍 QUICK CHECK" + "\n")
        results['quick_check'] = run_quick_check()
    
    if args.basic or args.all:
        print("\n" + "🔧 BASIC VALIDATION" + "\n")
        results['basic_validation'] = run_basic_validation()
    
    if args.full or args.all:
        print("\n" + "🎵 END-TO-END VALIDATION" + "\n")
        results['end_to_end_validation'] = run_end_to_end_validation(
            use_youtube=args.youtube,
            save_results=args.save_results
        )
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION RUNNER SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} validations passed")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    if passed_tests == total_tests:
        print("\n🎉 All validations passed! System is ready.")
        if not args.full and not args.all:
            print("💡 Consider running full end-to-end validation: --full")
    elif passed_tests > 0:
        print("\n⚠️  Some validations passed. Check failed tests above.")
    else:
        print("\n❌ All validations failed. Check system setup.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if 'quick_check' in results and not results['quick_check']:
        print("   • Install missing dependencies")
        print("   • Ensure data directory structure is correct")
    
    if 'basic_validation' in results and not results['basic_validation']:
        print("   • Check video file accessibility")
        print("   • Verify FFmpeg installation")
    
    if 'end_to_end_validation' in results and not results['end_to_end_validation']:
        print("   • Check all services are properly implemented")
        print("   • Verify annotation files exist")
        print("   • Test with simpler inputs first")
    
    if all(results.values()):
        print("   • System is ready for production use!")
        print("   • Consider performance optimization for large datasets")
        print("   • Test with various music styles and tempos")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)