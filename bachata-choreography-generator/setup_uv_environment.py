#!/usr/bin/env python3
"""
UV-specific setup script for the optimized Bachata choreography generation pipeline.
Designed to work with UV package manager.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_system_requirements():
    """Check system requirements."""
    logger.info("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required. Current version: %s", sys.version)
        return False
    logger.info("✅ Python version: %s", sys.version.split()[0])
    
    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            logger.info("✅ FFmpeg available: %s", version_line)
        else:
            logger.error("❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ FFmpeg not found. Please install FFmpeg:")
        logger.error("   macOS: brew install ffmpeg")
        logger.error("   Ubuntu/Debian: sudo apt-get install ffmpeg")
        return False
    
    # Check UV
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ UV available: %s", result.stdout.strip())
        else:
            logger.error("❌ UV not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ UV not found. Please install UV first:")
        logger.error("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    
    # Check Docker (optional)
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ Docker available: %s", result.stdout.strip())
        else:
            logger.warning("⚠️  Docker not working properly (optional for Qdrant)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("⚠️  Docker not found (optional for Qdrant vector database)")
    
    return True


def create_directories():
    """Create required directories."""
    logger.info("📁 Creating required directories...")
    
    required_dirs = [
        "data/songs",
        "data/moves", 
        "data/output",
        "data/cache",
        "data/temp",
        "data/choreography_metadata",
        "data/test_results",
        "logs"
    ]
    
    try:
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info("✅ Created: %s", dir_path)
        return True
    except Exception as e:
        logger.error("❌ Failed to create directories: %s", e)
        return False


def install_packages_with_uv():
    """Install packages using UV."""
    logger.info("📦 Installing packages with UV...")
    
    # Core packages for the pipeline
    packages = [
        "librosa>=0.9.0",
        "numpy>=1.21.0", 
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "tqdm>=4.62.0",
        "pydantic>=1.8.0",
        "yt-dlp>=2023.1.0",
        "psutil>=5.8.0"
    ]
    
    # Install all packages at once
    try:
        logger.info("Installing core packages...")
        cmd = ["uv", "add"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✅ Core packages installed successfully")
        else:
            logger.error("❌ Failed to install core packages: %s", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout installing packages")
        return False
    except Exception as e:
        logger.error("❌ Error installing packages: %s", e)
        return False
    
    # Install optional packages
    optional_packages = ["qdrant-client>=1.0.0"]
    
    for package in optional_packages:
        try:
            logger.info("Installing optional package: %s", package)
            result = subprocess.run(["uv", "add", package], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Installed optional: %s", package)
            else:
                logger.warning("⚠️  Failed to install optional %s: %s", package, result.stderr)
        except Exception as e:
            logger.warning("⚠️  Error installing optional %s: %s", package, e)
    
    return True


def validate_installation():
    """Validate the installation by testing imports."""
    logger.info("🧪 Validating installation...")
    
    test_imports = [
        ("librosa", "Audio processing"),
        ("cv2", "Computer vision"),
        ("mediapipe", "Pose detection"),
        ("numpy", "Numerical computing"),
        ("tqdm", "Progress bars"),
        ("pydantic", "Data validation"),
        ("yt_dlp", "YouTube downloading")
    ]
    
    # Test required imports
    for module, description in test_imports:
        try:
            __import__(module)
            logger.info("✅ %s (%s)", module, description)
        except ImportError as e:
            logger.error("❌ Failed to import %s (%s): %s", module, description, e)
            return False
    
    # Test optional imports
    try:
        import qdrant_client
        logger.info("✅ qdrant_client (Vector database) - optional")
    except ImportError:
        logger.warning("⚠️  qdrant_client not available - optional")
    
    # Test pipeline imports
    try:
        sys.path.append(str(Path.cwd()))
        from app.services.choreography_pipeline import ChoreoGenerationPipeline
        from app.services.optimized_recommendation_engine import OptimizedRecommendationEngine
        from app.services.data_persistence import DataPersistenceManager
        logger.info("✅ Pipeline services imported successfully")
    except ImportError as e:
        logger.error("❌ Failed to import pipeline services: %s", e)
        return False
    
    return True


def check_data_availability():
    """Check availability of test data."""
    logger.info("📊 Checking test data availability...")
    
    data_status = {
        "songs": 0,
        "moves": 0,
        "annotations": False
    }
    
    # Check songs
    songs_dir = Path("data/songs")
    if songs_dir.exists():
        songs = list(songs_dir.glob("*.mp3"))
        data_status["songs"] = len(songs)
        logger.info("🎵 Found %d song files", len(songs))
    else:
        logger.warning("⚠️  No songs directory found")
    
    # Check moves
    moves_dir = Path("data/moves")
    if moves_dir.exists():
        moves = list(moves_dir.glob("*.mp4"))
        data_status["moves"] = len(moves)
        logger.info("💃 Found %d move video files", len(moves))
    else:
        logger.warning("⚠️  No moves directory found")
    
    # Check annotations
    annotations_file = Path("data/bachata_annotations.json")
    if annotations_file.exists():
        data_status["annotations"] = True
        logger.info("✅ Annotations file found")
    else:
        logger.warning("⚠️  No annotations file found")
    
    return data_status


def create_test_script():
    """Create a simple test script."""
    logger.info("📝 Creating test script...")
    
    test_script = '''#!/usr/bin/env python3
"""
Simple test script for the optimized pipeline.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all imports work."""
    print("🧪 Testing imports...")
    
    try:
        import librosa
        print("✅ librosa")
    except ImportError as e:
        print(f"❌ librosa: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python")
    except ImportError as e:
        print(f"❌ opencv-python: {e}")
        return False
    
    try:
        import mediapipe
        print("✅ mediapipe")
    except ImportError as e:
        print(f"❌ mediapipe: {e}")
        return False
    
    try:
        from app.services.choreography_pipeline import ChoreoGenerationPipeline
        print("✅ Pipeline services")
    except ImportError as e:
        print(f"❌ Pipeline services: {e}")
        return False
    
    return True

def check_data():
    """Check for test data."""
    print("📊 Checking test data...")
    
    songs_dir = Path("data/songs")
    songs = list(songs_dir.glob("*.mp3")) if songs_dir.exists() else []
    print(f"🎵 Songs: {len(songs)} files")
    
    moves_dir = Path("data/moves")
    moves = list(moves_dir.glob("*.mp4")) if moves_dir.exists() else []
    print(f"💃 Moves: {len(moves)} files")
    
    annotations = Path("data/bachata_annotations.json").exists()
    print(f"📋 Annotations: {'✅' if annotations else '❌'}")
    
    return len(songs) > 0 and len(moves) > 0 and annotations

def main():
    print("🎵 Optimized Pipeline Test")
    print("=" * 30)
    
    if not test_imports():
        print("❌ Import test failed")
        return 1
    
    if not check_data():
        print("⚠️  Test data not complete")
        print("   Add .mp3 files to data/songs/")
        print("   Add .mp4 files to data/moves/")
        print("   Ensure bachata_annotations.json exists")
    else:
        print("✅ Ready to test pipeline!")
        print("   Run: uv run python optimized_choreography_generator.py --list-songs")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    try:
        test_file = Path("test_pipeline_simple.py")
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        os.chmod(test_file, 0o755)
        logger.info("✅ Created test script: test_pipeline_simple.py")
        return True
    except Exception as e:
        logger.error("❌ Failed to create test script: %s", e)
        return False


def print_next_steps(data_status):
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 UV SETUP COMPLETED!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    
    # Data preparation
    if data_status["songs"] == 0:
        print("\n1. 📁 ADD TEST DATA:")
        print("   • Copy .mp3 files to data/songs/")
        print("   • Copy dance move .mp4 files to data/moves/")
        print("   • Ensure bachata_annotations.json exists in data/")
    else:
        print(f"\n1. ✅ TEST DATA READY:")
        print(f"   • Songs: {data_status['songs']} files")
        print(f"   • Moves: {data_status['moves']} files")
        print(f"   • Annotations: {'✅' if data_status['annotations'] else '❌'}")
    
    # Testing
    print("\n2. 🧪 RUN TESTS:")
    print("   uv run python test_pipeline_simple.py")
    print("   uv run python optimized_choreography_generator.py --list-songs")
    
    # Basic usage
    print("\n3. 🚀 GENERATE CHOREOGRAPHY:")
    print("   # Fast test (if you have songs)")
    print("   uv run python optimized_choreography_generator.py data/songs/your_song.mp3 --quality fast --duration 30s")
    print("   ")
    print("   # Comprehensive test")
    print("   uv run python optimized_choreography_generator.py --test-all")
    
    # Optional Qdrant
    print("\n4. 🔧 OPTIONAL - SETUP QDRANT:")
    print("   uv run python setup_qdrant.py start")
    print("   uv run python setup_qdrant.py migrate")
    
    print("\n" + "="*60)
    print("🎵 Ready for Bachata choreography generation! 💃")
    print("="*60)


def main():
    """Main setup function."""
    print("🎵 UV-Based Bachata Pipeline Setup")
    print("="*40)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met.")
        return 1
    
    # Create directories
    if not create_directories():
        print("\n❌ Failed to create directories.")
        return 1
    
    # Install packages
    if not install_packages_with_uv():
        print("\n❌ Failed to install packages.")
        return 1
    
    # Validate installation
    if not validate_installation():
        print("\n❌ Installation validation failed.")
        return 1
    
    # Check data
    data_status = check_data_availability()
    
    # Create test script
    create_test_script()
    
    # Print next steps
    print_next_steps(data_status)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())