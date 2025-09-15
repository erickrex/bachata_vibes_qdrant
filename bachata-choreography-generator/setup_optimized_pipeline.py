#!/usr/bin/env python3
"""
Setup script for the optimized Bachata choreography generation pipeline.
Helps users prepare their environment and validate the installation.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineSetup:
    """Setup manager for the optimized choreography pipeline."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_dirs = [
            "data/songs",
            "data/moves", 
            "data/output",
            "data/cache",
            "data/temp",
            "data/choreography_metadata",
            "data/test_results",
            "logs"
        ]
        
        self.required_packages = [
            "librosa>=0.9.0",
            "numpy>=1.21.0",
            "opencv-python>=4.5.0",
            "mediapipe>=0.8.0",
            "tqdm>=4.62.0",
            "pydantic>=1.8.0",
            "yt-dlp>=2023.1.0",
            "psutil>=5.8.0"
        ]
        
        self.optional_packages = [
            "qdrant-client>=1.0.0"
        ]
    
    def check_system_requirements(self) -> bool:
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
            logger.error("   Ubuntu/Debian: sudo apt-get install ffmpeg")
            logger.error("   macOS: brew install ffmpeg")
            logger.error("   Windows: Download from https://ffmpeg.org/download.html")
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
            logger.info("   Install Docker from: https://docs.docker.com/get-docker/")
        
        return True
    
    def create_directories(self) -> bool:
        """Create required directories."""
        logger.info("📁 Creating required directories...")
        
        try:
            for dir_path in self.required_dirs:
                full_path = self.project_root / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info("✅ Created: %s", dir_path)
            return True
        except Exception as e:
            logger.error("❌ Failed to create directories: %s", e)
            return False
    
    def install_packages(self, include_optional: bool = True) -> bool:
        """Install required Python packages."""
        logger.info("📦 Installing Python packages...")
        
        # Detect if we're in a uv environment
        uv_available = self._check_uv_available()
        
        # Install required packages
        for package in self.required_packages:
            try:
                logger.info("Installing %s...", package)
                
                if uv_available:
                    # Use uv add command
                    result = subprocess.run([
                        "uv", "add", package
                    ], capture_output=True, text=True, timeout=300)
                else:
                    # Fallback to pip
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info("✅ Installed: %s", package)
                else:
                    logger.error("❌ Failed to install %s: %s", package, result.stderr)
                    return False
            except subprocess.TimeoutExpired:
                logger.error("❌ Timeout installing %s", package)
                return False
            except Exception as e:
                logger.error("❌ Error installing %s: %s", package, e)
                return False
        
        # Install optional packages
        if include_optional:
            for package in self.optional_packages:
                try:
                    logger.info("Installing optional package %s...", package)
                    
                    if uv_available:
                        result = subprocess.run([
                            "uv", "add", package
                        ], capture_output=True, text=True, timeout=300)
                    else:
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        logger.info("✅ Installed optional: %s", package)
                    else:
                        logger.warning("⚠️  Failed to install optional %s: %s", package, result.stderr)
                except Exception as e:
                    logger.warning("⚠️  Error installing optional %s: %s", package, e)
        
        return True
    
    def _check_uv_available(self) -> bool:
        """Check if uv is available and we're in a uv project."""
        try:
            # Check if uv command is available
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False
            
            # Check if we're in a uv project (pyproject.toml exists)
            pyproject_path = self.project_root / "pyproject.toml"
            if pyproject_path.exists():
                logger.info("✅ UV environment detected")
                return True
            
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def validate_installation(self) -> bool:
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
        
        optional_imports = [
            ("qdrant_client", "Vector database")
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
        for module, description in optional_imports:
            try:
                __import__(module)
                logger.info("✅ %s (%s) - optional", module, description)
            except ImportError:
                logger.warning("⚠️  %s (%s) not available - optional", module, description)
        
        # Test pipeline imports
        try:
            sys.path.append(str(self.project_root))
            from app.services.choreography_pipeline import ChoreoGenerationPipeline
            from app.services.optimized_recommendation_engine import OptimizedRecommendationEngine
            from app.services.data_persistence import DataPersistenceManager
            logger.info("✅ Pipeline services imported successfully")
        except ImportError as e:
            logger.error("❌ Failed to import pipeline services: %s", e)
            return False
        
        return True
    
    def check_data_availability(self) -> dict:
        """Check availability of test data."""
        logger.info("📊 Checking test data availability...")
        
        data_status = {
            "songs": 0,
            "moves": 0,
            "annotations": False
        }
        
        # Check songs
        songs_dir = self.project_root / "data/songs"
        if songs_dir.exists():
            songs = list(songs_dir.glob("*.mp3"))
            data_status["songs"] = len(songs)
            logger.info("🎵 Found %d song files", len(songs))
        else:
            logger.warning("⚠️  No songs directory found")
        
        # Check moves
        moves_dir = self.project_root / "data/moves"
        if moves_dir.exists():
            moves = list(moves_dir.glob("*.mp4"))
            data_status["moves"] = len(moves)
            logger.info("💃 Found %d move video files", len(moves))
        else:
            logger.warning("⚠️  No moves directory found")
        
        # Check annotations
        annotations_file = self.project_root / "data/bachata_annotations.json"
        if annotations_file.exists():
            data_status["annotations"] = True
            logger.info("✅ Annotations file found")
        else:
            logger.warning("⚠️  No annotations file found")
        
        return data_status
    
    def create_sample_test_script(self) -> bool:
        """Create a sample test script for users."""
        logger.info("📝 Creating sample test script...")
        
        test_script = '''#!/usr/bin/env python3
"""
Sample test script for the optimized Bachata choreography pipeline.
Run this script to test basic functionality.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

async def test_basic_functionality():
    """Test basic pipeline functionality."""
    try:
        from app.services.choreography_pipeline import ChoreoGenerationPipeline, PipelineConfig
        
        print("🚀 Testing optimized choreography pipeline...")
        
        # Create pipeline with fast configuration for testing
        config = PipelineConfig(
            quality_mode="fast",
            target_fps=10,
            max_workers=2,
            enable_caching=True
        )
        
        pipeline = ChoreoGenerationPipeline(config)
        print("✅ Pipeline initialized successfully")
        
        # Check for test data
        songs_dir = Path("data/songs")
        if not songs_dir.exists() or not list(songs_dir.glob("*.mp3")):
            print("⚠️  No test songs found in data/songs/")
            print("   Please add at least one .mp3 file to test the pipeline")
            return
        
        test_song = list(songs_dir.glob("*.mp3"))[0]
        print(f"🎵 Using test song: {test_song.name}")
        
        # Test pipeline (this would actually generate choreography)
        print("🧪 Pipeline test would run here...")
        print("   To run actual test: python optimized_choreography_generator.py", str(test_song))
        
        print("✅ Basic functionality test completed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please run setup_optimized_pipeline.py first")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
'''
        
        try:
            test_file = self.project_root / "test_pipeline.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            # Make executable
            os.chmod(test_file, 0o755)
            
            logger.info("✅ Created test script: test_pipeline.py")
            return True
        except Exception as e:
            logger.error("❌ Failed to create test script: %s", e)
            return False
    
    def print_next_steps(self, data_status: dict) -> None:
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
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
        
        # Basic testing
        print("\n2. 🧪 RUN BASIC TESTS:")
        print("   python test_pipeline.py")
        print("   python optimized_choreography_generator.py --list-songs")
        
        # Full testing
        print("\n3. 🚀 RUN FULL PIPELINE:")
        print("   # Fast mode test")
        print("   python optimized_choreography_generator.py data/songs/your_song.mp3 --quality fast")
        print("   ")
        print("   # Comprehensive test suite")
        print("   python optimized_choreography_generator.py --test-all")
        
        # Optional Qdrant setup
        print("\n4. 🔧 OPTIONAL - SETUP QDRANT (for better performance):")
        print("   python setup_qdrant.py start")
        print("   python setup_qdrant.py migrate")
        print("   python setup_qdrant.py test")
        
        # Documentation
        print("\n5. 📚 READ DOCUMENTATION:")
        print("   • OPTIMIZED_PIPELINE_DOCUMENTATION.md - Complete feature guide")
        print("   • TESTING_INSTRUCTIONS.md - Comprehensive testing procedures")
        
        print("\n" + "="*60)
        print("🎵 Ready to generate amazing Bachata choreographies! 💃")
        print("="*60)


def main():
    """Main setup function."""
    print("🎵 Bachata Choreography Pipeline Setup")
    print("="*50)
    
    setup = PipelineSetup()
    
    # Check system requirements
    if not setup.check_system_requirements():
        print("\n❌ System requirements not met. Please install missing dependencies.")
        return 1
    
    # Create directories
    if not setup.create_directories():
        print("\n❌ Failed to create required directories.")
        return 1
    
    # Install packages
    print("\n📦 Installing Python packages...")
    print("This may take several minutes...")
    if not setup.install_packages():
        print("\n❌ Failed to install required packages.")
        return 1
    
    # Validate installation
    if not setup.validate_installation():
        print("\n❌ Installation validation failed.")
        return 1
    
    # Check data availability
    data_status = setup.check_data_availability()
    
    # Create test script
    setup.create_sample_test_script()
    
    # Print next steps
    setup.print_next_steps(data_status)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())