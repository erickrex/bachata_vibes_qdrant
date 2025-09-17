#!/usr/bin/env python3
"""
Comprehensive test script for the complete Bachata Choreography Generation Pipeline.
Tests the entire pipeline including Qdrant integration, Superlinked embeddings, and video generation.

Usage:
    python test_complete_pipeline.py --song Aventura --difficulty intermediate --energy high
    python test_complete_pipeline.py --all-songs --quick-test
    python test_complete_pipeline.py --interactive
"""

import asyncio
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add the app directory to the path
sys.path.append('.')
sys.path.append('app')

from app.services.choreography_pipeline import ChoreoGenerationPipeline, PipelineConfig, PipelineResult
from app.services.qdrant_service import QdrantConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_test.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineTester:
    """Comprehensive pipeline testing class."""
    
    def __init__(self):
        self.available_songs = self._get_available_songs()
        self.test_results = []
        self.total_tests = 0
        self.successful_tests = 0
        
    def _get_available_songs(self) -> List[str]:
        """Get list of available songs."""
        songs_dir = Path("data/songs")
        if not songs_dir.exists():
            logger.error("Songs directory not found!")
            return []
        
        songs = []
        for song_file in songs_dir.glob("*.mp3"):
            songs.append(song_file.stem)
        
        logger.info(f"Found {len(songs)} available songs: {', '.join(songs)}")
        return songs
    
    async def test_pipeline_configuration(self, config: PipelineConfig) -> Dict[str, Any]:
        """Test a specific pipeline configuration."""
        logger.info(f"Testing pipeline with {config.quality_mode} quality mode")
        
        try:
            pipeline = ChoreoGenerationPipeline(config)
            
            # Test pipeline initialization
            health_status = pipeline.get_qdrant_health_status()
            logger.info(f"Qdrant health status: {health_status}")
            
            # Test service initialization (lazy loading)
            services_status = {
                "music_analyzer": pipeline.music_analyzer is not None,
                "move_analyzer": pipeline.move_analyzer is not None,
                "recommendation_engine": pipeline.recommendation_engine is not None,
                "video_generator": pipeline.video_generator is not None,
                "annotation_interface": pipeline.annotation_interface is not None,
                "feature_fusion": pipeline.feature_fusion is not None,
                "youtube_service": pipeline.youtube_service is not None,
                "qdrant_service": pipeline.qdrant_service is not None
            }
            
            logger.info(f"Services initialized: {services_status}")
            
            return {
                "success": True,
                "health_status": health_status,
                "services_status": services_status,
                "config": {
                    "quality_mode": config.quality_mode,
                    "enable_qdrant": config.enable_qdrant,
                    "enable_caching": config.enable_caching,
                    "max_workers": config.max_workers
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline configuration test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_choreography_generation(
        self,
        song_name: str,
        difficulty: str = "intermediate",
        energy_level: Optional[str] = None,
        role_focus: Optional[str] = None,
        move_types: Optional[List[str]] = None,
        tempo_range: Optional[List[int]] = None,
        quality_mode: str = "balanced"
    ) -> Dict[str, Any]:
        """Test complete choreography generation for a specific song."""
        
        self.total_tests += 1
        test_start_time = time.time()
        
        logger.info("=" * 80)
        logger.info(f"🎵 TESTING CHOREOGRAPHY GENERATION")
        logger.info(f"Song: {song_name}")
        logger.info(f"Difficulty: {difficulty}")
        logger.info(f"Energy Level: {energy_level}")
        logger.info(f"Role Focus: {role_focus}")
        logger.info(f"Move Types: {move_types}")
        logger.info(f"Tempo Range: {tempo_range}")
        logger.info(f"Quality Mode: {quality_mode}")
        logger.info("=" * 80)
        
        try:
            # Validate song exists
            song_path = Path(f"data/songs/{song_name}.mp3")
            if not song_path.exists():
                raise FileNotFoundError(f"Song file not found: {song_path}")
            
            # Create pipeline configuration
            config = PipelineConfig(
                quality_mode=quality_mode,
                enable_caching=True,
                enable_qdrant=True,
                auto_populate_qdrant=True,
                max_workers=4,
                cleanup_after_generation=True
            )
            
            # Initialize pipeline
            pipeline = ChoreoGenerationPipeline(config)
            
            # Generate choreography
            logger.info("🚀 Starting choreography generation...")
            result = await pipeline.generate_choreography(
                audio_input=str(song_path),
                difficulty=difficulty,
                energy_level=energy_level,
                role_focus=role_focus,
                move_types=move_types,
                tempo_range=tempo_range
            )
            
            test_duration = time.time() - test_start_time
            
            if result.success:
                self.successful_tests += 1
                logger.info("✅ CHOREOGRAPHY GENERATION SUCCESSFUL!")
                logger.info(f"📊 Processing time: {result.processing_time:.2f}s")
                logger.info(f"🎬 Output video: {result.output_path}")
                logger.info(f"📝 Metadata: {result.metadata_path}")
                logger.info(f"💃 Moves analyzed: {result.moves_analyzed}")
                logger.info(f"🎯 Recommendations generated: {result.recommendations_generated}")
                logger.info(f"⏱️ Sequence duration: {result.sequence_duration:.1f}s")
                
                # Qdrant statistics
                if result.qdrant_enabled:
                    logger.info(f"🔍 Qdrant embeddings stored: {result.qdrant_embeddings_stored}")
                    logger.info(f"📥 Qdrant embeddings retrieved: {result.qdrant_embeddings_retrieved}")
                    logger.info(f"⚡ Qdrant search time: {result.qdrant_search_time:.3f}s")
                
                # Cache statistics
                logger.info(f"💾 Cache hits: {result.cache_hits}")
                logger.info(f"🔄 Cache misses: {result.cache_misses}")
                
                # Verify output file exists
                if result.output_path and Path(result.output_path).exists():
                    file_size = Path(result.output_path).stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"📁 Output file size: {file_size:.1f} MB")
                else:
                    logger.warning("⚠️ Output video file not found!")
                
                test_result = {
                    "success": True,
                    "song": song_name,
                    "parameters": {
                        "difficulty": difficulty,
                        "energy_level": energy_level,
                        "role_focus": role_focus,
                        "move_types": move_types,
                        "tempo_range": tempo_range,
                        "quality_mode": quality_mode
                    },
                    "results": {
                        "processing_time": result.processing_time,
                        "output_path": result.output_path,
                        "metadata_path": result.metadata_path,
                        "moves_analyzed": result.moves_analyzed,
                        "recommendations_generated": result.recommendations_generated,
                        "sequence_duration": result.sequence_duration,
                        "qdrant_enabled": result.qdrant_enabled,
                        "qdrant_embeddings_stored": result.qdrant_embeddings_stored,
                        "qdrant_embeddings_retrieved": result.qdrant_embeddings_retrieved,
                        "qdrant_search_time": result.qdrant_search_time,
                        "cache_hits": result.cache_hits,
                        "cache_misses": result.cache_misses
                    },
                    "test_duration": test_duration,
                    "file_size_mb": file_size if result.output_path and Path(result.output_path).exists() else 0
                }
                
            else:
                logger.error("❌ CHOREOGRAPHY GENERATION FAILED!")
                logger.error(f"Error: {result.error_message}")
                
                test_result = {
                    "success": False,
                    "song": song_name,
                    "parameters": {
                        "difficulty": difficulty,
                        "energy_level": energy_level,
                        "role_focus": role_focus,
                        "move_types": move_types,
                        "tempo_range": tempo_range,
                        "quality_mode": quality_mode
                    },
                    "error": result.error_message,
                    "test_duration": test_duration,
                    "partial_results": {
                        "processing_time": result.processing_time,
                        "moves_analyzed": result.moves_analyzed,
                        "qdrant_enabled": result.qdrant_enabled,
                        "cache_hits": result.cache_hits,
                        "cache_misses": result.cache_misses
                    }
                }
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            logger.error(f"❌ TEST FAILED WITH EXCEPTION: {e}")
            test_duration = time.time() - test_start_time
            
            test_result = {
                "success": False,
                "song": song_name,
                "parameters": {
                    "difficulty": difficulty,
                    "energy_level": energy_level,
                    "role_focus": role_focus,
                    "move_types": move_types,
                    "tempo_range": tempo_range,
                    "quality_mode": quality_mode
                },
                "error": str(e),
                "test_duration": test_duration
            }
            
            self.test_results.append(test_result)
            return test_result
    
    async def test_all_quality_modes(self, song_name: str) -> List[Dict[str, Any]]:
        """Test all quality modes for a specific song."""
        logger.info(f"🎯 Testing all quality modes for {song_name}")
        
        quality_modes = ["fast", "balanced", "high_quality"]
        results = []
        
        for quality_mode in quality_modes:
            logger.info(f"Testing {quality_mode} quality mode...")
            result = await self.test_choreography_generation(
                song_name=song_name,
                difficulty="intermediate",
                quality_mode=quality_mode
            )
            results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        return results
    
    async def test_different_parameters(self, song_name: str) -> List[Dict[str, Any]]:
        """Test different parameter combinations for a song."""
        logger.info(f"🎯 Testing different parameters for {song_name}")
        
        test_cases = [
            {
                "name": "Beginner High Energy",
                "difficulty": "beginner",
                "energy_level": "high",
                "role_focus": "both"
            },
            {
                "name": "Advanced Low Energy",
                "difficulty": "advanced",
                "energy_level": "low",
                "role_focus": "follow_focus"
            },
            {
                "name": "Intermediate Lead Focus",
                "difficulty": "intermediate",
                "energy_level": "medium",
                "role_focus": "lead_focus"
            },
            {
                "name": "Basic Moves Only",
                "difficulty": "beginner",
                "move_types": ["basic_step", "cross_body_lead"]
            },
            {
                "name": "Fast Tempo Range",
                "difficulty": "advanced",
                "tempo_range": [135, 150]
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            logger.info(f"Testing: {test_case['name']}")
            
            # Extract parameters
            params = test_case.copy()
            del params['name']
            
            result = await self.test_choreography_generation(
                song_name=song_name,
                **params
            )
            
            result['test_case_name'] = test_case['name']
            results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(2)
        
        return results
    
    async def test_qdrant_integration(self) -> Dict[str, Any]:
        """Test Qdrant integration specifically."""
        logger.info("🔍 Testing Qdrant integration...")
        
        try:
            config = PipelineConfig(
                enable_qdrant=True,
                auto_populate_qdrant=True
            )
            
            pipeline = ChoreoGenerationPipeline(config)
            
            # Test Qdrant health
            health_status = pipeline.get_qdrant_health_status()
            
            # Test recommendation engine Qdrant integration
            rec_engine = pipeline.recommendation_engine
            qdrant_available = rec_engine.is_qdrant_available()
            
            # Get performance stats
            perf_stats = pipeline.get_performance_comparison()
            
            return {
                "success": True,
                "health_status": health_status,
                "qdrant_available": qdrant_available,
                "performance_stats": perf_stats
            }
            
        except Exception as e:
            logger.error(f"Qdrant integration test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_quick_test(self) -> Dict[str, Any]:
        """Run a quick test with one song to verify everything works."""
        logger.info("🚀 Running quick pipeline test...")
        
        if not self.available_songs:
            return {"success": False, "error": "No songs available"}
        
        # Use the first available song
        test_song = self.available_songs[0]
        
        # Test basic configuration
        config_test = await self.test_pipeline_configuration(PipelineConfig())
        
        # Test Qdrant integration
        qdrant_test = await self.test_qdrant_integration()
        
        # Test choreography generation
        choreo_test = await self.test_choreography_generation(
            song_name=test_song,
            difficulty="intermediate",
            energy_level="medium"
        )
        
        return {
            "success": choreo_test["success"],
            "config_test": config_test,
            "qdrant_test": qdrant_test,
            "choreography_test": choreo_test
        }
    
    async def run_comprehensive_test(self, songs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive tests on multiple songs."""
        logger.info("🎯 Running comprehensive pipeline tests...")
        
        test_songs = songs or self.available_songs[:3]  # Test first 3 songs by default
        
        all_results = []
        
        for song in test_songs:
            logger.info(f"Testing song: {song}")
            
            # Test different quality modes
            quality_results = await self.test_all_quality_modes(song)
            all_results.extend(quality_results)
            
            # Test different parameters
            param_results = await self.test_different_parameters(song)
            all_results.extend(param_results)
        
        return {
            "success": self.successful_tests > 0,
            "total_tests": self.total_tests,
            "successful_tests": self.successful_tests,
            "success_rate": (self.successful_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            "all_results": all_results
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("🎵 BACHATA CHOREOGRAPHY PIPELINE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Total Tests: {self.total_tests}")
        report.append(f"Successful Tests: {self.successful_tests}")
        report.append(f"Failed Tests: {self.total_tests - self.successful_tests}")
        report.append(f"Success Rate: {(self.successful_tests / self.total_tests * 100):.1f}%" if self.total_tests > 0 else "N/A")
        report.append("")
        
        # Group results by success/failure
        successful_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        if successful_tests:
            report.append("✅ SUCCESSFUL TESTS:")
            report.append("-" * 40)
            for test in successful_tests:
                song = test["song"]
                params = test["parameters"]
                results = test.get("results", {})
                
                report.append(f"🎵 {song} ({params['difficulty']}, {params['quality_mode']})")
                report.append(f"   ⏱️  Processing: {results.get('processing_time', 0):.2f}s")
                report.append(f"   💃 Moves: {results.get('moves_analyzed', 0)}")
                report.append(f"   🎯 Recommendations: {results.get('recommendations_generated', 0)}")
                report.append(f"   📁 File size: {test.get('file_size_mb', 0):.1f} MB")
                if results.get('qdrant_enabled'):
                    report.append(f"   🔍 Qdrant: {results.get('qdrant_embeddings_stored', 0)} stored, {results.get('qdrant_embeddings_retrieved', 0)} retrieved")
                report.append("")
        
        if failed_tests:
            report.append("❌ FAILED TESTS:")
            report.append("-" * 40)
            for test in failed_tests:
                song = test["song"]
                params = test["parameters"]
                error = test.get("error", "Unknown error")
                
                report.append(f"🎵 {song} ({params['difficulty']}, {params['quality_mode']})")
                report.append(f"   ❌ Error: {error}")
                report.append("")
        
        # Performance summary
        if successful_tests:
            avg_processing_time = sum(t.get("results", {}).get("processing_time", 0) for t in successful_tests) / len(successful_tests)
            avg_moves = sum(t.get("results", {}).get("moves_analyzed", 0) for t in successful_tests) / len(successful_tests)
            total_file_size = sum(t.get("file_size_mb", 0) for t in successful_tests)
            
            report.append("📊 PERFORMANCE SUMMARY:")
            report.append("-" * 40)
            report.append(f"Average processing time: {avg_processing_time:.2f}s")
            report.append(f"Average moves analyzed: {avg_moves:.1f}")
            report.append(f"Total output size: {total_file_size:.1f} MB")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_detailed_results(self, filename: str = "pipeline_test_results.json"):
        """Save detailed test results to JSON file."""
        results_data = {
            "test_summary": {
                "total_tests": self.total_tests,
                "successful_tests": self.successful_tests,
                "failed_tests": self.total_tests - self.successful_tests,
                "success_rate": (self.successful_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "test_results": self.test_results,
            "available_songs": self.available_songs
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {filename}")

async def interactive_mode():
    """Interactive mode for testing specific configurations."""
    tester = PipelineTester()
    
    print("🎵 Bachata Choreography Pipeline Interactive Tester")
    print("=" * 60)
    
    if not tester.available_songs:
        print("❌ No songs found in data/songs directory!")
        return
    
    print(f"Available songs: {', '.join(tester.available_songs)}")
    print()
    
    while True:
        try:
            # Get song selection
            song = input(f"Enter song name (or 'quit' to exit): ").strip()
            if song.lower() == 'quit':
                break
            
            if song not in tester.available_songs:
                print(f"❌ Song '{song}' not found. Available: {', '.join(tester.available_songs)}")
                continue
            
            # Get parameters
            difficulty = input("Difficulty (beginner/intermediate/advanced) [intermediate]: ").strip() or "intermediate"
            energy_level = input("Energy level (low/medium/high) [auto]: ").strip() or None
            role_focus = input("Role focus (lead_focus/follow_focus/both) [both]: ").strip() or None
            quality_mode = input("Quality mode (fast/balanced/high_quality) [balanced]: ").strip() or "balanced"
            
            print(f"\n🚀 Generating choreography for {song}...")
            
            # Run test
            result = await tester.test_choreography_generation(
                song_name=song,
                difficulty=difficulty,
                energy_level=energy_level,
                role_focus=role_focus,
                quality_mode=quality_mode
            )
            
            if result["success"]:
                print("✅ Generation successful!")
                print(f"📁 Output: {result['results']['output_path']}")
            else:
                print(f"❌ Generation failed: {result['error']}")
            
            print("\n" + "-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Test Bachata Choreography Pipeline")
    parser.add_argument("--song", type=str, help="Specific song to test")
    parser.add_argument("--difficulty", type=str, default="intermediate", 
                       choices=["beginner", "intermediate", "advanced"],
                       help="Difficulty level")
    parser.add_argument("--energy", type=str, choices=["low", "medium", "high"],
                       help="Energy level")
    parser.add_argument("--role-focus", type=str, choices=["lead_focus", "follow_focus", "both"],
                       help="Role focus")
    parser.add_argument("--quality", type=str, default="balanced",
                       choices=["fast", "balanced", "high_quality"],
                       help="Quality mode")
    parser.add_argument("--all-songs", action="store_true",
                       help="Test all available songs")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test with one song")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive tests")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--save-results", type=str, default="pipeline_test_results.json",
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    tester = PipelineTester()
    
    try:
        if args.interactive:
            await interactive_mode()
            return
        
        if args.quick_test:
            logger.info("Running quick test...")
            result = await tester.run_quick_test()
            print(tester.generate_report())
            
        elif args.comprehensive:
            logger.info("Running comprehensive tests...")
            result = await tester.run_comprehensive_test()
            print(tester.generate_report())
            
        elif args.all_songs:
            logger.info("Testing all songs...")
            for song in tester.available_songs:
                await tester.test_choreography_generation(
                    song_name=song,
                    difficulty=args.difficulty,
                    energy_level=args.energy,
                    role_focus=args.role_focus,
                    quality_mode=args.quality
                )
            print(tester.generate_report())
            
        elif args.song:
            logger.info(f"Testing specific song: {args.song}")
            result = await tester.test_choreography_generation(
                song_name=args.song,
                difficulty=args.difficulty,
                energy_level=args.energy,
                role_focus=args.role_focus,
                quality_mode=args.quality
            )
            print(tester.generate_report())
            
        else:
            # Default: run quick test
            logger.info("No specific test specified, running quick test...")
            result = await tester.run_quick_test()
            print(tester.generate_report())
        
        # Save results
        if tester.test_results:
            tester.save_detailed_results(args.save_results)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())