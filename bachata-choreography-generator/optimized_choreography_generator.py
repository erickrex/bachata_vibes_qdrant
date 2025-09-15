#!/usr/bin/env python3
"""
Optimized Bachata Choreography Generator with comprehensive pipeline testing.
Enhanced version with batch processing, quality modes, and detailed performance metrics.
"""

import sys
import time
import asyncio
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

# Add app to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import optimized services
from app.services.choreography_pipeline import ChoreoGenerationPipeline, PipelineConfig, PipelineResult
from app.services.optimized_recommendation_engine import OptimizedRecommendationEngine, OptimizationConfig


@dataclass
class TestConfig:
    """Configuration for comprehensive testing."""
    # Quality modes
    quality_mode: str = "balanced"  # fast, balanced, high_quality
    
    # Test modes
    enable_batch_processing: bool = False
    enable_performance_benchmarking: bool = True
    enable_error_recovery_testing: bool = True
    
    # Output settings
    output_dir: str = "data/output"
    test_results_dir: str = "data/test_results"
    
    # Validation settings
    validate_all_stages: bool = True
    generate_detailed_reports: bool = True


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_name: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    
    # Pipeline metrics
    pipeline_result: Optional[PipelineResult] = None
    
    # Quality metrics
    output_file_size_mb: float = 0.0
    video_duration: float = 0.0
    moves_count: int = 0
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class BatchTestResult:
    """Result of batch testing."""
    total_tests: int
    successful_tests: int
    failed_tests: int
    total_execution_time: float
    average_execution_time: float
    
    individual_results: List[TestResult]
    performance_summary: Dict[str, Any]


class OptimizedChoreoGenerator:
    """
    Optimized choreography generator with comprehensive testing capabilities.
    Features:
    - Multiple quality modes (fast/balanced/high_quality)
    - Batch processing for multiple songs
    - Comprehensive error handling and recovery
    - Detailed performance metrics and benchmarking
    - Pipeline stage validation with success/failure reporting
    """
    
    def __init__(self, test_config: Optional[TestConfig] = None):
        """Initialize the optimized choreography generator."""
        self.test_config = test_config or TestConfig()
        
        # Initialize pipeline with optimized configuration
        pipeline_config = self._create_pipeline_config()
        self.pipeline = ChoreoGenerationPipeline(pipeline_config)
        
        # Initialize optimized recommendation engine
        optimization_config = self._create_optimization_config()
        self.optimized_engine = OptimizedRecommendationEngine(optimization_config)
        
        # Test results storage
        self.test_results: List[TestResult] = []
        
        # Ensure directories exist
        Path(self.test_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.test_config.test_results_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"OptimizedChoreoGenerator initialized with {self.test_config.quality_mode} quality mode")
    
    def _create_pipeline_config(self) -> PipelineConfig:
        """Create pipeline configuration based on test config."""
        if self.test_config.quality_mode == "fast":
            return PipelineConfig(
                quality_mode="fast",
                target_fps=10,
                min_detection_confidence=0.3,
                max_workers=2,
                enable_caching=True,
                enable_parallel_move_analysis=True,
                lazy_loading=True,
                cleanup_after_generation=True
            )
        elif self.test_config.quality_mode == "high_quality":
            return PipelineConfig(
                quality_mode="high_quality",
                target_fps=30,
                min_detection_confidence=0.6,
                max_workers=6,
                enable_caching=True,
                enable_parallel_move_analysis=True,
                lazy_loading=True,
                cleanup_after_generation=False  # Keep files for quality analysis
            )
        else:  # balanced
            return PipelineConfig(
                quality_mode="balanced",
                target_fps=20,
                min_detection_confidence=0.4,
                max_workers=4,
                enable_caching=True,
                enable_parallel_move_analysis=True,
                lazy_loading=True,
                cleanup_after_generation=True
            )
    
    def _create_optimization_config(self) -> OptimizationConfig:
        """Create optimization configuration based on test config."""
        return OptimizationConfig(
            enable_embedding_cache=True,
            enable_similarity_cache=True,
            enable_precomputed_matrices=True,
            batch_size=32 if self.test_config.quality_mode == "fast" else 16,
            max_workers=4,
            enable_parallel_scoring=True,
            fast_mode=self.test_config.quality_mode == "fast",
            similarity_threshold=0.1 if self.test_config.quality_mode == "fast" else 0.05
        )
    
    async def run_comprehensive_test(
        self,
        audio_inputs: List[str],
        difficulty: str = "intermediate"
    ) -> BatchTestResult:
        """
        Run comprehensive test suite with multiple audio inputs.
        Always generates choreography for the full song duration.
        """
        logger.info(f"Starting comprehensive test with {len(audio_inputs)} inputs")
        logger.info(f"Configuration: {self.test_config.quality_mode} quality, full song duration")
        
        start_time = time.time()
        test_results = []
        
        # Run individual tests
        for i, audio_input in enumerate(audio_inputs):
            test_name = f"test_{i+1}_{Path(audio_input).stem if Path(audio_input).exists() else 'youtube'}"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Test {i+1}/{len(audio_inputs)}: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                test_result = await self._run_single_test(
                    test_name, audio_input, difficulty
                )
                test_results.append(test_result)
                
                if test_result.success:
                    logger.info(f"✅ Test {i+1} PASSED in {test_result.execution_time:.2f}s")
                else:
                    logger.error(f"❌ Test {i+1} FAILED: {test_result.error_message}")
                
            except Exception as e:
                logger.error(f"❌ Test {i+1} CRASHED: {e}")
                test_results.append(TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=0.0,
                    error_message=f"Test crashed: {str(e)}"
                ))
        
        # Create batch result
        total_time = time.time() - start_time
        successful_tests = sum(1 for r in test_results if r.success)
        failed_tests = len(test_results) - successful_tests
        
        batch_result = BatchTestResult(
            total_tests=len(test_results),
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            total_execution_time=total_time,
            average_execution_time=total_time / len(test_results) if test_results else 0.0,
            individual_results=test_results,
            performance_summary=self._generate_performance_summary(test_results)
        )
        
        # Generate comprehensive report
        await self._generate_comprehensive_report(batch_result)
        
        # Print summary
        self._print_test_summary(batch_result)
        
        return batch_result
    
    async def _run_single_test(
        self,
        test_name: str,
        audio_input: str,
        difficulty: str
    ) -> TestResult:
        """Run a single comprehensive test with full validation."""
        start_time = time.time()
        
        try:
            # Monitor system resources
            initial_memory = self._get_memory_usage()
            
            # Run pipeline for full song
            pipeline_result = await self.pipeline.generate_choreography(
                audio_input=audio_input,
                difficulty=difficulty
            )
            
            # Monitor final resources
            final_memory = self._get_memory_usage()
            memory_usage = final_memory - initial_memory
            
            execution_time = time.time() - start_time
            
            if pipeline_result.success:
                # Validate output
                validation_result = await self._validate_pipeline_output(pipeline_result)
                
                # Get output metrics
                output_metrics = self._get_output_metrics(pipeline_result.output_path)
                
                test_result = TestResult(
                    test_name=test_name,
                    success=validation_result["overall_success"],
                    execution_time=execution_time,
                    pipeline_result=pipeline_result,
                    output_file_size_mb=output_metrics["file_size_mb"],
                    video_duration=output_metrics["duration"],
                    moves_count=output_metrics["moves_count"],
                    memory_usage_mb=memory_usage,
                    error_message=None if validation_result["overall_success"] else validation_result["error_summary"]
                )
            else:
                test_result = TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=execution_time,
                    error_message=pipeline_result.error_message,
                    memory_usage_mb=memory_usage
                )
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Test {test_name} failed with exception: {e}")
            logger.error(traceback.format_exc())
            
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time=execution_time,
                error_message=f"Exception: {str(e)}"
            )
    
    async def _validate_pipeline_output(self, pipeline_result: PipelineResult) -> Dict[str, Any]:
        """Comprehensive validation of pipeline output."""
        validation_results = {
            "file_exists": False,
            "file_size_valid": False,
            "video_playable": False,
            "metadata_exists": False,
            "duration_reasonable": False,
            "overall_success": False,
            "error_summary": ""
        }
        
        errors = []
        
        try:
            # Check if output file exists
            if pipeline_result.output_path and Path(pipeline_result.output_path).exists():
                validation_results["file_exists"] = True
                
                # Check file size
                file_size = Path(pipeline_result.output_path).stat().st_size
                if file_size > 1024 * 1024:  # At least 1MB
                    validation_results["file_size_valid"] = True
                else:
                    errors.append(f"File too small: {file_size} bytes")
                
                # Check video duration using ffprobe
                duration = await self._get_video_duration(pipeline_result.output_path)
                if duration and duration > 5.0:  # At least 5 seconds
                    validation_results["duration_reasonable"] = True
                    validation_results["video_playable"] = True
                else:
                    errors.append(f"Invalid duration: {duration}")
            else:
                errors.append("Output file does not exist")
            
            # Check metadata file
            if pipeline_result.metadata_path and Path(pipeline_result.metadata_path).exists():
                validation_results["metadata_exists"] = True
            else:
                errors.append("Metadata file missing")
            
            # Overall success
            validation_results["overall_success"] = all([
                validation_results["file_exists"],
                validation_results["file_size_valid"],
                validation_results["video_playable"]
            ])
            
            if errors:
                validation_results["error_summary"] = "; ".join(errors)
            
        except Exception as e:
            validation_results["error_summary"] = f"Validation failed: {str(e)}"
        
        return validation_results
    
    async def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration using ffprobe."""
        try:
            import subprocess
            import json
            
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return float(info.get("format", {}).get("duration", 0))
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
        
        return None
    
    def _get_output_metrics(self, output_path: Optional[str]) -> Dict[str, Any]:
        """Get metrics for output file."""
        metrics = {
            "file_size_mb": 0.0,
            "duration": 0.0,
            "moves_count": 0
        }
        
        if output_path and Path(output_path).exists():
            # File size
            file_size = Path(output_path).stat().st_size
            metrics["file_size_mb"] = file_size / (1024 * 1024)
            
            # Duration (would need ffprobe, simplified for now)
            metrics["duration"] = 60.0  # Placeholder
            
            # Moves count (would need metadata parsing)
            metrics["moves_count"] = 8  # Placeholder
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _generate_performance_summary(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Generate performance summary from test results."""
        successful_results = [r for r in test_results if r.success]
        
        if not successful_results:
            return {"error": "No successful tests to analyze"}
        
        execution_times = [r.execution_time for r in successful_results]
        memory_usages = [r.memory_usage_mb for r in successful_results if r.memory_usage_mb > 0]
        file_sizes = [r.output_file_size_mb for r in successful_results if r.output_file_size_mb > 0]
        
        summary = {
            "execution_time": {
                "min": min(execution_times),
                "max": max(execution_times),
                "avg": sum(execution_times) / len(execution_times),
                "total": sum(execution_times)
            },
            "memory_usage": {
                "min": min(memory_usages) if memory_usages else 0,
                "max": max(memory_usages) if memory_usages else 0,
                "avg": sum(memory_usages) / len(memory_usages) if memory_usages else 0
            },
            "output_file_size": {
                "min": min(file_sizes) if file_sizes else 0,
                "max": max(file_sizes) if file_sizes else 0,
                "avg": sum(file_sizes) / len(file_sizes) if file_sizes else 0
            },
            "success_rate": len(successful_results) / len(test_results),
            "quality_mode": self.test_config.quality_mode
        }
        
        return summary
    
    async def _generate_comprehensive_report(self, batch_result: BatchTestResult) -> None:
        """Generate comprehensive test report."""
        if not self.test_config.generate_detailed_reports:
            return
        
        report_path = Path(self.test_config.test_results_dir) / f"test_report_{int(time.time())}.json"
        
        report_data = {
            "test_configuration": asdict(self.test_config),
            "batch_results": asdict(batch_result),
            "pipeline_metrics": self.pipeline._cache_hits if hasattr(self.pipeline, '_cache_hits') else 0,
            "recommendation_metrics": self.optimized_engine.get_performance_metrics(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self._get_system_info()
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"📊 Comprehensive report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report."""
        try:
            import platform
            import psutil
            
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_free_gb": psutil.disk_usage('.').free / (1024**3)
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def _print_test_summary(self, batch_result: BatchTestResult) -> None:
        """Print comprehensive test summary."""
        print(f"\n{'='*80}")
        print(f"🎵 COMPREHENSIVE TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {batch_result.total_tests}")
        print(f"   ✅ Successful: {batch_result.successful_tests}")
        print(f"   ❌ Failed: {batch_result.failed_tests}")
        print(f"   📈 Success Rate: {batch_result.successful_tests/batch_result.total_tests*100:.1f}%")
        
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Total Execution Time: {batch_result.total_execution_time:.2f}s")
        print(f"   Average per Test: {batch_result.average_execution_time:.2f}s")
        
        if batch_result.performance_summary.get("execution_time"):
            exec_stats = batch_result.performance_summary["execution_time"]
            print(f"   Fastest Test: {exec_stats['min']:.2f}s")
            print(f"   Slowest Test: {exec_stats['max']:.2f}s")
        
        print(f"\n🎯 Quality Mode: {self.test_config.quality_mode.upper()}")
        
        # Print individual test results
        print(f"\n📋 Individual Test Results:")
        for i, result in enumerate(batch_result.individual_results, 1):
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"   {i:2d}. {result.test_name:<30} {status} ({result.execution_time:.2f}s)")
            if not result.success and result.error_message:
                print(f"       Error: {result.error_message}")
        
        # Recommendation engine metrics
        rec_metrics = self.optimized_engine.get_performance_metrics()
        if rec_metrics["total_requests"] > 0:
            print(f"\n🚀 Recommendation Engine Performance:")
            print(f"   Cache Hit Rate: {rec_metrics['cache_hit_rate']*100:.1f}%")
            print(f"   Avg Response Time: {rec_metrics['avg_response_time']*1000:.1f}ms")
            print(f"   Total Requests: {rec_metrics['total_requests']}")
        
        print(f"\n{'='*80}")
    
    async def run_batch_processing_test(
        self,
        audio_inputs: List[str],
        batch_size: int = 3
    ) -> BatchTestResult:
        """Test batch processing capabilities."""
        logger.info(f"Running batch processing test with {len(audio_inputs)} inputs, batch size {batch_size}")
        
        # Split inputs into batches
        batches = [
            audio_inputs[i:i + batch_size]
            for i in range(0, len(audio_inputs), batch_size)
        ]
        
        all_results = []
        start_time = time.time()
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")
            
            # Process batch concurrently
            batch_tasks = [
                self._run_single_test(
                    f"batch_{batch_idx}_item_{i}",
                    audio_input,
                    "1min",
                    "intermediate"
                )
                for i, audio_input in enumerate(batch)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    all_results.append(TestResult(
                        test_name=f"batch_{batch_idx}_item_{i}",
                        success=False,
                        execution_time=0.0,
                        error_message=f"Batch processing exception: {str(result)}"
                    ))
                else:
                    all_results.append(result)
        
        total_time = time.time() - start_time
        successful_tests = sum(1 for r in all_results if r.success)
        
        return BatchTestResult(
            total_tests=len(all_results),
            successful_tests=successful_tests,
            failed_tests=len(all_results) - successful_tests,
            total_execution_time=total_time,
            average_execution_time=total_time / len(all_results) if all_results else 0.0,
            individual_results=all_results,
            performance_summary=self._generate_performance_summary(all_results)
        )
    
    def list_available_test_songs(self) -> List[str]:
        """List available songs for testing."""
        songs_dir = Path("data/songs")
        if not songs_dir.exists():
            return []
        
        songs = []
        for file_path in songs_dir.glob("*.mp3"):
            songs.append(str(file_path))
        
        return sorted(songs)
    
    async def run_error_recovery_test(self) -> TestResult:
        """Test error recovery mechanisms."""
        logger.info("Running error recovery test...")
        
        start_time = time.time()
        
        # Test with invalid inputs
        test_cases = [
            ("nonexistent_file.mp3", "File not found recovery"),
            ("https://invalid-youtube-url", "Invalid URL recovery"),
            ("", "Empty input recovery")
        ]
        
        recovery_successes = 0
        
        for invalid_input, test_description in test_cases:
            try:
                result = await self.pipeline.generate_choreography(
                    audio_input=invalid_input,
                    duration="30s"
                )
                
                # Should fail gracefully
                if not result.success and result.error_message:
                    recovery_successes += 1
                    logger.info(f"✅ {test_description}: Graceful failure")
                else:
                    logger.warning(f"⚠️  {test_description}: Unexpected success")
                    
            except Exception as e:
                logger.error(f"❌ {test_description}: Unhandled exception: {e}")
        
        execution_time = time.time() - start_time
        success = recovery_successes == len(test_cases)
        
        return TestResult(
            test_name="error_recovery_test",
            success=success,
            execution_time=execution_time,
            error_message=None if success else f"Only {recovery_successes}/{len(test_cases)} recovery tests passed"
        )


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Optimized Bachata Choreography Generator")
    parser.add_argument("inputs", nargs="*", help="Audio files or YouTube URLs to process")
    parser.add_argument("--quality", choices=["fast", "balanced", "high_quality"], 
                       default="balanced", help="Processing quality mode")

    parser.add_argument("--difficulty", choices=["beginner", "intermediate", "advanced"], 
                       default="intermediate", help="Target difficulty level")
    parser.add_argument("--batch", action="store_true", help="Enable batch processing")
    parser.add_argument("--test-all", action="store_true", help="Run comprehensive test suite")
    parser.add_argument("--test-recovery", action="store_true", help="Test error recovery")
    parser.add_argument("--list-songs", action="store_true", help="List available test songs")
    
    args = parser.parse_args()
    
    # Create test configuration
    test_config = TestConfig(
        quality_mode=args.quality,
        enable_batch_processing=args.batch,
        enable_performance_benchmarking=True,
        enable_error_recovery_testing=args.test_recovery
    )
    
    # Initialize generator
    generator = OptimizedChoreoGenerator(test_config)
    
    if args.list_songs:
        songs = generator.list_available_test_songs()
        print(f"Available test songs ({len(songs)}):")
        for i, song in enumerate(songs, 1):
            print(f"  {i:2d}. {Path(song).name}")
        return
    
    if args.test_recovery:
        result = await generator.run_error_recovery_test()
        print(f"Error recovery test: {'PASSED' if result.success else 'FAILED'}")
        if not result.success:
            print(f"Error: {result.error_message}")
        return
    
    if args.test_all:
        # Use available songs for comprehensive testing
        test_songs = generator.list_available_test_songs()[:5]  # Limit to 5 for testing
        if not test_songs:
            print("No test songs available. Please add MP3 files to data/songs/")
            return
        
        print(f"Running comprehensive test with {len(test_songs)} songs...")
        batch_result = await generator.run_comprehensive_test(
            test_songs, args.difficulty
        )
        return
    
    if not args.inputs:
        print("No inputs provided. Use --help for usage information.")
        return
    
    # Process provided inputs
    if args.batch:
        batch_result = await generator.run_batch_processing_test(args.inputs)
        print(f"Batch processing completed: {batch_result.successful_tests}/{batch_result.total_tests} successful")
    else:
        # Process single input
        batch_result = await generator.run_comprehensive_test(
            args.inputs, args.difficulty
        )


if __name__ == "__main__":
    asyncio.run(main())