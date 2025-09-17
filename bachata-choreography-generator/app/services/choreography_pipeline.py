"""
Optimized service integration layer for Bachata choreography generation.
Efficiently coordinates all services with caching, parallel processing, and smart initialization.
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import hashlib
import pickle

from .music_analyzer import MusicAnalyzer, MusicFeatures
from .move_analyzer import MoveAnalyzer, MoveAnalysisResult
from .recommendation_engine import (
    RecommendationEngine, RecommendationRequest, MoveCandidate, RecommendationScore
)
from .superlinked_recommendation_engine import (
    SuperlinkedRecommendationEngine, SuperlinkedRecommendationScore, create_superlinked_recommendation_engine
)
from .video_generator import VideoGenerator, VideoGenerationConfig
from .annotation_interface import AnnotationInterface
from .feature_fusion import FeatureFusion, MultiModalEmbedding
from .youtube_service import YouTubeService
from .qdrant_service import create_qdrant_service, QdrantConfig, SuperlinkedQdrantService, MockSuperlinkedQdrantService
from ..models.video_models import ChoreographySequence

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the choreography generation pipeline."""
    # Processing quality settings
    quality_mode: str = "balanced"  # fast, balanced, high_quality
    target_fps: int = 20
    min_detection_confidence: float = 0.4
    
    # Caching settings
    enable_caching: bool = True
    cache_dir: str = "data/cache"
    cache_ttl_hours: int = 24
    
    # Parallel processing settings
    max_workers: int = 4
    enable_parallel_move_analysis: bool = True
    
    # Memory management
    lazy_loading: bool = True
    cleanup_after_generation: bool = True
    max_cache_size_mb: int = 500
    
    # Output settings
    output_dir: str = "data/output"
    temp_dir: str = "data/temp"
    
    # Service-specific settings
    youtube_output_dir: str = "data/temp"
    annotation_data_dir: str = "data"
    
    # Qdrant settings
    enable_qdrant: bool = True
    auto_populate_qdrant: bool = True  # Automatically populate Qdrant with analyzed moves


@dataclass
class PipelineResult:
    """Result of choreography generation pipeline."""
    success: bool
    output_path: Optional[str] = None
    metadata_path: Optional[str] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    
    # Detailed metrics
    music_analysis_time: float = 0.0
    move_analysis_time: float = 0.0
    recommendation_time: float = 0.0
    video_generation_time: float = 0.0
    
    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Quality metrics
    moves_analyzed: int = 0
    recommendations_generated: int = 0
    sequence_duration: float = 0.0
    
    # Qdrant statistics
    qdrant_enabled: bool = False
    qdrant_embeddings_stored: int = 0
    qdrant_embeddings_retrieved: int = 0
    qdrant_search_time: float = 0.0


class ServiceCache:
    """Thread-safe caching system for service results."""
    
    def __init__(self, cache_dir: str, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self._lock = threading.Lock()
        
        # In-memory cache for frequently accessed items
        self._memory_cache = {}
        self._memory_cache_timestamps = {}
        
    def _get_cache_key(self, service: str, input_data: Any) -> str:
        """Generate cache key from service name and input data."""
        if isinstance(input_data, str):
            # For file paths, use file path + modification time
            if Path(input_data).exists():
                mtime = Path(input_data).stat().st_mtime
                key_data = f"{service}:{input_data}:{mtime}"
            else:
                key_data = f"{service}:{input_data}"
        else:
            # For other data, serialize and hash
            key_data = f"{service}:{str(input_data)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, service: str, input_data: Any) -> Optional[Any]:
        """Get cached result for service and input."""
        cache_key = self._get_cache_key(service, input_data)
        
        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                timestamp = self._memory_cache_timestamps[cache_key]
                if time.time() - timestamp < self.ttl_seconds:
                    logger.debug(f"Memory cache hit for {service}")
                    return self._memory_cache[cache_key]
                else:
                    # Expired, remove from memory cache
                    del self._memory_cache[cache_key]
                    del self._memory_cache_timestamps[cache_key]
            
            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    file_age = time.time() - cache_file.stat().st_mtime
                    if file_age < self.ttl_seconds:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        
                        # Store in memory cache for faster access
                        self._memory_cache[cache_key] = result
                        self._memory_cache_timestamps[cache_key] = time.time()
                        
                        logger.debug(f"Disk cache hit for {service}")
                        return result
                    else:
                        # Expired, remove file
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return None
    
    def set(self, service: str, input_data: Any, result: Any) -> None:
        """Cache result for service and input."""
        cache_key = self._get_cache_key(service, input_data)
        
        with self._lock:
            # Store in memory cache
            self._memory_cache[cache_key] = result
            self._memory_cache_timestamps[cache_key] = time.time()
            
            # Store in disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {service}")
            except Exception as e:
                logger.warning(f"Error writing cache file {cache_file}: {e}")
    
    def clear_expired(self) -> None:
        """Clear expired cache entries."""
        current_time = time.time()
        
        with self._lock:
            # Clear expired memory cache entries
            expired_keys = [
                key for key, timestamp in self._memory_cache_timestamps.items()
                if current_time - timestamp >= self.ttl_seconds
            ]
            for key in expired_keys:
                del self._memory_cache[key]
                del self._memory_cache_timestamps[key]
            
            # Clear expired disk cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age >= self.ttl_seconds:
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error removing expired cache file {cache_file}: {e}")
    
    def get_cache_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                total_size += cache_file.stat().st_size
            except Exception:
                pass
        return total_size / (1024 * 1024)


class ChoreoGenerationPipeline:
    """
    Optimized choreography generation pipeline that efficiently coordinates all services.
    Features:
    - Service caching to avoid redundant analysis
    - Parallel processing for move analysis
    - Smart service initialization (lazy loading)
    - Memory optimization and cleanup
    - Comprehensive error handling and recovery
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the choreography generation pipeline."""
        self.config = config or PipelineConfig()
        
        # Initialize cache system
        if self.config.enable_caching:
            self.cache = ServiceCache(self.config.cache_dir, self.config.cache_ttl_hours)
        else:
            self.cache = None
        
        # Service instances (lazy loaded)
        self._music_analyzer = None
        self._move_analyzer = None
        self._recommendation_engine = None
        self._video_generator = None
        self._annotation_interface = None
        self._feature_fusion = None
        self._youtube_service = None
        self._qdrant_service = None
        
        # Thread pool for parallel processing
        self._executor = None
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._qdrant_embeddings_stored = 0
        self._qdrant_embeddings_retrieved = 0
        
        # Ensure directories exist
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChoreoGenerationPipeline initialized with {self.config.quality_mode} quality mode")
    
    @property
    def music_analyzer(self) -> MusicAnalyzer:
        """Lazy-loaded music analyzer."""
        if self._music_analyzer is None:
            logger.debug("Initializing MusicAnalyzer")
            self._music_analyzer = MusicAnalyzer()
        return self._music_analyzer
    
    @property
    def move_analyzer(self) -> MoveAnalyzer:
        """Lazy-loaded move analyzer."""
        if self._move_analyzer is None:
            logger.debug("Initializing MoveAnalyzer")
            self._move_analyzer = MoveAnalyzer(
                target_fps=self.config.target_fps,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_detection_confidence
            )
        return self._move_analyzer
    
    @property
    def recommendation_engine(self) -> SuperlinkedRecommendationEngine:
        """Lazy-loaded SuperlinkedRecommendationEngine with unified vector similarity and Qdrant integration."""
        if self._recommendation_engine is None:
            logger.debug("Initializing SuperlinkedRecommendationEngine with unified embeddings and Qdrant integration")
            
            # Create Qdrant config for the recommendation engine
            qdrant_config = None
            if self.config.enable_qdrant:
                # Use environment-based configuration for cloud deployment
                qdrant_config = QdrantConfig.from_env()
            
            self._recommendation_engine = create_superlinked_recommendation_engine(
                data_dir=self.config.annotation_data_dir,
                qdrant_config=qdrant_config
            )
            
            logger.info("SuperlinkedRecommendationEngine initialized with Qdrant integration - eliminated multi-factor scoring")
                
        return self._recommendation_engine
    
    @property
    def video_generator(self) -> VideoGenerator:
        """Lazy-loaded video generator."""
        if self._video_generator is None:
            logger.debug("Initializing VideoGenerator")
            
            # Configure based on quality mode
            if self.config.quality_mode == "fast":
                video_config = VideoGenerationConfig(
                    output_path=f"{self.config.output_dir}/choreography_fast.mp4",
                    resolution="1280x720",
                    video_bitrate="4M",
                    audio_bitrate="128k",
                    cleanup_temp_files=self.config.cleanup_after_generation
                )
            elif self.config.quality_mode == "high_quality":
                video_config = VideoGenerationConfig(
                    output_path=f"{self.config.output_dir}/choreography_hq.mp4",
                    resolution="1920x1080",
                    video_bitrate="8M",
                    audio_bitrate="320k",
                    cleanup_temp_files=self.config.cleanup_after_generation
                )
            else:  # balanced
                video_config = VideoGenerationConfig(
                    output_path=f"{self.config.output_dir}/choreography_balanced.mp4",
                    resolution="1280x720",
                    video_bitrate="6M",
                    audio_bitrate="192k",
                    cleanup_temp_files=self.config.cleanup_after_generation
                )
            
            self._video_generator = VideoGenerator(video_config)
        return self._video_generator
    
    @property
    def annotation_interface(self) -> AnnotationInterface:
        """Lazy-loaded annotation interface."""
        if self._annotation_interface is None:
            logger.debug("Initializing AnnotationInterface")
            self._annotation_interface = AnnotationInterface(
                data_dir=self.config.annotation_data_dir
            )
        return self._annotation_interface
    
    @property
    def feature_fusion(self) -> FeatureFusion:
        """Lazy-loaded feature fusion."""
        if self._feature_fusion is None:
            logger.debug("Initializing FeatureFusion")
            self._feature_fusion = FeatureFusion()
        return self._feature_fusion
    
    @property
    def youtube_service(self) -> YouTubeService:
        """Lazy-loaded YouTube service."""
        if self._youtube_service is None:
            logger.debug("Initializing YouTubeService")
            self._youtube_service = YouTubeService(
                output_dir=self.config.youtube_output_dir
            )
        return self._youtube_service
    
    @property
    def executor(self) -> ThreadPoolExecutor:
        """Lazy-loaded thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self._executor
    
    @property
    def qdrant_service(self) -> Optional[SuperlinkedQdrantService]:
        """Lazy-loaded Superlinked Qdrant service with cloud deployment support."""
        if self._qdrant_service is None and self.config.enable_qdrant:
            logger.debug("Initializing SuperlinkedQdrantService for cloud deployment")
            try:
                # Use environment-based configuration for cloud deployment
                qdrant_config = QdrantConfig.from_env()
                self._qdrant_service = create_qdrant_service(qdrant_config)
                
                # Perform health check
                health_status = self._qdrant_service.health_check()
                if health_status.get("qdrant_available", False):
                    if qdrant_config.url:
                        logger.info(f"SuperlinkedQdrantService initialized successfully (Cloud): {qdrant_config.url}")
                    else:
                        logger.info(f"SuperlinkedQdrantService initialized successfully (Local): {qdrant_config.host}:{qdrant_config.port}")
                    
                    # Auto-populate if requested and collection is empty
                    if self.config.auto_populate_qdrant:
                        self._auto_populate_qdrant_if_needed()
                else:
                    logger.warning(f"Qdrant health check failed: {health_status.get('error_message', 'Unknown error')}")
                    if isinstance(self._qdrant_service, MockSuperlinkedQdrantService):
                        logger.info("Using mock Superlinked Qdrant service - embeddings will not be persisted")
                
            except Exception as e:
                logger.error(f"Failed to initialize SuperlinkedQdrantService: {e}")
                logger.info("Continuing without Qdrant - using in-memory similarity search only")
                self._qdrant_service = None
        
        return self._qdrant_service
    
    def _auto_populate_qdrant_if_needed(self) -> None:
        """Auto-populate Qdrant collection with existing move embeddings if empty."""
        try:
            if not self.qdrant_service or isinstance(self.qdrant_service, MockSuperlinkedQdrantService):
                return
            
            # Check if collection has any points
            collection_info = self.qdrant_service.get_collection_info()
            points_count = collection_info.get("points_count", 0)
            
            if points_count == 0:
                logger.info("Qdrant collection is empty - auto-populating with existing move embeddings")
                # This will be done lazily during the first choreography generation
                # to avoid blocking initialization
            else:
                logger.info(f"Qdrant collection already contains {points_count} move embeddings")
                
        except Exception as e:
            logger.warning(f"Failed to check Qdrant collection status: {e}")
    
    def get_qdrant_health_status(self) -> Dict[str, Any]:
        """Get comprehensive Qdrant health status."""
        if not self.config.enable_qdrant:
            return {"enabled": False, "status": "disabled"}
        
        try:
            if self.qdrant_service:
                health_status = self.qdrant_service.health_check()
                collection_info = self.qdrant_service.get_collection_info()
                stats = self.qdrant_service.get_statistics()
                
                return {
                    "enabled": True,
                    "status": "healthy" if health_status.get("qdrant_available") else "unhealthy",
                    "health_check": health_status,
                    "collection_info": collection_info,
                    "statistics": {
                        "total_points": stats.total_points,
                        "search_requests": stats.search_requests,
                        "avg_search_time_ms": stats.avg_search_time_ms,
                        "collection_size_mb": stats.collection_size_mb
                    }
                }
            else:
                return {"enabled": True, "status": "not_initialized"}
                
        except Exception as e:
            return {"enabled": True, "status": "error", "error": str(e)}
    
    async def _populate_qdrant_with_moves(self, move_candidates: List[MoveCandidate]) -> None:
        """Populate Qdrant with move embeddings if not already done."""
        try:
            if not self.recommendation_engine.is_qdrant_available():
                logger.debug("Qdrant not available, skipping population")
                return
            
            # Check if already populated
            if self._qdrant_embeddings_stored > 0:
                logger.debug("Qdrant already populated, skipping")
                return
            
            logger.info(f"Populating Qdrant with {len(move_candidates)} move embeddings")
            start_time = time.time()
            
            # Use the recommendation engine's populate method
            summary = self.recommendation_engine.populate_qdrant_from_candidates(move_candidates)
            
            if "error" not in summary:
                self._qdrant_embeddings_stored = summary.get("successful_migrations", 0)
                population_time = time.time() - start_time
                
                logger.info(f"Successfully populated Qdrant with {self._qdrant_embeddings_stored} embeddings in {population_time:.2f}s")
            else:
                logger.error(f"Failed to populate Qdrant: {summary['error']}")
                
        except Exception as e:
            logger.error(f"Error populating Qdrant: {e}")
    
    def _populate_qdrant_on_startup(self) -> None:
        """Populate Qdrant on startup (called from recommendation_engine property)."""
        # This is a placeholder - actual population happens during first choreography generation
        # to avoid blocking initialization with potentially expensive operations
        logger.debug("Qdrant auto-population will occur during first choreography generation")
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance statistics for SuperlinkedRecommendationEngine."""
        if self.recommendation_engine:
            engine_stats = self.recommendation_engine.get_performance_stats()
            
            # Add pipeline-level statistics
            pipeline_stats = {
                "pipeline_cache_hits": self._cache_hits,
                "pipeline_cache_misses": self._cache_misses,
                "pipeline_total_generations": getattr(self, '_total_generations', 0),
                "engine_type": "SuperlinkedRecommendationEngine",
                "unified_vector_approach": True,
                "eliminated_multi_factor_scoring": True
            }
            
            # Combine statistics
            combined_stats = {**engine_stats, **pipeline_stats}
            
            # Log performance information
            performance = combined_stats.get('performance', {})
            unified_searches = performance.get('unified_searches', 0)
            avg_time = performance.get('avg_search_time_ms', 0)
            
            if unified_searches > 0 and avg_time > 0:
                logger.info(f"SuperlinkedRecommendationEngine: {unified_searches} unified vector searches, "
                          f"avg {avg_time:.2f}ms per search")
            
            return combined_stats
        else:
            return {"error": "SuperlinkedRecommendationEngine not initialized"}
    
    async def generate_choreography(
        self,
        audio_input: str,
        difficulty: str = "intermediate",
        energy_level: Optional[str] = None,
        role_focus: Optional[str] = None,
        move_types: Optional[List[str]] = None,
        tempo_range: Optional[List[int]] = None
    ) -> PipelineResult:
        """
        Generate choreography from audio input with optimized pipeline.
        Always creates choreography for the full song duration.
        
        Args:
            audio_input: Path to audio file or YouTube URL
            difficulty: Target difficulty ("beginner", "intermediate", "advanced")
            energy_level: Target energy level (None for auto-detect, or "low"/"medium"/"high")
            role_focus: Role focus preference ("lead_focus", "follow_focus", "both")
            move_types: List of preferred move types
            tempo_range: Tempo range [min, max] in BPM
        
        Returns:
            PipelineResult with generation results and metrics
        """
        start_time = time.time()
        result = PipelineResult(success=False)
        
        try:
            logger.info(f"Starting choreography generation: {audio_input}")
            
            # Step 1: Get audio file (with caching for YouTube downloads)
            audio_path, audio_time = await self._get_audio_file(audio_input)
            if not audio_path:
                result.error_message = "Failed to get audio file"
                return result
            
            # Step 2: Analyze music (with caching)
            music_features, music_time = await self._analyze_music_cached(audio_path)
            if not music_features:
                result.error_message = "Music analysis failed"
                return result
            result.music_analysis_time = music_time
            
            # Step 3: Analyze moves (with parallel processing and caching)
            move_candidates, moves_time = await self._analyze_moves_parallel(music_features)
            if not move_candidates:
                result.error_message = "Move analysis failed"
                return result
            result.move_analysis_time = moves_time
            result.moves_analyzed = len(move_candidates)
            
            # Step 4: Generate recommendations (optimized)
            recommendations, rec_time = await self._generate_recommendations_optimized(
                music_features, move_candidates, difficulty, energy_level
            )
            if not recommendations:
                result.error_message = "Recommendation generation failed"
                return result
            result.recommendation_time = rec_time
            result.recommendations_generated = len(recommendations)
            
            # Step 5: Create sequence for full song
            sequence = await self._create_optimized_sequence(
                recommendations, music_features
            )
            if not sequence:
                result.error_message = "Sequence creation failed"
                return result
            result.sequence_duration = sequence.total_duration
            
            # Step 6: Generate video
            video_result, video_time = await self._generate_video_optimized(
                sequence, music_features, audio_path
            )
            if not video_result:
                result.error_message = "Video generation failed"
                return result
            result.video_generation_time = video_time
            
            # Step 7: Export metadata
            metadata_path = await self._export_metadata(
                sequence, video_result, music_features, audio_path
            )
            
            # Cleanup if requested
            if self.config.cleanup_after_generation:
                await self._cleanup_resources()
            
            # Set success result
            result.success = True
            result.output_path = video_result.get("output_path")
            result.metadata_path = metadata_path
            result.processing_time = time.time() - start_time
            result.cache_hits = self._cache_hits
            result.cache_misses = self._cache_misses
            
            # Qdrant statistics
            result.qdrant_enabled = self.config.enable_qdrant and self.qdrant_service is not None
            result.qdrant_embeddings_stored = self._qdrant_embeddings_stored
            result.qdrant_embeddings_retrieved = self._qdrant_embeddings_retrieved
            
            # Get search time from SuperlinkedRecommendationEngine
            if self.recommendation_engine:
                perf_stats = self.recommendation_engine.get_performance_stats()
                performance = perf_stats.get('performance', {})
                result.qdrant_search_time = performance.get('avg_search_time_ms', 0.0) / 1000.0  # Convert to seconds
            
            logger.info(f"Choreography generation completed successfully in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            result.cache_hits = self._cache_hits
            result.cache_misses = self._cache_misses
            result.qdrant_enabled = self.config.enable_qdrant and self.qdrant_service is not None
            result.qdrant_embeddings_stored = self._qdrant_embeddings_stored
            result.qdrant_embeddings_retrieved = self._qdrant_embeddings_retrieved
            
            # Get Qdrant search time from recommendation engine (error case)
            if self.recommendation_engine:
                try:
                    perf_stats = self.recommendation_engine.get_performance_stats()
                    performance = perf_stats.get('performance', {})
                    result.qdrant_search_time = performance.get('avg_search_time_ms', 0.0) / 1000.0  # Convert to seconds
                except Exception:
                    result.qdrant_search_time = 0.0
            
            return result
    
    async def _get_audio_file(self, audio_input: str) -> Tuple[Optional[str], float]:
        """Get audio file with caching for YouTube downloads."""
        start_time = time.time()
        
        # Check if it's a YouTube URL
        if self.youtube_service.validate_url(audio_input):
            # Check cache for YouTube downloads
            if self.cache:
                cached_result = self.cache.get("youtube_download", audio_input)
                if cached_result:
                    self._cache_hits += 1
                    logger.info(f"Using cached YouTube download: {cached_result}")
                    return cached_result, time.time() - start_time
            
            # Download from YouTube
            self._cache_misses += 1
            logger.info("Downloading from YouTube...")
            download_result = await self.youtube_service.download_audio(audio_input)
            
            if download_result.success:
                audio_path = download_result.file_path
                # Cache the result
                if self.cache:
                    self.cache.set("youtube_download", audio_input, audio_path)
                return audio_path, time.time() - start_time
            else:
                logger.error(f"YouTube download failed: {download_result.error_message}")
                return None, time.time() - start_time
        
        # Check if local file exists
        elif Path(audio_input).exists():
            return audio_input, time.time() - start_time
        
        else:
            logger.error(f"Audio file not found: {audio_input}")
            return None, time.time() - start_time
    
    async def _analyze_music_cached(self, audio_path: str) -> Tuple[Optional[MusicFeatures], float]:
        """Analyze music with caching."""
        start_time = time.time()
        
        # Check cache
        if self.cache:
            cached_result = self.cache.get("music_analysis", audio_path)
            if cached_result:
                self._cache_hits += 1
                logger.info("Using cached music analysis")
                return cached_result, time.time() - start_time
        
        # Perform analysis
        self._cache_misses += 1
        logger.info("Analyzing music...")
        
        try:
            music_features = self.music_analyzer.analyze_audio(audio_path)
            
            # Cache the result
            if self.cache:
                self.cache.set("music_analysis", audio_path, music_features)
            
            return music_features, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Music analysis failed: {e}")
            return None, time.time() - start_time
    
    async def _analyze_moves_parallel(
        self, 
        music_features: MusicFeatures
    ) -> Tuple[Optional[List[MoveCandidate]], float]:
        """Analyze moves with parallel processing and caching."""
        start_time = time.time()
        
        try:
            # Load annotations
            collection = self.annotation_interface.load_annotations("bachata_annotations.json")
            logger.info(f"Loaded {collection.total_clips} move clips")
            
            # Select moves based on quality mode
            max_moves = {
                "fast": 8,
                "balanced": 12,
                "high_quality": 20
            }.get(self.config.quality_mode, 12)
            
            selected_clips = self._select_diverse_moves(collection.clips, max_moves)
            logger.info(f"Selected {len(selected_clips)} moves for analysis")
            
            # Analyze moves with parallel processing if enabled
            if self.config.enable_parallel_move_analysis and len(selected_clips) > 2:
                move_candidates = await self._analyze_moves_parallel_execution(
                    selected_clips, music_features
                )
            else:
                move_candidates = await self._analyze_moves_sequential(
                    selected_clips, music_features
                )
            
            return move_candidates, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Move analysis failed: {e}")
            return None, time.time() - start_time
    
    async def _analyze_moves_parallel_execution(
        self,
        clips: List,
        music_features: MusicFeatures
    ) -> List[MoveCandidate]:
        """Execute move analysis in parallel using thread pool."""
        move_candidates = []
        
        def analyze_single_move(clip) -> Optional[MoveCandidate]:
            """Analyze a single move clip with Qdrant integration."""
            try:
                video_path = Path(self.config.annotation_data_dir) / clip.video_path
                if not video_path.exists():
                    return None
                
                # Step 1: Check Qdrant first for existing embedding
                candidate = None
                if self.qdrant_service and not isinstance(self.qdrant_service, MockSuperlinkedQdrantService):
                    try:
                        qdrant_result = self.qdrant_service.get_move_by_id(clip.clip_id)
                        if qdrant_result:
                            logger.debug(f"Found move in Qdrant: {clip.clip_id}")
                            self._qdrant_embeddings_retrieved += 1
                            
                            # Reconstruct candidate from Qdrant data
                            # Note: We still need analysis_result, so we check cache or analyze
                            cache_key = str(video_path)
                            if self.cache:
                                cached_result = self.cache.get("move_analysis", cache_key)
                                if cached_result:
                                    analysis_result, _ = cached_result  # Ignore cached embedding, use Qdrant's
                                    self._cache_hits += 1
                                else:
                                    analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                                    self._cache_misses += 1
                            else:
                                analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                                self._cache_misses += 1
                            
                            # Create multimodal embedding from Qdrant vector
                            if qdrant_result.embedding is not None:
                                # Split the combined embedding back into audio and pose components
                                combined_embedding = qdrant_result.embedding
                                audio_embedding = combined_embedding[:128]  # First 128 dimensions
                                pose_embedding = combined_embedding[128:512]  # Next 384 dimensions
                                
                                multimodal_embedding = MultiModalEmbedding(
                                    audio_embedding=audio_embedding,
                                    pose_embedding=pose_embedding
                                )
                            else:
                                # Fallback: create new embedding
                                multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
                                    music_features, analysis_result
                                )
                            
                            candidate = MoveCandidate(
                                move_id=clip.clip_id,
                                video_path=str(video_path),
                                move_label=clip.move_label,
                                analysis_result=analysis_result,
                                multimodal_embedding=multimodal_embedding,
                                energy_level=clip.energy_level,
                                difficulty=clip.difficulty,
                                estimated_tempo=clip.estimated_tempo,
                                lead_follow_roles=clip.lead_follow_roles
                            )
                    except Exception as e:
                        logger.debug(f"Qdrant lookup failed for {clip.clip_id}: {e}")
                
                # Step 2: If not found in Qdrant, check cache or analyze
                if candidate is None:
                    cache_key = str(video_path)
                    if self.cache:
                        cached_result = self.cache.get("move_analysis", cache_key)
                        if cached_result:
                            analysis_result, multimodal_embedding = cached_result
                            self._cache_hits += 1
                        else:
                            # Perform analysis
                            analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                            multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
                                music_features, analysis_result
                            )
                            # Cache the result
                            self.cache.set("move_analysis", cache_key, (analysis_result, multimodal_embedding))
                            self._cache_misses += 1
                    else:
                        # No caching
                        analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                        multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
                            music_features, analysis_result
                        )
                        self._cache_misses += 1
                    
                    # Create candidate
                    candidate = MoveCandidate(
                        move_id=clip.clip_id,
                        video_path=str(video_path),
                        move_label=clip.move_label,
                        analysis_result=analysis_result,
                        multimodal_embedding=multimodal_embedding,
                        energy_level=clip.energy_level,
                        difficulty=clip.difficulty,
                        estimated_tempo=clip.estimated_tempo,
                        lead_follow_roles=clip.lead_follow_roles
                    )
                    
                    # Step 3: Store new embedding in Qdrant
                    if self.qdrant_service and not isinstance(self.qdrant_service, MockSuperlinkedQdrantService):
                        try:
                            self.qdrant_service.store_move_embedding(candidate)
                            self._qdrant_embeddings_stored += 1
                            logger.debug(f"Stored move embedding in Qdrant: {clip.clip_id}")
                        except Exception as e:
                            logger.warning(f"Failed to store embedding in Qdrant for {clip.clip_id}: {e}")
                
                return candidate
                
            except Exception as e:
                logger.warning(f"Failed to analyze move {clip.clip_id}: {e}")
                return None
        
        # Submit tasks to thread pool
        future_to_clip = {
            self.executor.submit(analyze_single_move, clip): clip
            for clip in clips
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_clip):
            clip = future_to_clip[future]
            try:
                candidate = future.result()
                if candidate:
                    move_candidates.append(candidate)
                    logger.debug(f"Analyzed move: {clip.move_label}")
            except Exception as e:
                logger.warning(f"Move analysis failed for {clip.clip_id}: {e}")
        
        logger.info(f"Parallel analysis completed: {len(move_candidates)} moves analyzed")
        return move_candidates
    
    async def _analyze_moves_sequential(
        self,
        clips: List,
        music_features: MusicFeatures
    ) -> List[MoveCandidate]:
        """Analyze moves sequentially (fallback or for small datasets)."""
        move_candidates = []
        
        for clip in clips:
            try:
                video_path = Path(self.config.annotation_data_dir) / clip.video_path
                if not video_path.exists():
                    continue
                
                # Step 1: Check Qdrant first for existing embedding
                candidate = None
                if self.qdrant_service and not isinstance(self.qdrant_service, MockSuperlinkedQdrantService):
                    try:
                        qdrant_result = self.qdrant_service.get_move_by_id(clip.clip_id)
                        if qdrant_result:
                            logger.debug(f"Found move in Qdrant: {clip.clip_id}")
                            self._qdrant_embeddings_retrieved += 1
                            
                            # Still need analysis_result, check cache or analyze
                            cache_key = str(video_path)
                            if self.cache:
                                cached_result = self.cache.get("move_analysis", cache_key)
                                if cached_result:
                                    analysis_result, _ = cached_result
                                    self._cache_hits += 1
                                else:
                                    analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                                    self._cache_misses += 1
                            else:
                                analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                                self._cache_misses += 1
                            
                            # Reconstruct multimodal embedding from Qdrant
                            if qdrant_result.embedding is not None:
                                combined_embedding = qdrant_result.embedding
                                audio_embedding = combined_embedding[:128]
                                pose_embedding = combined_embedding[128:512]
                                
                                multimodal_embedding = MultiModalEmbedding(
                                    audio_embedding=audio_embedding,
                                    pose_embedding=pose_embedding
                                )
                            else:
                                multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
                                    music_features, analysis_result
                                )
                            
                            candidate = MoveCandidate(
                                move_id=clip.clip_id,
                                video_path=str(video_path),
                                move_label=clip.move_label,
                                analysis_result=analysis_result,
                                multimodal_embedding=multimodal_embedding,
                                energy_level=clip.energy_level,
                                difficulty=clip.difficulty,
                                estimated_tempo=clip.estimated_tempo,
                                lead_follow_roles=clip.lead_follow_roles
                            )
                    except Exception as e:
                        logger.debug(f"Qdrant lookup failed for {clip.clip_id}: {e}")
                
                # Step 2: If not found in Qdrant, check cache or analyze
                if candidate is None:
                    cache_key = str(video_path)
                    if self.cache:
                        cached_result = self.cache.get("move_analysis", cache_key)
                        if cached_result:
                            analysis_result, multimodal_embedding = cached_result
                            self._cache_hits += 1
                        else:
                            analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                            multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
                                music_features, analysis_result
                            )
                            self.cache.set("move_analysis", cache_key, (analysis_result, multimodal_embedding))
                            self._cache_misses += 1
                    else:
                        analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                        multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
                            music_features, analysis_result
                        )
                        self._cache_misses += 1
                    
                    candidate = MoveCandidate(
                        move_id=clip.clip_id,
                        video_path=str(video_path),
                        move_label=clip.move_label,
                        analysis_result=analysis_result,
                        multimodal_embedding=multimodal_embedding,
                        energy_level=clip.energy_level,
                        difficulty=clip.difficulty,
                        estimated_tempo=clip.estimated_tempo,
                        lead_follow_roles=clip.lead_follow_roles
                    )
                    
                    # Step 3: Store new embedding in Qdrant
                    if self.qdrant_service and not isinstance(self.qdrant_service, MockSuperlinkedQdrantService):
                        try:
                            self.qdrant_service.store_move_embedding(candidate)
                            self._qdrant_embeddings_stored += 1
                            logger.debug(f"Stored move embedding in Qdrant: {clip.clip_id}")
                        except Exception as e:
                            logger.warning(f"Failed to store embedding in Qdrant for {clip.clip_id}: {e}")
                
                move_candidates.append(candidate)
                logger.debug(f"Analyzed move: {clip.move_label}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze move {clip.clip_id}: {e}")
                continue
        
        return move_candidates
    
    def _select_diverse_moves(self, clips: List, max_count: int) -> List:
        """Select diverse moves for analysis to ensure variety."""
        # Group by move category
        categories = {}
        for clip in clips:
            category = clip.move_label
            if category not in categories:
                categories[category] = []
            categories[category].append(clip)
        
        selected = []
        categories_list = list(categories.keys())
        
        # First pass: one from each category
        for category in categories_list:
            if len(selected) < max_count and categories[category]:
                selected.append(categories[category][0])
        
        # Second pass: fill remaining slots with variety
        category_idx = 0
        while len(selected) < max_count and category_idx < len(categories_list) * 3:
            category = categories_list[category_idx % len(categories_list)]
            remaining = [c for c in categories[category] if c not in selected]
            
            if remaining:
                selected.append(remaining[0])
            
            category_idx += 1
        
        return selected[:max_count]
    
    async def populate_qdrant_with_all_moves(self) -> Dict[str, Any]:
        """
        Populate Qdrant collection with all available move embeddings.
        This is useful for initial setup or when rebuilding the collection.
        """
        if not self.qdrant_service or isinstance(self.qdrant_service, MockSuperlinkedQdrantService):
            return {"success": False, "error": "Qdrant service not available"}
        
        try:
            logger.info("Starting batch population of Qdrant with all move embeddings")
            start_time = time.time()
            
            # Load all annotations
            collection = self.annotation_interface.load_annotations("bachata_annotations.json")
            logger.info(f"Found {collection.total_clips} total move clips")
            
            # Create a dummy music features for embedding generation
            # We'll use a representative Bachata song's features
            dummy_audio_path = "data/songs/Aventura.mp3"  # Use existing song if available
            if Path(dummy_audio_path).exists():
                music_features = self.music_analyzer.analyze_audio(dummy_audio_path)
            else:
                # Create minimal dummy features if no song available
                import numpy as np
                music_features = MusicFeatures(
                    tempo=120.0,
                    beat_positions=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
                    duration=30.0,
                    mfcc_features=np.random.random((13, 100)),
                    chroma_features=np.random.random((12, 100)),
                    spectral_centroid=np.random.random(100),
                    rms_energy=np.random.random(100),
                    energy_profile=np.random.random(100),
                    percussive_component=np.random.random(100)
                )
            
            # Analyze all moves and collect candidates
            move_candidates = []
            successful_analyses = 0
            failed_analyses = 0
            
            for clip in collection.clips:
                try:
                    video_path = Path(self.config.annotation_data_dir) / clip.video_path
                    if not video_path.exists():
                        logger.warning(f"Video file not found: {video_path}")
                        failed_analyses += 1
                        continue
                    
                    # Check if already exists in Qdrant
                    existing = self.qdrant_service.get_move_by_id(clip.clip_id)
                    if existing:
                        logger.debug(f"Move {clip.clip_id} already exists in Qdrant, skipping")
                        continue
                    
                    # Analyze move
                    analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                    multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
                        music_features, analysis_result
                    )
                    
                    candidate = MoveCandidate(
                        move_id=clip.clip_id,
                        video_path=str(video_path),
                        move_label=clip.move_label,
                        analysis_result=analysis_result,
                        multimodal_embedding=multimodal_embedding,
                        energy_level=clip.energy_level,
                        difficulty=clip.difficulty,
                        estimated_tempo=clip.estimated_tempo,
                        lead_follow_roles=clip.lead_follow_roles
                    )
                    
                    move_candidates.append(candidate)
                    successful_analyses += 1
                    logger.debug(f"Analyzed move for Qdrant: {clip.move_label}")
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze move {clip.clip_id} for Qdrant: {e}")
                    failed_analyses += 1
            
            # Batch store in Qdrant
            if move_candidates:
                logger.info(f"Batch storing {len(move_candidates)} move embeddings in Qdrant")
                point_ids = self.qdrant_service.batch_store_embeddings(move_candidates)
                logger.info(f"Successfully stored {len(point_ids)} embeddings in Qdrant")
            
            processing_time = time.time() - start_time
            
            # Get final collection info
            collection_info = self.qdrant_service.get_collection_info()
            
            result = {
                "success": True,
                "total_clips": collection.total_clips,
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "embeddings_stored": len(move_candidates),
                "processing_time": processing_time,
                "final_collection_size": collection_info.get("points_count", 0)
            }
            
            logger.info(f"Qdrant population completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to populate Qdrant: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_recommendations_optimized(
        self,
        music_features: MusicFeatures,
        move_candidates: List[MoveCandidate],
        difficulty: str,
        energy_level: Optional[str]
    ) -> Tuple[Optional[List[SuperlinkedRecommendationScore]], float]:
        """Generate recommendations using unified Superlinked vector similarity."""
        start_time = time.time()
        
        try:
            # Generate recommendations using unified vector similarity
            # This replaces the complex multi-factor scoring (audio 40%, tempo 30%, energy 20%, difficulty 10%)
            recommendations = self.recommendation_engine.recommend_moves(
                music_features=music_features,
                target_difficulty=difficulty,
                target_energy=energy_level,
                role_focus="both",  # Default to both roles
                description="",  # Let the engine generate description from music features
                top_k=len(move_candidates) if move_candidates else 40  # Get all available moves
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations using unified Superlinked vectors")
            return recommendations, time.time() - start_time
            
        except Exception as e:
            logger.error(f"Superlinked recommendation generation failed: {e}")
            return None, time.time() - start_time
    
    async def _create_optimized_sequence(
        self,
        recommendations: List[SuperlinkedRecommendationScore],
        music_features: MusicFeatures,
        duration: str = "full"
    ) -> Optional[ChoreographySequence]:
        """Create optimized choreography sequence for the full song duration."""
        try:
            # Always use full song duration
            target_duration = music_features.duration
            
            # Select diverse moves for sequence to fill the entire song
            selected_moves = self._select_sequence_moves_for_full_song(recommendations, target_duration)
            
            # Create sequence that fills the entire song duration
            video_paths = [rec.move_candidate.video_path for rec in selected_moves]
            sequence = self._create_full_song_sequence(video_paths, target_duration)
            
            # Update sequence properties
            sequence.difficulty_level = self._determine_sequence_difficulty(selected_moves)
            sequence.audio_tempo = music_features.tempo
            
            logger.info(f"Created sequence with {len(sequence.moves)} moves, {sequence.total_duration:.1f}s")
            return sequence
            
        except Exception as e:
            logger.error(f"Sequence creation failed: {e}")
            return None
    
    def _select_sequence_moves_for_full_song(
        self, 
        recommendations: List[SuperlinkedRecommendationScore], 
        target_duration: float
    ) -> List[SuperlinkedRecommendationScore]:
        """Select diverse moves to fill the entire song duration."""
        # Calculate number of moves needed for full song
        avg_move_duration = 8.0  # seconds per move
        num_moves_needed = max(8, int(target_duration / avg_move_duration))
        
        logger.info(f"Generating sequence for {target_duration:.1f}s song - need ~{num_moves_needed} moves")
        
        # Group by move type for diversity
        move_groups = {}
        for rec in recommendations:
            move_type = rec.move_candidate.move_label
            if move_type not in move_groups:
                move_groups[move_type] = []
            move_groups[move_type].append(rec)
        
        selected = []
        
        # Cycle through move types to create variety throughout the song
        move_types = list(move_groups.keys())
        type_index = 0
        
        while len(selected) < num_moves_needed:
            # Get next move type in rotation
            current_type = move_types[type_index % len(move_types)]
            
            # Find next unused move from this type
            available_moves = [
                move for move in move_groups[current_type] 
                if move not in selected
            ]
            
            if available_moves:
                selected.append(available_moves[0])
            else:
                # If no more moves of this type, get best remaining overall
                remaining_recs = [rec for rec in recommendations if rec not in selected]
                if remaining_recs:
                    selected.append(remaining_recs[0])
                else:
                    # If we've used all unique moves, start repeating the best ones
                    selected.append(recommendations[len(selected) % len(recommendations)])
            
            type_index += 1
        
        logger.info(f"Selected {len(selected)} moves for full song choreography")
        return selected
    
    def _select_sequence_moves(
        self, 
        recommendations: List[RecommendationScore], 
        target_duration: float
    ) -> List[RecommendationScore]:
        """Legacy method - now delegates to full song method."""
        return self._select_sequence_moves_for_full_song(recommendations, target_duration)
    
    def _determine_sequence_difficulty(self, selected_moves: List[SuperlinkedRecommendationScore]) -> str:
        """Determine overall sequence difficulty."""
        # Convert difficulty scores back to difficulty levels
        difficulties = []
        for move in selected_moves:
            score = move.move_candidate.difficulty_score
            if score <= 1.5:
                difficulties.append("beginner")
            elif score <= 2.5:
                difficulties.append("intermediate")
            else:
                difficulties.append("advanced")
        difficulty_counts = {
            "beginner": difficulties.count("beginner"),
            "intermediate": difficulties.count("intermediate"),
            "advanced": difficulties.count("advanced")
        }
        
        # Return the most common difficulty
        return max(difficulty_counts, key=difficulty_counts.get)
    
    def _create_full_song_sequence(self, video_paths: List[str], target_duration: float) -> ChoreographySequence:
        """Create a sequence that fills the entire song duration by repeating moves as needed."""
        from ..models.video_models import ChoreographySequence, SelectedMove, TransitionType
        
        moves = []
        current_time = 0.0
        move_index = 0
        avg_move_duration = 8.0  # seconds per move
        
        logger.info(f"Creating full song sequence for {target_duration:.1f}s duration")
        
        # Keep adding moves until we reach the target duration
        while current_time < target_duration:
            # Cycle through available moves
            video_path = video_paths[move_index % len(video_paths)]
            
            # Calculate remaining time
            remaining_time = target_duration - current_time
            
            # Use average move duration, but don't exceed remaining time
            move_duration = min(avg_move_duration, remaining_time)
            
            # Don't create moves shorter than 4 seconds
            if move_duration < 4.0:
                break
            
            move = SelectedMove(
                clip_id=f"move_{move_index + 1}",
                video_path=video_path,
                start_time=current_time,
                duration=move_duration,
                transition_type=TransitionType.CUT
            )
            
            moves.append(move)
            current_time += move_duration
            move_index += 1
            
            # Safety check to prevent infinite loops
            if move_index > 100:  # Reasonable limit
                logger.warning("Reached maximum move limit, stopping sequence creation")
                break
        
        sequence = ChoreographySequence(
            moves=moves,
            total_duration=current_time,
            difficulty_level="mixed",
            generation_parameters={
                "sync_type": "full_song_sequence",
                "target_duration": target_duration,
                "actual_duration": current_time,
                "moves_used": len(moves),
                "moves_repeated": move_index > len(video_paths)
            }
        )
        
        logger.info(f"Created full song sequence: {len(moves)} moves, {current_time:.1f}s duration")
        return sequence
    
    async def _generate_video_optimized(
        self,
        sequence: ChoreographySequence,
        music_features: MusicFeatures,
        audio_path: str
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Generate video with optimized settings."""
        start_time = time.time()
        
        try:
            # Generate dynamic output path based on audio file
            song_name = Path(audio_path).stem
            safe_name = "".join(c for c in song_name if c.isalnum() or c in "_-")[:30]
            output_filename = f"{safe_name}_{self.config.quality_mode}_choreography.mp4"
            output_path = Path(self.config.output_dir) / output_filename
            
            # Update video generator config with dynamic output path
            self.video_generator.config.output_path = str(output_path)
            
            # Convert MusicFeatures to dict for video generator
            music_dict = {
                "tempo": music_features.tempo,
                "duration": music_features.duration,
                "beat_positions": music_features.beat_positions
            }
            
            result = self.video_generator.generate_choreography_video(
                sequence=sequence,
                audio_path=audio_path,
                music_features=music_dict
            )
            
            if result.success:
                video_result = {
                    "output_path": result.output_path,
                    "duration": result.duration,
                    "file_size": result.file_size,
                    "processing_time": result.processing_time
                }
                return video_result, time.time() - start_time
            else:
                logger.error(f"Video generation failed: {result.error_message}")
                return None, time.time() - start_time
                
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None, time.time() - start_time
    
    async def _export_metadata(
        self,
        sequence: ChoreographySequence,
        video_result: Dict[str, Any],
        music_features: MusicFeatures,
        audio_path: str
    ) -> Optional[str]:
        """Export comprehensive metadata."""
        try:
            metadata_dir = Path("data/choreography_metadata")
            metadata_dir.mkdir(exist_ok=True)
            
            # Generate filename
            song_name = Path(audio_path).stem
            safe_name = "".join(c for c in song_name if c.isalnum() or c in "_-")[:30]
            metadata_path = metadata_dir / f"{safe_name}_pipeline_metadata.json"
            
            # Create metadata
            metadata = {
                "pipeline_info": {
                    "quality_mode": self.config.quality_mode,
                    "processing_time": video_result["processing_time"],
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses
                },
                "sequence_info": {
                    "total_moves": len(sequence.moves),
                    "total_duration": sequence.total_duration,
                    "difficulty_level": sequence.difficulty_level,
                    "audio_tempo": sequence.audio_tempo
                },
                "music_analysis": {
                    "tempo": music_features.tempo,
                    "duration": music_features.duration,
                    "sections": len(music_features.sections),
                    "rhythm_strength": music_features.rhythm_pattern_strength,
                    "syncopation": music_features.syncopation_level
                },
                "moves_used": [
                    {
                        "clip_id": move.clip_id,
                        "video_path": move.video_path,
                        "start_time": move.start_time,
                        "duration": move.duration
                    }
                    for move in sequence.moves
                ],
                "video_output": video_result
            }
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Metadata exported to: {metadata_path}")
            return str(metadata_path)
            
        except Exception as e:
            logger.error(f"Metadata export failed: {e}")
            return None
    
    async def _cleanup_resources(self) -> None:
        """Clean up resources and temporary files."""
        try:
            # Clear expired cache entries
            if self.cache:
                self.cache.clear_expired()
                
                # Check cache size and clean if too large
                cache_size_mb = self.cache.get_cache_size_mb()
                if cache_size_mb > self.config.max_cache_size_mb:
                    logger.info(f"Cache size ({cache_size_mb:.1f} MB) exceeds limit, clearing old entries")
                    # Additional cleanup logic could be added here
            
            # Clean up temporary files
            temp_dir = Path(self.config.temp_dir)
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*.tmp"):
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass
            
            logger.debug("Resource cleanup completed")
            
        except Exception as e:
            logger.warning(f"Resource cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if self._executor:
            self._executor.shutdown(wait=False)