"""
Data persistence and caching layer for Bachata choreography generation.
Provides lightweight JSON-based caching, metadata persistence, and cleanup utilities.
"""

import json
import pickle
import time
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import threading
from datetime import datetime, timedelta

from .music_analyzer import MusicFeatures
from .move_analyzer import MoveAnalysisResult
from ..models.video_models import ChoreographySequence

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    data: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int = 0
    total_size_mb: float = 0.0
    hit_rate: float = 0.0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    oldest_entry_age_hours: float = 0.0


@dataclass
class ChoreographyMetadata:
    """Metadata for a generated choreography."""
    choreography_id: str
    audio_source: str
    audio_hash: str
    generation_timestamp: str
    duration: float
    difficulty: str
    quality_mode: str
    
    # Music analysis summary
    tempo: float
    sections_count: int
    rhythm_strength: float
    
    # Move analysis summary
    moves_count: int
    move_types: List[str]
    
    # Output information
    output_path: str
    file_size_mb: float
    processing_time: float
    
    # Search tags
    tags: List[str]


class DataPersistenceManager:
    """
    Comprehensive data persistence and caching system.
    Features:
    - Lightweight JSON-based caching for music analysis results
    - Move embedding cache to avoid recomputing MediaPipe features
    - Metadata persistence for generated choreographies with search capabilities
    - Cleanup utilities for managing temporary files and cache size
    - Data export/import functionality for sharing analysis results
    """
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 metadata_dir: str = "data/choreography_metadata",
                 temp_dir: str = "data/temp",
                 max_cache_size_mb: int = 500,
                 default_ttl_hours: int = 24):
        """
        Initialize data persistence manager.
        
        Args:
            cache_dir: Directory for cache files
            metadata_dir: Directory for choreography metadata
            temp_dir: Directory for temporary files
            max_cache_size_mb: Maximum cache size in MB
            default_ttl_hours: Default TTL for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = Path(metadata_dir)
        self.temp_dir = Path(temp_dir)
        self.max_cache_size_mb = max_cache_size_mb
        self.default_ttl_seconds = default_ttl_hours * 3600
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Cache statistics
        self._stats = CacheStats()
        
        # In-memory cache for frequently accessed items
        self._memory_cache: Dict[str, CacheEntry] = {}
        
        logger.info(f"DataPersistenceManager initialized: cache_dir={cache_dir}, max_size={max_cache_size_mb}MB")
    
    def cache_music_analysis(self, audio_path: str, music_features: MusicFeatures) -> str:
        """
        Cache music analysis results with JSON serialization.
        
        Args:
            audio_path: Path to the audio file
            music_features: Music analysis results
            
        Returns:
            Cache key for the stored data
        """
        cache_key = self._generate_cache_key("music_analysis", audio_path)
        
        # Convert MusicFeatures to serializable format
        cache_data = {
            "tempo": music_features.tempo,
            "beat_positions": music_features.beat_positions,
            "duration": music_features.duration,
            "sections": [
                {
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "section_type": s.section_type,
                    "energy_level": s.energy_level,
                    "tempo_stability": s.tempo_stability,
                    "recommended_move_types": s.recommended_move_types
                }
                for s in music_features.sections
            ],
            "rhythm_pattern_strength": music_features.rhythm_pattern_strength,
            "syncopation_level": music_features.syncopation_level,
            "audio_embedding": music_features.audio_embedding,
            "tempo_confidence": music_features.tempo_confidence,
            "energy_profile": music_features.energy_profile,
            # Store numpy arrays as lists for JSON serialization
            "mfcc_features": music_features.mfcc_features.tolist(),
            "chroma_features": music_features.chroma_features.tolist(),
            "spectral_centroid": music_features.spectral_centroid.tolist(),
            "zero_crossing_rate": music_features.zero_crossing_rate.tolist(),
            "rms_energy": music_features.rms_energy.tolist(),
            "harmonic_component": music_features.harmonic_component.tolist(),
            "percussive_component": music_features.percussive_component.tolist()
        }
        
        self._store_cache_entry(cache_key, cache_data, "json")
        logger.info(f"Cached music analysis for {Path(audio_path).name}")
        return cache_key
    
    def load_cached_music_analysis(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Load cached music analysis results.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Cached music analysis data or None if not found
        """
        cache_key = self._generate_cache_key("music_analysis", audio_path)
        return self._load_cache_entry(cache_key, "json")
    
    def cache_move_embedding(self, video_path: str, analysis_result: MoveAnalysisResult) -> str:
        """
        Cache move analysis results and embeddings using pickle for numpy arrays.
        
        Args:
            video_path: Path to the video file
            analysis_result: Move analysis results
            
        Returns:
            Cache key for the stored data
        """
        cache_key = self._generate_cache_key("move_embedding", video_path)
        
        # Store the complete analysis result using pickle for numpy arrays
        cache_data = {
            "pose_embedding": analysis_result.pose_embedding,
            "movement_embedding": analysis_result.movement_embedding,
            "movement_complexity_score": analysis_result.movement_complexity_score,
            "tempo_compatibility_range": analysis_result.tempo_compatibility_range,
            "difficulty_score": analysis_result.difficulty_score,
            "analysis_quality": analysis_result.analysis_quality,
            "pose_detection_rate": analysis_result.pose_detection_rate,
            "duration": analysis_result.duration,
            "frame_count": analysis_result.frame_count,
            "fps": analysis_result.fps
        }
        
        self._store_cache_entry(cache_key, cache_data, "pickle")
        logger.info(f"Cached move embedding for {Path(video_path).name}")
        return cache_key
    
    def load_cached_move_embedding(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Load cached move analysis results.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Cached move analysis data or None if not found
        """
        cache_key = self._generate_cache_key("move_embedding", video_path)
        return self._load_cache_entry(cache_key, "pickle")
    
    def store_choreography_metadata(self, 
                                  choreography_id: str,
                                  sequence: ChoreographySequence,
                                  music_features: MusicFeatures,
                                  audio_source: str,
                                  output_path: str,
                                  processing_time: float,
                                  quality_mode: str = "balanced") -> str:
        """
        Store comprehensive metadata for a generated choreography.
        
        Args:
            choreography_id: Unique identifier for the choreography
            sequence: Generated choreography sequence
            music_features: Music analysis features
            audio_source: Source audio file or URL
            output_path: Path to generated video
            processing_time: Time taken to generate
            quality_mode: Quality mode used for generation
            
        Returns:
            Path to stored metadata file
        """
        # Generate audio hash for deduplication
        audio_hash = self._generate_audio_hash(audio_source)
        
        # Extract move types
        move_types = list(set([
            Path(move.video_path).stem.split('_')[0] 
            for move in sequence.moves
        ]))
        
        # Generate search tags
        tags = self._generate_search_tags(music_features, move_types, quality_mode)
        
        # Get file size
        file_size_mb = 0.0
        if Path(output_path).exists():
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        
        # Create metadata
        metadata = ChoreographyMetadata(
            choreography_id=choreography_id,
            audio_source=audio_source,
            audio_hash=audio_hash,
            generation_timestamp=datetime.now().isoformat(),
            duration=sequence.total_duration,
            difficulty=sequence.difficulty_level,
            quality_mode=quality_mode,
            tempo=music_features.tempo,
            sections_count=len(music_features.sections),
            rhythm_strength=music_features.rhythm_pattern_strength,
            moves_count=len(sequence.moves),
            move_types=move_types,
            output_path=output_path,
            file_size_mb=file_size_mb,
            processing_time=processing_time,
            tags=tags
        )
        
        # Store metadata
        metadata_file = self.metadata_dir / f"{choreography_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        logger.info(f"Stored choreography metadata: {metadata_file}")
        return str(metadata_file)
    
    def search_choreographies(self, 
                            query: Optional[str] = None,
                            tempo_range: Optional[Tuple[float, float]] = None,
                            difficulty: Optional[str] = None,
                            duration_range: Optional[Tuple[float, float]] = None,
                            tags: Optional[List[str]] = None,
                            limit: int = 20) -> List[ChoreographyMetadata]:
        """
        Search choreographies by various criteria.
        
        Args:
            query: Text query to search in audio source and tags
            tempo_range: Tuple of (min_tempo, max_tempo)
            difficulty: Target difficulty level
            duration_range: Tuple of (min_duration, max_duration)
            tags: List of tags to match
            limit: Maximum number of results
            
        Returns:
            List of matching choreography metadata
        """
        results = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    metadata = ChoreographyMetadata(**data)
                
                # Apply filters
                if query and query.lower() not in metadata.audio_source.lower():
                    if not any(query.lower() in tag.lower() for tag in metadata.tags):
                        continue
                
                if tempo_range:
                    if not (tempo_range[0] <= metadata.tempo <= tempo_range[1]):
                        continue
                
                if difficulty and metadata.difficulty != difficulty:
                    continue
                
                if duration_range:
                    if not (duration_range[0] <= metadata.duration <= duration_range[1]):
                        continue
                
                if tags:
                    if not any(tag in metadata.tags for tag in tags):
                        continue
                
                results.append(metadata)
                
                if len(results) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                continue
        
        # Sort by generation timestamp (newest first)
        results.sort(key=lambda x: x.generation_timestamp, reverse=True)
        
        return results
    
    def export_analysis_data(self, export_path: str, include_embeddings: bool = False) -> Dict[str, Any]:
        """
        Export cached analysis data for sharing.
        
        Args:
            export_path: Path to export file
            include_embeddings: Whether to include large embedding data
            
        Returns:
            Export summary
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "music_analyses": {},
            "move_embeddings": {} if include_embeddings else {},
            "choreography_metadata": []
        }
        
        # Export music analyses
        music_cache_dir = self.cache_dir / "music_analysis"
        if music_cache_dir.exists():
            for cache_file in music_cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        export_data["music_analyses"][cache_file.stem] = data
                except Exception as e:
                    logger.warning(f"Error exporting music analysis {cache_file}: {e}")
        
        # Export move embeddings (if requested)
        if include_embeddings:
            move_cache_dir = self.cache_dir / "move_embedding"
            if move_cache_dir.exists():
                for cache_file in move_cache_dir.glob("*.pkl"):
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                            # Convert numpy arrays to lists for JSON serialization
                            serializable_data = self._make_serializable(data)
                            export_data["move_embeddings"][cache_file.stem] = serializable_data
                    except Exception as e:
                        logger.warning(f"Error exporting move embedding {cache_file}: {e}")
        
        # Export choreography metadata
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    export_data["choreography_metadata"].append(data)
            except Exception as e:
                logger.warning(f"Error exporting metadata {metadata_file}: {e}")
        
        # Save export file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        summary = {
            "export_path": export_path,
            "music_analyses_count": len(export_data["music_analyses"]),
            "move_embeddings_count": len(export_data["move_embeddings"]),
            "choreographies_count": len(export_data["choreography_metadata"]),
            "file_size_mb": Path(export_path).stat().st_size / (1024 * 1024)
        }
        
        logger.info(f"Exported analysis data: {summary}")
        return summary
    
    def import_analysis_data(self, import_path: str) -> Dict[str, Any]:
        """
        Import analysis data from export file.
        
        Args:
            import_path: Path to import file
            
        Returns:
            Import summary
        """
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        imported_counts = {
            "music_analyses": 0,
            "move_embeddings": 0,
            "choreographies": 0
        }
        
        # Import music analyses
        for cache_key, data in import_data.get("music_analyses", {}).items():
            cache_file = self.cache_dir / "music_analysis" / f"{cache_key}.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            imported_counts["music_analyses"] += 1
        
        # Import move embeddings
        for cache_key, data in import_data.get("move_embeddings", {}).items():
            cache_file = self.cache_dir / "move_embedding" / f"{cache_key}.pkl"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert lists back to numpy arrays
            numpy_data = self._restore_numpy_arrays(data)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(numpy_data, f)
            imported_counts["move_embeddings"] += 1
        
        # Import choreography metadata
        for metadata in import_data.get("choreography_metadata", []):
            choreography_id = metadata.get("choreography_id", f"imported_{int(time.time())}")
            metadata_file = self.metadata_dir / f"{choreography_id}.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            imported_counts["choreographies"] += 1
        
        logger.info(f"Imported analysis data: {imported_counts}")
        return imported_counts
    
    def cleanup_temporary_files(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age of files to keep
            
        Returns:
            Cleanup summary
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleanup_stats = {
            "files_removed": 0,
            "space_freed_mb": 0.0,
            "errors": []
        }
        
        # Clean temp directory
        for temp_file in self.temp_dir.rglob("*"):
            if temp_file.is_file():
                try:
                    if temp_file.stat().st_mtime < cutoff_time:
                        file_size = temp_file.stat().st_size
                        temp_file.unlink()
                        cleanup_stats["files_removed"] += 1
                        cleanup_stats["space_freed_mb"] += file_size / (1024 * 1024)
                except Exception as e:
                    cleanup_stats["errors"].append(f"Error removing {temp_file}: {e}")
        
        logger.info(f"Temporary file cleanup: {cleanup_stats}")
        return cleanup_stats
    
    def manage_cache_size(self) -> Dict[str, Any]:
        """
        Manage cache size by removing least recently used entries.
        
        Returns:
            Cache management summary
        """
        current_size_mb = self._calculate_cache_size()
        
        if current_size_mb <= self.max_cache_size_mb:
            return {
                "action": "no_cleanup_needed",
                "current_size_mb": current_size_mb,
                "max_size_mb": self.max_cache_size_mb
            }
        
        # Get all cache files with their access times
        cache_files = []
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                cache_files.append((cache_file, cache_file.stat().st_atime))
        
        # Sort by access time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove files until under size limit
        removed_count = 0
        freed_mb = 0.0
        
        for cache_file, _ in cache_files:
            if current_size_mb <= self.max_cache_size_mb:
                break
            
            try:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                removed_count += 1
                freed_mb += file_size / (1024 * 1024)
                current_size_mb -= file_size / (1024 * 1024)
                self._stats.evictions += 1
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {e}")
        
        summary = {
            "action": "cleanup_performed",
            "files_removed": removed_count,
            "space_freed_mb": freed_mb,
            "final_size_mb": current_size_mb,
            "max_size_mb": self.max_cache_size_mb
        }
        
        logger.info(f"Cache size management: {summary}")
        return summary
    
    def get_cache_statistics(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        # Update current statistics
        self._stats.total_entries = len(list(self.cache_dir.rglob("*")))
        self._stats.total_size_mb = self._calculate_cache_size()
        
        if self._stats.hits + self._stats.misses > 0:
            self._stats.hit_rate = self._stats.hits / (self._stats.hits + self._stats.misses)
        
        # Find oldest entry
        oldest_time = time.time()
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                oldest_time = min(oldest_time, cache_file.stat().st_mtime)
        
        self._stats.oldest_entry_age_hours = (time.time() - oldest_time) / 3600
        
        return self._stats
    
    def _generate_cache_key(self, cache_type: str, input_path: str) -> str:
        """Generate cache key from type and input path."""
        # Include file modification time for cache invalidation
        if Path(input_path).exists():
            mtime = Path(input_path).stat().st_mtime
            key_data = f"{cache_type}:{input_path}:{mtime}"
        else:
            key_data = f"{cache_type}:{input_path}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _store_cache_entry(self, cache_key: str, data: Any, format_type: str) -> None:
        """Store cache entry in appropriate format."""
        cache_subdir = self.cache_dir / format_type.replace("_", "/")
        cache_subdir.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            cache_file = cache_subdir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type == "pickle":
            cache_file = cache_subdir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        
        # Update statistics
        self._stats.misses += 1
    
    def _load_cache_entry(self, cache_key: str, format_type: str) -> Optional[Any]:
        """Load cache entry from appropriate format."""
        cache_subdir = self.cache_dir / format_type.replace("_", "/")
        
        if format_type == "json":
            cache_file = cache_subdir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    self._stats.hits += 1
                    return data
                except Exception as e:
                    logger.warning(f"Error loading JSON cache {cache_file}: {e}")
        elif format_type == "pickle":
            cache_file = cache_subdir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    self._stats.hits += 1
                    return data
                except Exception as e:
                    logger.warning(f"Error loading pickle cache {cache_file}: {e}")
        
        self._stats.misses += 1
        return None
    
    def _generate_audio_hash(self, audio_source: str) -> str:
        """Generate hash for audio source."""
        if Path(audio_source).exists():
            # Hash file content for local files
            with open(audio_source, 'rb') as f:
                content = f.read(8192)  # First 8KB for speed
                return hashlib.md5(content).hexdigest()
        else:
            # Hash URL for remote sources
            return hashlib.md5(audio_source.encode()).hexdigest()
    
    def _generate_search_tags(self, music_features: MusicFeatures, move_types: List[str], quality_mode: str) -> List[str]:
        """Generate search tags for choreography."""
        tags = []
        
        # Tempo-based tags
        if music_features.tempo < 100:
            tags.append("slow")
        elif music_features.tempo > 140:
            tags.append("fast")
        else:
            tags.append("medium_tempo")
        
        # Energy-based tags
        avg_energy = sum(music_features.energy_profile) / len(music_features.energy_profile)
        if avg_energy > 0.3:
            tags.append("high_energy")
        elif avg_energy < 0.15:
            tags.append("low_energy")
        else:
            tags.append("medium_energy")
        
        # Move type tags
        tags.extend(move_types)
        
        # Quality tag
        tags.append(f"quality_{quality_mode}")
        
        # Rhythm tags
        if music_features.rhythm_pattern_strength > 0.7:
            tags.append("strong_rhythm")
        if music_features.syncopation_level > 0.6:
            tags.append("syncopated")
        
        return tags
    
    def _calculate_cache_size(self) -> float:
        """Calculate total cache size in MB."""
        total_size = 0
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                try:
                    total_size += cache_file.stat().st_size
                except Exception:
                    pass
        return total_size / (1024 * 1024)
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif hasattr(data, 'tolist'):  # numpy array
            return data.tolist()
        else:
            return data
    
    def _restore_numpy_arrays(self, data: Any) -> Any:
        """Restore numpy arrays from lists (placeholder - would need numpy)."""
        # This would require importing numpy and converting lists back to arrays
        # For now, return as-is
        return data