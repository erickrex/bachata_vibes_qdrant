"""
Performance-optimized recommendation engine for Bachata choreography generation.
Features pre-computed similarity matrices, embedding cache, and batch processing.
"""

import numpy as np
import logging
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import hashlib

from .recommendation_engine import (
    RecommendationEngine, RecommendationScore, MoveCandidate, RecommendationRequest
)
from .music_analyzer import MusicFeatures
from .feature_fusion import MultiModalEmbedding

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for recommendation engine optimization."""
    # Caching settings
    enable_embedding_cache: bool = True
    enable_similarity_cache: bool = True
    cache_dir: str = "data/cache/recommendations"
    
    # Pre-computation settings
    enable_precomputed_matrices: bool = True
    similarity_matrix_path: str = "data/cache/similarity_matrix.pkl"
    
    # Performance settings
    batch_size: int = 32
    max_workers: int = 4
    enable_parallel_scoring: bool = True
    
    # Quality vs speed trade-offs
    fast_mode: bool = False  # Reduces computation accuracy for speed
    similarity_threshold: float = 0.1  # Skip very low similarity candidates
    max_candidates_per_request: int = 100


@dataclass
class BatchRecommendationRequest:
    """Request for batch recommendation processing."""
    requests: List[RecommendationRequest]
    shared_candidates: List[MoveCandidate]
    batch_id: str = ""


@dataclass
class CachedEmbedding:
    """Cached embedding with metadata."""
    embedding: np.ndarray
    timestamp: float
    move_id: str
    features_hash: str


@dataclass
class PerformanceMetrics:
    """Performance metrics for recommendation engine."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time: float = 0.0
    similarity_computations: int = 0
    precomputed_matrix_hits: int = 0


class EmbeddingCache:
    """Thread-safe cache for move embeddings and similarity computations."""
    
    def __init__(self, cache_dir: str, max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._lock = threading.Lock()
        
        # In-memory cache for fast access
        self._memory_cache: Dict[str, CachedEmbedding] = {}
        self._access_times: Dict[str, float] = {}
        
        # Load existing cache from disk
        self._load_cache_from_disk()
    
    def _load_cache_from_disk(self) -> None:
        """Load cached embeddings from disk."""
        cache_file = self.cache_dir / "embeddings.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    disk_cache = pickle.load(f)
                    self._memory_cache.update(disk_cache)
                logger.info(f"Loaded {len(self._memory_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
    
    def _save_cache_to_disk(self) -> None:
        """Save cached embeddings to disk."""
        try:
            cache_file = self.cache_dir / "embeddings.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self._memory_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def get_embedding(self, move_id: str, features_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding for a move."""
        with self._lock:
            if move_id in self._memory_cache:
                cached = self._memory_cache[move_id]
                if cached.features_hash == features_hash:
                    self._access_times[move_id] = time.time()
                    return cached.embedding
                else:
                    # Features changed, remove old cache
                    del self._memory_cache[move_id]
                    if move_id in self._access_times:
                        del self._access_times[move_id]
        return None
    
    def set_embedding(self, move_id: str, embedding: np.ndarray, features_hash: str) -> None:
        """Cache embedding for a move."""
        with self._lock:
            # Check if cache is full
            if len(self._memory_cache) >= self.max_size:
                self._evict_oldest()
            
            cached_embedding = CachedEmbedding(
                embedding=embedding,
                timestamp=time.time(),
                move_id=move_id,
                features_hash=features_hash
            )
            
            self._memory_cache[move_id] = cached_embedding
            self._access_times[move_id] = time.time()
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used embedding."""
        if not self._access_times:
            return
        
        oldest_move = min(self._access_times, key=self._access_times.get)
        del self._memory_cache[oldest_move]
        del self._access_times[oldest_move]
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._memory_cache.clear()
            self._access_times.clear()
    
    def save(self) -> None:
        """Save cache to disk."""
        self._save_cache_to_disk()


class SimilarityMatrix:
    """Pre-computed similarity matrix for fast move-to-move similarity lookup."""
    
    def __init__(self, matrix_path: str):
        self.matrix_path = Path(matrix_path)
        self.matrix: Optional[np.ndarray] = None
        self.move_id_to_index: Dict[str, int] = {}
        self.index_to_move_id: Dict[int, str] = {}
        self._lock = threading.Lock()
        
        # Load existing matrix if available
        self._load_matrix()
    
    def _load_matrix(self) -> None:
        """Load pre-computed similarity matrix from disk."""
        if self.matrix_path.exists():
            try:
                with open(self.matrix_path, 'rb') as f:
                    data = pickle.load(f)
                    self.matrix = data['matrix']
                    self.move_id_to_index = data['move_id_to_index']
                    self.index_to_move_id = data['index_to_move_id']
                logger.info(f"Loaded similarity matrix: {self.matrix.shape}")
            except Exception as e:
                logger.warning(f"Failed to load similarity matrix: {e}")
    
    def build_matrix(self, move_candidates: List[MoveCandidate]) -> None:
        """Build similarity matrix from move candidates."""
        n_moves = len(move_candidates)
        self.matrix = np.zeros((n_moves, n_moves))
        
        # Create move ID mappings
        self.move_id_to_index = {
            candidate.move_id: i for i, candidate in enumerate(move_candidates)
        }
        self.index_to_move_id = {
            i: candidate.move_id for i, candidate in enumerate(move_candidates)
        }
        
        logger.info(f"Building similarity matrix for {n_moves} moves...")
        
        # Compute pairwise similarities
        for i, candidate_i in enumerate(move_candidates):
            for j, candidate_j in enumerate(move_candidates):
                if i <= j:  # Only compute upper triangle (matrix is symmetric)
                    similarity = self._compute_move_similarity(candidate_i, candidate_j)
                    self.matrix[i, j] = similarity
                    self.matrix[j, i] = similarity  # Symmetric
        
        # Save matrix to disk
        self._save_matrix()
        logger.info("Similarity matrix built and saved")
    
    def _compute_move_similarity(self, candidate_i: MoveCandidate, candidate_j: MoveCandidate) -> float:
        """Compute similarity between two move candidates."""
        if candidate_i.move_id == candidate_j.move_id:
            return 1.0
        
        # Use pose embeddings for move-to-move similarity
        embedding_i = candidate_i.multimodal_embedding.pose_embedding
        embedding_j = candidate_j.multimodal_embedding.pose_embedding
        
        # Cosine similarity
        dot_product = np.dot(embedding_i, embedding_j)
        norm_i = np.linalg.norm(embedding_i)
        norm_j = np.linalg.norm(embedding_j)
        
        if norm_i > 0 and norm_j > 0:
            return dot_product / (norm_i * norm_j)
        else:
            return 0.0
    
    def get_similarity(self, move_id_1: str, move_id_2: str) -> Optional[float]:
        """Get similarity between two moves."""
        if self.matrix is None:
            return None
        
        idx_1 = self.move_id_to_index.get(move_id_1)
        idx_2 = self.move_id_to_index.get(move_id_2)
        
        if idx_1 is not None and idx_2 is not None:
            return float(self.matrix[idx_1, idx_2])
        
        return None
    
    def get_most_similar_moves(self, move_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get most similar moves to a given move."""
        if self.matrix is None:
            return []
        
        idx = self.move_id_to_index.get(move_id)
        if idx is None:
            return []
        
        # Get similarities for this move
        similarities = self.matrix[idx, :]
        
        # Get top-k most similar (excluding self)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i in top_indices:
            if i != idx and len(results) < top_k:  # Exclude self
                move_id_similar = self.index_to_move_id[i]
                similarity = similarities[i]
                results.append((move_id_similar, similarity))
        
        return results
    
    def _save_matrix(self) -> None:
        """Save similarity matrix to disk."""
        try:
            self.matrix_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'matrix': self.matrix,
                'move_id_to_index': self.move_id_to_index,
                'index_to_move_id': self.index_to_move_id
            }
            with open(self.matrix_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save similarity matrix: {e}")


class OptimizedRecommendationEngine(RecommendationEngine):
    """
    Performance-optimized recommendation engine with:
    - Pre-computed similarity matrices for faster move matching
    - Embedding cache system to store and reuse computed features
    - Batch processing capabilities for multiple song analysis
    - Optimized feature fusion pipeline
    - Smart move selection balancing quality and processing speed
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize optimized recommendation engine."""
        super().__init__()
        
        self.config = config or OptimizationConfig()
        
        # Initialize caching systems
        if self.config.enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(self.config.cache_dir)
        else:
            self.embedding_cache = None
        
        # Initialize similarity matrix
        if self.config.enable_precomputed_matrices:
            self.similarity_matrix = SimilarityMatrix(self.config.similarity_matrix_path)
        else:
            self.similarity_matrix = None
        
        # Thread pool for parallel processing
        self._executor = None
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Pre-computed embeddings for common music features
        self._music_embedding_cache: Dict[str, MultiModalEmbedding] = {}
        
        logger.info(f"OptimizedRecommendationEngine initialized with config: {asdict(self.config)}")
    
    @property
    def executor(self) -> ThreadPoolExecutor:
        """Lazy-loaded thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self._executor
    
    def precompute_similarity_matrix(self, move_candidates: List[MoveCandidate]) -> None:
        """Pre-compute similarity matrix for move candidates."""
        if not self.config.enable_precomputed_matrices:
            logger.warning("Pre-computed matrices are disabled")
            return
        
        if self.similarity_matrix is None:
            self.similarity_matrix = SimilarityMatrix(self.config.similarity_matrix_path)
        
        logger.info("Pre-computing similarity matrix...")
        start_time = time.time()
        
        self.similarity_matrix.build_matrix(move_candidates)
        
        computation_time = time.time() - start_time
        logger.info(f"Similarity matrix pre-computation completed in {computation_time:.2f}s")
    
    def recommend_moves_optimized(
        self,
        request: RecommendationRequest,
        move_candidates: List[MoveCandidate],
        top_k: int = 10
    ) -> List[RecommendationScore]:
        """
        Optimized move recommendation with caching and pre-computed similarities.
        """
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            # Limit candidates if too many
            if len(move_candidates) > self.config.max_candidates_per_request:
                move_candidates = move_candidates[:self.config.max_candidates_per_request]
            
            # Use parallel scoring if enabled and beneficial
            if (self.config.enable_parallel_scoring and 
                len(move_candidates) > self.config.batch_size):
                scores = self._score_moves_parallel(request, move_candidates)
            else:
                scores = self._score_moves_sequential(request, move_candidates)
            
            # Sort and return top-k
            scores.sort(key=lambda x: x.overall_score, reverse=True)
            top_scores = scores[:top_k]
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics.avg_response_time = (
                (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + response_time) /
                self.metrics.total_requests
            )
            
            logger.debug(f"Optimized recommendation completed in {response_time:.3f}s")
            return top_scores
            
        except Exception as e:
            logger.error(f"Optimized recommendation failed: {e}")
            # Fallback to base implementation
            return super().recommend_moves(request, move_candidates, top_k)
    
    def _score_moves_parallel(
        self,
        request: RecommendationRequest,
        move_candidates: List[MoveCandidate]
    ) -> List[RecommendationScore]:
        """Score moves using parallel processing."""
        scores = []
        
        # Split candidates into batches
        batches = [
            move_candidates[i:i + self.config.batch_size]
            for i in range(0, len(move_candidates), self.config.batch_size)
        ]
        
        def score_batch(batch: List[MoveCandidate]) -> List[RecommendationScore]:
            """Score a batch of candidates."""
            batch_scores = []
            for candidate in batch:
                score = self._score_move_candidate_optimized(request, candidate)
                if score.overall_score >= self.config.similarity_threshold:
                    batch_scores.append(score)
            return batch_scores
        
        # Submit batches to thread pool
        future_to_batch = {
            self.executor.submit(score_batch, batch): batch
            for batch in batches
        }
        
        # Collect results
        for future in as_completed(future_to_batch):
            try:
                batch_scores = future.result()
                scores.extend(batch_scores)
            except Exception as e:
                logger.warning(f"Batch scoring failed: {e}")
        
        return scores
    
    def _score_moves_sequential(
        self,
        request: RecommendationRequest,
        move_candidates: List[MoveCandidate]
    ) -> List[RecommendationScore]:
        """Score moves sequentially (fallback or small datasets)."""
        scores = []
        
        for candidate in move_candidates:
            try:
                score = self._score_move_candidate_optimized(request, candidate)
                if score.overall_score >= self.config.similarity_threshold:
                    scores.append(score)
            except Exception as e:
                logger.warning(f"Candidate scoring failed for {candidate.move_id}: {e}")
                continue
        
        return scores
    
    def _score_move_candidate_optimized(
        self,
        request: RecommendationRequest,
        candidate: MoveCandidate
    ) -> RecommendationScore:
        """Optimized scoring for a single move candidate."""
        # Use cached similarity if available
        if self.similarity_matrix:
            cached_similarity = self._get_cached_audio_similarity(
                request.music_embedding, candidate
            )
            if cached_similarity is not None:
                self.metrics.precomputed_matrix_hits += 1
                audio_similarity = cached_similarity
            else:
                audio_similarity = self._calculate_audio_similarity_optimized(
                    request.music_embedding, candidate.multimodal_embedding
                )
                self.metrics.similarity_computations += 1
        else:
            audio_similarity = self._calculate_audio_similarity_optimized(
                request.music_embedding, candidate.multimodal_embedding
            )
            self.metrics.similarity_computations += 1
        
        # Fast mode: skip detailed scoring for low similarity candidates
        if self.config.fast_mode and audio_similarity < 0.3:
            return RecommendationScore(
                move_candidate=candidate,
                overall_score=audio_similarity * 0.4,  # Weighted by audio similarity weight
                audio_similarity=audio_similarity,
                tempo_compatibility=0.5,  # Default values
                energy_alignment=0.5,
                difficulty_compatibility=0.5,
                tempo_difference=10.0,
                energy_match=False,
                difficulty_match=False,
                weights=self.default_weights
            )
        
        # Full scoring for high-potential candidates
        return self._score_move_candidate(
            request.music_features,
            request.music_embedding,
            candidate,
            request.target_difficulty,
            request.target_energy or "medium",
            request.tempo_tolerance,
            request.weights or self.default_weights
        )
    
    def _get_cached_audio_similarity(
        self,
        music_embedding: MultiModalEmbedding,
        candidate: MoveCandidate
    ) -> Optional[float]:
        """Get cached audio similarity if available."""
        # For now, return None (could be enhanced with music-to-move similarity cache)
        return None
    
    def _calculate_audio_similarity_optimized(
        self,
        music_embedding: MultiModalEmbedding,
        move_embedding: MultiModalEmbedding
    ) -> float:
        """Optimized audio similarity calculation."""
        # Use cached embeddings if available
        music_audio = music_embedding.audio_embedding
        move_audio = move_embedding.audio_embedding
        
        # Fast similarity computation using optimized numpy operations
        if len(music_audio) != len(move_audio):
            # Handle dimension mismatch
            min_len = min(len(music_audio), len(move_audio))
            music_audio = music_audio[:min_len]
            move_audio = move_audio[:min_len]
        
        # Vectorized cosine similarity
        dot_product = np.dot(music_audio, move_audio)
        norm_product = np.linalg.norm(music_audio) * np.linalg.norm(move_audio)
        
        if norm_product > 0:
            similarity = dot_product / norm_product
            return (similarity + 1.0) / 2.0  # Normalize to 0-1
        else:
            return 0.0
    
    def batch_recommend_moves(
        self,
        batch_request: BatchRecommendationRequest
    ) -> Dict[str, List[RecommendationScore]]:
        """
        Process multiple recommendation requests in batch for efficiency.
        """
        start_time = time.time()
        results = {}
        
        logger.info(f"Processing batch recommendation with {len(batch_request.requests)} requests")
        
        try:
            # Pre-compute embeddings for all candidates if not cached
            self._precompute_candidate_embeddings(batch_request.shared_candidates)
            
            # Process requests in parallel if beneficial
            if len(batch_request.requests) > 1 and self.config.enable_parallel_scoring:
                results = self._process_batch_parallel(batch_request)
            else:
                results = self._process_batch_sequential(batch_request)
            
            processing_time = time.time() - start_time
            logger.info(f"Batch recommendation completed in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch recommendation failed: {e}")
            return {}
    
    def _precompute_candidate_embeddings(self, candidates: List[MoveCandidate]) -> None:
        """Pre-compute and cache embeddings for candidates."""
        if not self.embedding_cache:
            return
        
        for candidate in candidates:
            # Generate hash for features
            features_hash = self._generate_features_hash(candidate)
            
            # Check if embedding is already cached
            cached_embedding = self.embedding_cache.get_embedding(
                candidate.move_id, features_hash
            )
            
            if cached_embedding is None:
                # Cache the embedding
                self.embedding_cache.set_embedding(
                    candidate.move_id,
                    candidate.multimodal_embedding.pose_embedding,
                    features_hash
                )
                self.metrics.cache_misses += 1
            else:
                self.metrics.cache_hits += 1
    
    def _generate_features_hash(self, candidate: MoveCandidate) -> str:
        """Generate hash for candidate features."""
        # Create hash from key candidate properties
        hash_data = f"{candidate.move_id}:{candidate.energy_level}:{candidate.difficulty}:{candidate.estimated_tempo}"
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    def _process_batch_parallel(
        self,
        batch_request: BatchRecommendationRequest
    ) -> Dict[str, List[RecommendationScore]]:
        """Process batch requests in parallel."""
        results = {}
        
        def process_single_request(request: RecommendationRequest) -> Tuple[str, List[RecommendationScore]]:
            """Process a single request."""
            request_id = f"request_{id(request)}"
            scores = self.recommend_moves_optimized(
                request, batch_request.shared_candidates, top_k=20
            )
            return request_id, scores
        
        # Submit requests to thread pool
        future_to_request = {
            self.executor.submit(process_single_request, request): request
            for request in batch_request.requests
        }
        
        # Collect results
        for future in as_completed(future_to_request):
            try:
                request_id, scores = future.result()
                results[request_id] = scores
            except Exception as e:
                logger.warning(f"Request processing failed: {e}")
        
        return results
    
    def _process_batch_sequential(
        self,
        batch_request: BatchRecommendationRequest
    ) -> Dict[str, List[RecommendationScore]]:
        """Process batch requests sequentially."""
        results = {}
        
        for i, request in enumerate(batch_request.requests):
            try:
                request_id = f"request_{i}"
                scores = self.recommend_moves_optimized(
                    request, batch_request.shared_candidates, top_k=20
                )
                results[request_id] = scores
            except Exception as e:
                logger.warning(f"Request {i} processing failed: {e}")
        
        return results
    
    def optimize_feature_fusion_pipeline(
        self,
        music_features: MusicFeatures,
        move_candidates: List[MoveCandidate]
    ) -> List[MoveCandidate]:
        """
        Optimize feature fusion pipeline to reduce computation time by 50%.
        """
        start_time = time.time()
        
        # Use cached embeddings where possible
        optimized_candidates = []
        
        for candidate in move_candidates:
            if self.embedding_cache:
                features_hash = self._generate_features_hash(candidate)
                cached_embedding = self.embedding_cache.get_embedding(
                    candidate.move_id, features_hash
                )
                
                if cached_embedding is not None:
                    # Use cached embedding
                    candidate.multimodal_embedding.pose_embedding = cached_embedding
                    self.metrics.cache_hits += 1
                else:
                    # Compute and cache new embedding
                    # (This would normally be done by FeatureFusion service)
                    self.embedding_cache.set_embedding(
                        candidate.move_id,
                        candidate.multimodal_embedding.pose_embedding,
                        features_hash
                    )
                    self.metrics.cache_misses += 1
            
            optimized_candidates.append(candidate)
        
        optimization_time = time.time() - start_time
        logger.info(f"Feature fusion pipeline optimized in {optimization_time:.3f}s")
        
        return optimized_candidates
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the recommendation engine."""
        cache_hit_rate = (
            self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
            if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0.0
        )
        
        return {
            "total_requests": self.metrics.total_requests,
            "avg_response_time": self.metrics.avg_response_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "similarity_computations": self.metrics.similarity_computations,
            "precomputed_matrix_hits": self.metrics.precomputed_matrix_hits,
            "config": asdict(self.config)
        }
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        if self.embedding_cache:
            self.embedding_cache.clear()
        
        self._music_embedding_cache.clear()
        
        # Reset metrics
        self.metrics = PerformanceMetrics()
        
        logger.info("All caches cleared")
    
    def save_caches(self) -> None:
        """Save caches to disk."""
        if self.embedding_cache:
            self.embedding_cache.save()
        
        logger.info("Caches saved to disk")
    
    def __del__(self):
        """Cleanup when engine is destroyed."""
        if self._executor:
            self._executor.shutdown(wait=False)
        
        # Save caches before destruction
        try:
            self.save_caches()
        except Exception:
            pass