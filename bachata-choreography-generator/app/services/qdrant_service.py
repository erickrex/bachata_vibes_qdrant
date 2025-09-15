"""
Qdrant vector database integration for faster similarity search.
Provides optimized vector search for multimodal embeddings with metadata filtering.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, MatchValue, Range, SearchRequest
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # Mock classes for when Qdrant is not available
    class QdrantClient:
        pass
    class models:
        pass

from .feature_fusion import MultiModalEmbedding
from .recommendation_engine import MoveCandidate

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "bachata_moves"
    vector_size: int = 512  # 128D audio + 384D pose features
    distance_metric: str = "Cosine"  # Cosine, Dot, Euclid
    
    # Performance settings
    hnsw_config: Dict[str, Any] = None
    quantization_config: Dict[str, Any] = None
    
    # Connection settings
    timeout: float = 30.0
    prefer_grpc: bool = False


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    move_id: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class QdrantStats:
    """Statistics for Qdrant operations."""
    total_points: int = 0
    search_requests: int = 0
    avg_search_time_ms: float = 0.0
    cache_hits: int = 0
    collection_size_mb: float = 0.0


class QdrantEmbeddingService:
    """
    Qdrant-based vector database service for fast similarity search.
    Features:
    - Single collection for 512-dimensional multimodal embeddings
    - Basic metadata filtering for tempo range and difficulty level
    - Batch upload for existing move embeddings
    - Optimized vector search replacing in-memory cosine similarity
    - Performance monitoring and statistics
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialize Qdrant embedding service.
        
        Args:
            config: Qdrant configuration
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not available. Install with: pip install qdrant-client"
            )
        
        self.config = config or QdrantConfig()
        self.client = None
        self.stats = QdrantStats()
        
        # Initialize connection
        self._connect()
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(f"QdrantEmbeddingService initialized: {self.config.host}:{self.config.port}")
    
    def _connect(self) -> None:
        """Establish connection to Qdrant."""
        try:
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc
            )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant: {len(collections.collections)} collections found")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.config.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.config.collection_name}")
                self._create_collection()
            else:
                logger.info(f"Collection {self.config.collection_name} already exists")
                
                # Update stats
                collection_info = self.client.get_collection(self.config.collection_name)
                self.stats.total_points = collection_info.points_count
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def _create_collection(self) -> None:
        """Create the collection with optimized settings."""
        # Default HNSW configuration for good performance
        hnsw_config = self.config.hnsw_config or {
            "m": 16,  # Number of bi-directional links for each node
            "ef_construct": 200,  # Size of the dynamic candidate list
            "full_scan_threshold": 10000  # Use full scan for small collections
        }
        
        # Distance metric mapping
        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID
        }
        
        distance = distance_map.get(self.config.distance_metric, Distance.COSINE)
        
        # Create collection
        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=distance,
                hnsw_config=models.HnswConfigDiff(**hnsw_config) if hnsw_config else None
            )
        )
        
        logger.info(f"Created collection {self.config.collection_name} with {self.config.vector_size}D vectors")
    
    def store_move_embedding(self, 
                           move_candidate: MoveCandidate,
                           embedding: Optional[np.ndarray] = None) -> str:
        """
        Store a move embedding in Qdrant.
        
        Args:
            move_candidate: Move candidate with metadata
            embedding: Optional custom embedding (uses candidate's embedding if not provided)
            
        Returns:
            Point ID in Qdrant
        """
        # Use provided embedding or extract from candidate
        if embedding is not None:
            vector = embedding.tolist()
        else:
            # Combine audio and pose embeddings
            audio_emb = move_candidate.multimodal_embedding.audio_embedding
            pose_emb = move_candidate.multimodal_embedding.pose_embedding
            
            # Ensure consistent dimensionality
            if len(audio_emb) + len(pose_emb) != self.config.vector_size:
                logger.warning(f"Embedding size mismatch: expected {self.config.vector_size}, "
                             f"got {len(audio_emb) + len(pose_emb)}")
                # Pad or truncate as needed
                combined = np.concatenate([audio_emb, pose_emb])
                if len(combined) < self.config.vector_size:
                    combined = np.pad(combined, (0, self.config.vector_size - len(combined)))
                elif len(combined) > self.config.vector_size:
                    combined = combined[:self.config.vector_size]
                vector = combined.tolist()
            else:
                vector = np.concatenate([audio_emb, pose_emb]).tolist()
        
        # Prepare metadata
        metadata = {
            "move_id": move_candidate.move_id,
            "move_label": move_candidate.move_label,
            "video_path": move_candidate.video_path,
            "energy_level": move_candidate.energy_level,
            "difficulty": move_candidate.difficulty,
            "estimated_tempo": move_candidate.estimated_tempo,
            "lead_follow_roles": move_candidate.lead_follow_roles,
            
            # Analysis metrics
            "movement_complexity": move_candidate.analysis_result.movement_complexity_score,
            "difficulty_score": move_candidate.analysis_result.difficulty_score,
            "analysis_quality": move_candidate.analysis_result.analysis_quality,
            "pose_detection_rate": move_candidate.analysis_result.pose_detection_rate,
            "duration": move_candidate.analysis_result.duration,
            
            # Tempo compatibility range
            "tempo_min": move_candidate.analysis_result.tempo_compatibility_range[0],
            "tempo_max": move_candidate.analysis_result.tempo_compatibility_range[1],
            
            # Timestamp
            "created_at": time.time()
        }
        
        # Generate point ID
        point_id = str(uuid.uuid4())
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=metadata
        )
        
        # Store in Qdrant
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[point]
            )
            
            self.stats.total_points += 1
            logger.debug(f"Stored move embedding: {move_candidate.move_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to store move embedding {move_candidate.move_id}: {e}")
            raise
    
    def batch_store_embeddings(self, move_candidates: List[MoveCandidate]) -> List[str]:
        """
        Store multiple move embeddings in batch for better performance.
        
        Args:
            move_candidates: List of move candidates to store
            
        Returns:
            List of point IDs in Qdrant
        """
        points = []
        point_ids = []
        
        for candidate in move_candidates:
            # Combine embeddings
            audio_emb = candidate.multimodal_embedding.audio_embedding
            pose_emb = candidate.multimodal_embedding.pose_embedding
            
            # Ensure consistent dimensionality
            combined = np.concatenate([audio_emb, pose_emb])
            if len(combined) != self.config.vector_size:
                if len(combined) < self.config.vector_size:
                    combined = np.pad(combined, (0, self.config.vector_size - len(combined)))
                else:
                    combined = combined[:self.config.vector_size]
            
            vector = combined.tolist()
            
            # Prepare metadata
            metadata = {
                "move_id": candidate.move_id,
                "move_label": candidate.move_label,
                "video_path": candidate.video_path,
                "energy_level": candidate.energy_level,
                "difficulty": candidate.difficulty,
                "estimated_tempo": candidate.estimated_tempo,
                "lead_follow_roles": candidate.lead_follow_roles,
                "movement_complexity": candidate.analysis_result.movement_complexity_score,
                "difficulty_score": candidate.analysis_result.difficulty_score,
                "analysis_quality": candidate.analysis_result.analysis_quality,
                "pose_detection_rate": candidate.analysis_result.pose_detection_rate,
                "duration": candidate.analysis_result.duration,
                "tempo_min": candidate.analysis_result.tempo_compatibility_range[0],
                "tempo_max": candidate.analysis_result.tempo_compatibility_range[1],
                "created_at": time.time()
            }
            
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=metadata
            ))
        
        # Batch upload
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
            
            self.stats.total_points += len(points)
            logger.info(f"Batch stored {len(points)} move embeddings")
            return point_ids
            
        except Exception as e:
            logger.error(f"Failed to batch store embeddings: {e}")
            raise
    
    def search_similar_moves(self,
                           query_embedding: np.ndarray,
                           limit: int = 10,
                           tempo_range: Optional[Tuple[float, float]] = None,
                           difficulty: Optional[str] = None,
                           energy_level: Optional[str] = None,
                           min_quality: Optional[float] = None) -> List[SearchResult]:
        """
        Search for similar moves using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            tempo_range: Optional tempo range filter (min_tempo, max_tempo)
            difficulty: Optional difficulty level filter
            energy_level: Optional energy level filter
            min_quality: Optional minimum analysis quality filter
            
        Returns:
            List of search results sorted by similarity score
        """
        start_time = time.time()
        
        try:
            # Ensure query embedding has correct dimensions
            if len(query_embedding) != self.config.vector_size:
                if len(query_embedding) < self.config.vector_size:
                    query_embedding = np.pad(query_embedding, (0, self.config.vector_size - len(query_embedding)))
                else:
                    query_embedding = query_embedding[:self.config.vector_size]
            
            # Build filter conditions
            filter_conditions = []
            
            if tempo_range:
                # Move is compatible if its tempo range overlaps with query range
                filter_conditions.extend([
                    FieldCondition(
                        key="tempo_min",
                        range=Range(lte=tempo_range[1])  # Move min <= query max
                    ),
                    FieldCondition(
                        key="tempo_max", 
                        range=Range(gte=tempo_range[0])  # Move max >= query min
                    )
                ])
            
            if difficulty:
                filter_conditions.append(
                    FieldCondition(
                        key="difficulty",
                        match=MatchValue(value=difficulty)
                    )
                )
            
            if energy_level:
                filter_conditions.append(
                    FieldCondition(
                        key="energy_level",
                        match=MatchValue(value=energy_level)
                    )
                )
            
            if min_quality:
                filter_conditions.append(
                    FieldCondition(
                        key="analysis_quality",
                        range=Range(gte=min_quality)
                    )
                )
            
            # Create filter
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                search_result = SearchResult(
                    move_id=result.payload["move_id"],
                    score=result.score,
                    metadata=result.payload
                )
                results.append(search_result)
            
            # Update statistics
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            self.stats.search_requests += 1
            self.stats.avg_search_time_ms = (
                (self.stats.avg_search_time_ms * (self.stats.search_requests - 1) + search_time) /
                self.stats.search_requests
            )
            
            logger.debug(f"Vector search completed: {len(results)} results in {search_time:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    def search_by_music_features(self,
                                music_embedding: np.ndarray,
                                tempo: float,
                                limit: int = 10,
                                tempo_tolerance: float = 10.0,
                                difficulty: Optional[str] = None) -> List[SearchResult]:
        """
        Search for moves compatible with music features.
        
        Args:
            music_embedding: Music embedding vector
            tempo: Music tempo in BPM
            limit: Maximum number of results
            tempo_tolerance: Tempo tolerance in BPM
            difficulty: Optional difficulty filter
            
        Returns:
            List of compatible moves
        """
        # Define tempo range
        tempo_range = (tempo - tempo_tolerance, tempo + tempo_tolerance)
        
        return self.search_similar_moves(
            query_embedding=music_embedding,
            limit=limit,
            tempo_range=tempo_range,
            difficulty=difficulty
        )
    
    def get_move_by_id(self, move_id: str) -> Optional[SearchResult]:
        """
        Get a specific move by its ID.
        
        Args:
            move_id: Move identifier
            
        Returns:
            SearchResult if found, None otherwise
        """
        try:
            # Search by move_id in payload
            search_results = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="move_id",
                            match=MatchValue(value=move_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=True
            )
            
            if search_results[0]:  # Points found
                result = search_results[0][0]  # First point
                return SearchResult(
                    move_id=result.payload["move_id"],
                    score=1.0,  # Exact match
                    metadata=result.payload,
                    embedding=np.array(result.vector) if result.vector else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get move by ID {move_id}: {e}")
            return None
    
    def delete_move(self, move_id: str) -> bool:
        """
        Delete a move from the collection.
        
        Args:
            move_id: Move identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            # Find the point first
            search_results = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="move_id",
                            match=MatchValue(value=move_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            if search_results[0]:  # Points found
                point_id = search_results[0][0].id
                
                # Delete the point
                self.client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=models.PointIdsList(
                        points=[point_id]
                    )
                )
                
                self.stats.total_points -= 1
                logger.info(f"Deleted move: {move_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete move {move_id}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all points from the collection.
        
        Returns:
            True if cleared successfully
        """
        try:
            # Delete all points
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter()  # Empty filter matches all
                )
            )
            
            self.stats.total_points = 0
            logger.info(f"Cleared collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information dictionary
        """
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            
            return {
                "name": self.config.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name,
                "status": collection_info.status.name,
                "optimizer_status": collection_info.optimizer_status.name if collection_info.optimizer_status else "unknown",
                "indexed_vectors_count": collection_info.indexed_vectors_count or 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def get_statistics(self) -> QdrantStats:
        """Get comprehensive statistics."""
        try:
            # Update collection size
            collection_info = self.client.get_collection(self.config.collection_name)
            self.stats.total_points = collection_info.points_count
            
            # Estimate collection size (rough calculation)
            # Each point: vector (4 bytes * vector_size) + metadata (~1KB)
            estimated_size_mb = (
                self.stats.total_points * (4 * self.config.vector_size + 1024)
            ) / (1024 * 1024)
            self.stats.collection_size_mb = estimated_size_mb
            
        except Exception as e:
            logger.warning(f"Failed to update statistics: {e}")
        
        return self.stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Qdrant service.
        
        Returns:
            Health check results
        """
        health_status = {
            "qdrant_available": False,
            "collection_exists": False,
            "can_search": False,
            "can_store": False,
            "error_message": None
        }
        
        try:
            # Check Qdrant connection
            collections = self.client.get_collections()
            health_status["qdrant_available"] = True
            
            # Check collection
            collection_names = [c.name for c in collections.collections]
            health_status["collection_exists"] = self.config.collection_name in collection_names
            
            if health_status["collection_exists"]:
                # Test search
                try:
                    dummy_vector = np.random.random(self.config.vector_size)
                    self.search_similar_moves(dummy_vector, limit=1)
                    health_status["can_search"] = True
                except Exception as e:
                    health_status["error_message"] = f"Search test failed: {e}"
                
                # Test store (create and delete dummy point)
                try:
                    dummy_point = PointStruct(
                        id="health_check_test",
                        vector=dummy_vector.tolist(),
                        payload={"test": True}
                    )
                    
                    self.client.upsert(
                        collection_name=self.config.collection_name,
                        points=[dummy_point]
                    )
                    
                    self.client.delete(
                        collection_name=self.config.collection_name,
                        points_selector=models.PointIdsList(points=["health_check_test"])
                    )
                    
                    health_status["can_store"] = True
                    
                except Exception as e:
                    health_status["error_message"] = f"Store test failed: {e}"
            
        except Exception as e:
            health_status["error_message"] = f"Health check failed: {e}"
        
        return health_status
    
    def migrate_from_memory_cache(self, move_candidates: List[MoveCandidate]) -> Dict[str, Any]:
        """
        Migrate existing move embeddings from in-memory cache to Qdrant.
        
        Args:
            move_candidates: List of move candidates to migrate
            
        Returns:
            Migration summary
        """
        logger.info(f"Starting migration of {len(move_candidates)} move embeddings to Qdrant")
        
        start_time = time.time()
        successful_migrations = 0
        failed_migrations = 0
        
        # Clear existing data
        self.clear_collection()
        
        # Batch store all candidates
        try:
            point_ids = self.batch_store_embeddings(move_candidates)
            successful_migrations = len(point_ids)
            
        except Exception as e:
            logger.error(f"Batch migration failed: {e}")
            
            # Fallback to individual storage
            for candidate in move_candidates:
                try:
                    self.store_move_embedding(candidate)
                    successful_migrations += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate {candidate.move_id}: {e}")
                    failed_migrations += 1
        
        migration_time = time.time() - start_time
        
        summary = {
            "total_candidates": len(move_candidates),
            "successful_migrations": successful_migrations,
            "failed_migrations": failed_migrations,
            "migration_time_seconds": migration_time,
            "final_collection_size": self.stats.total_points
        }
        
        logger.info(f"Migration completed: {summary}")
        return summary


def setup_local_qdrant_docker() -> str:
    """
    Provide instructions for setting up local Qdrant instance using Docker.
    
    Returns:
        Docker command string
    """
    docker_command = """
# Set up local Qdrant instance using Docker
docker run -p 6333:6333 -p 6334:6334 \\
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \\
    qdrant/qdrant

# Or with docker-compose, create docker-compose.yml:
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage:z
    """
    
    return docker_command.strip()


# Fallback implementation when Qdrant is not available
class MockQdrantEmbeddingService:
    """Mock implementation for when Qdrant is not available."""
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        self.config = config or QdrantConfig()
        logger.warning("Qdrant not available, using mock implementation")
    
    def store_move_embedding(self, move_candidate: MoveCandidate, embedding: Optional[np.ndarray] = None) -> str:
        return "mock_id"
    
    def batch_store_embeddings(self, move_candidates: List[MoveCandidate]) -> List[str]:
        return ["mock_id"] * len(move_candidates)
    
    def search_similar_moves(self, query_embedding: np.ndarray, limit: int = 10, **kwargs) -> List[SearchResult]:
        return []
    
    def health_check(self) -> Dict[str, Any]:
        return {"qdrant_available": False, "error_message": "Qdrant client not installed"}


# Factory function to create appropriate service
def create_qdrant_service(config: Optional[QdrantConfig] = None) -> Union[QdrantEmbeddingService, MockQdrantEmbeddingService]:
    """
    Create Qdrant service or mock if not available.
    
    Args:
        config: Qdrant configuration
        
    Returns:
        QdrantEmbeddingService or MockQdrantEmbeddingService
    """
    if QDRANT_AVAILABLE:
        try:
            return QdrantEmbeddingService(config)
        except Exception as e:
            logger.warning(f"Failed to create Qdrant service: {e}, using mock")
            return MockQdrantEmbeddingService(config)
    else:
        return MockQdrantEmbeddingService(config)