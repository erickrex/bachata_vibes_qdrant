"""
Qdrant vector database integration for Superlinked embeddings.
Provides optimized vector search for unified Superlinked embeddings with metadata filtering.
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
        Filter, FieldCondition, MatchValue, Range, SearchRequest,
        KeywordIndexParams, IntegerIndexParams, FloatIndexParams
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # Mock classes for when Qdrant is not available
    class QdrantClient:
        pass
    class models:
        pass

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "superlinked_bachata_moves"
    vector_size: int = 512  # Superlinked unified embedding dimension
    distance_metric: str = "Cosine"  # Cosine, Dot, Euclid
    
    # Cloud deployment settings
    api_key: Optional[str] = None
    url: Optional[str] = None  # Full URL for cloud deployment
    
    # Performance settings
    hnsw_config: Dict[str, Any] = None
    quantization_config: Dict[str, Any] = None
    
    # Connection settings
    timeout: float = 30.0
    prefer_grpc: bool = False
    
    @classmethod
    def from_env(cls) -> 'QdrantConfig':
        """Create configuration from environment variables."""
        import os
        
        # Try to load .env file if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # dotenv not available, continue with system env vars
        
        # Check for cloud deployment first
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        # Remove quotes if present
        if qdrant_url and qdrant_url.startswith("'") and qdrant_url.endswith("'"):
            qdrant_url = qdrant_url[1:-1]
        if qdrant_api_key and qdrant_api_key.startswith("'") and qdrant_api_key.endswith("'"):
            qdrant_api_key = qdrant_api_key[1:-1]
        
        if qdrant_url and qdrant_api_key:
            # Cloud deployment
            return cls(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name="superlinked_bachata_moves",
                vector_size=512,
                distance_metric="Cosine",
                timeout=30.0,
                prefer_grpc=False
            )
        else:
            # Local deployment (fallback)
            return cls(
                host=os.getenv('QDRANT_HOST', 'localhost'),
                port=int(os.getenv('QDRANT_PORT', '6333')),
                collection_name="superlinked_bachata_moves",
                vector_size=512,
                distance_metric="Cosine",
                timeout=30.0,
                prefer_grpc=False
            )


@dataclass
class SuperlinkedSearchResult:
    """Result from Superlinked vector similarity search."""
    clip_id: str
    move_label: str
    move_description: str
    tempo: float
    difficulty_score: float
    energy_level: str
    role_focus: str
    video_path: str
    notes: str
    similarity_score: float
    transition_compatibility: List[str]
    embedding: Optional[np.ndarray] = None


@dataclass
class QdrantStats:
    """Statistics for Qdrant operations."""
    total_points: int = 0
    search_requests: int = 0
    avg_search_time_ms: float = 0.0
    cache_hits: int = 0
    collection_size_mb: float = 0.0


class SuperlinkedQdrantService:
    """
    Qdrant-based vector database service for Superlinked unified embeddings.
    Features:
    - Single collection for Superlinked unified embeddings
    - Metadata filtering for tempo, difficulty, energy, and role focus
    - Batch upload for Superlinked move embeddings
    - Optimized vector search with preserved linear relationships
    - Performance monitoring and statistics
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialize Superlinked Qdrant service.
        
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
        
        if self.config.url:
            logger.info(f"SuperlinkedQdrantService initialized: {self.config.url}")
        else:
            logger.info(f"SuperlinkedQdrantService initialized: {self.config.host}:{self.config.port}")
    
    def _connect(self) -> None:
        """Establish connection to Qdrant (cloud or local)."""
        try:
            # Check if using cloud deployment
            if self.config.url and self.config.api_key:
                # Cloud deployment with API key authentication
                self.client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    prefer_grpc=self.config.prefer_grpc
                )
                logger.info(f"Connected to Qdrant Cloud: {self.config.url}")
            else:
                # Local deployment
                self.client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout,
                    prefer_grpc=self.config.prefer_grpc
                )
                logger.info(f"Connected to local Qdrant: {self.config.host}:{self.config.port}")
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Qdrant connection successful: {len(collections.collections)} collections found")
            
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
        """Create the collection with optimized settings and proper indexing."""
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
        
        # Create indexes for metadata fields to enable efficient filtering and retrieval
        try:
            # Index for clip_id (keyword field for exact matching)
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="clip_id",
                field_schema=models.KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )
            logger.info("Created keyword index for clip_id field")
            
            # Index for move_label (keyword field for filtering)
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="move_label",
                field_schema=models.KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )
            logger.info("Created keyword index for move_label field")
            
            # Index for energy_level (keyword field for filtering)
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="energy_level",
                field_schema=models.KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )
            logger.info("Created keyword index for energy_level field")
            
            # Index for role_focus (keyword field for filtering)
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="role_focus",
                field_schema=models.KeywordIndexParams(
                    type="keyword",
                    is_tenant=False
                )
            )
            logger.info("Created keyword index for role_focus field")
            
            # Index for tempo (integer field for range filtering)
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="tempo",
                field_schema=models.IntegerIndexParams(type="integer")
            )
            logger.info("Created integer index for tempo field")
            
            # Index for difficulty_score (float field for range filtering)
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="difficulty_score",
                field_schema=models.FloatIndexParams(type="float")
            )
            logger.info("Created float index for difficulty_score field")
            
        except Exception as e:
            logger.warning(f"Failed to create some indexes (may already exist): {e}")
            # Continue anyway - indexes might already exist
    
    def store_superlinked_move(self, 
                             move_data: Dict[str, Any]) -> str:
        """
        Store a Superlinked move embedding in Qdrant.
        
        Args:
            move_data: Dictionary containing move data with Superlinked embedding
            
        Returns:
            Point ID in Qdrant
        """
        # Extract embedding
        embedding = move_data.get("embedding")
        if embedding is None:
            raise ValueError("Move data must contain 'embedding' field")
        
        # Ensure embedding is the correct size
        if len(embedding) != self.config.vector_size:
            logger.warning(f"Embedding size mismatch: expected {self.config.vector_size}, got {len(embedding)}")
            if len(embedding) < self.config.vector_size:
                embedding = np.pad(embedding, (0, self.config.vector_size - len(embedding)))
            else:
                embedding = embedding[:self.config.vector_size]
        
        vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Prepare metadata (exclude embedding from payload)
        metadata = {
            "clip_id": move_data["clip_id"],
            "move_label": move_data["move_label"],
            "move_description": move_data["move_description"],
            "video_path": move_data["video_path"],
            "tempo": move_data["tempo"],
            "difficulty_score": move_data["difficulty_score"],
            "energy_level": move_data["energy_level"],
            "role_focus": move_data["role_focus"],
            "notes": move_data["notes"],
            "transition_compatibility": move_data["transition_compatibility"],
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
            logger.debug(f"Stored Superlinked move embedding: {move_data['clip_id']}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to store Superlinked move embedding {move_data['clip_id']}: {e}")
            raise
    
    def batch_store_superlinked_moves(self, moves_data: List[Dict[str, Any]]) -> List[str]:
        """
        Store multiple Superlinked move embeddings in batch for better performance.
        
        Args:
            moves_data: List of move data dictionaries with Superlinked embeddings
            
        Returns:
            List of point IDs in Qdrant
        """
        points = []
        point_ids = []
        
        for move_data in moves_data:
            # Extract embedding
            embedding = move_data.get("embedding")
            if embedding is None:
                logger.warning(f"Skipping move {move_data.get('clip_id', 'unknown')} - no embedding")
                continue
            
            # Ensure consistent dimensionality
            if len(embedding) != self.config.vector_size:
                if len(embedding) < self.config.vector_size:
                    embedding = np.pad(embedding, (0, self.config.vector_size - len(embedding)))
                else:
                    embedding = embedding[:self.config.vector_size]
            
            vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Prepare metadata (exclude embedding from payload)
            metadata = {
                "clip_id": move_data["clip_id"],
                "move_label": move_data["move_label"],
                "move_description": move_data["move_description"],
                "video_path": move_data["video_path"],
                "tempo": move_data["tempo"],
                "difficulty_score": move_data["difficulty_score"],
                "energy_level": move_data["energy_level"],
                "role_focus": move_data["role_focus"],
                "notes": move_data["notes"],
                "transition_compatibility": move_data["transition_compatibility"],
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
            logger.info(f"Batch stored {len(points)} Superlinked move embeddings")
            return point_ids
            
        except Exception as e:
            logger.error(f"Failed to batch store Superlinked embeddings: {e}")
            raise
    
    def search_superlinked_moves(self,
                               query_embedding: np.ndarray,
                               limit: int = 10,
                               tempo_range: Optional[Tuple[float, float]] = None,
                               difficulty_range: Optional[Tuple[float, float]] = None,
                               energy_level: Optional[str] = None,
                               role_focus: Optional[str] = None) -> List[SuperlinkedSearchResult]:
        """
        Search for similar moves using Superlinked unified vector similarity.
        
        Args:
            query_embedding: Superlinked query embedding vector
            limit: Maximum number of results
            tempo_range: Optional tempo range filter (min_tempo, max_tempo)
            difficulty_range: Optional difficulty score range filter (min_score, max_score)
            energy_level: Optional energy level filter
            role_focus: Optional role focus filter
            
        Returns:
            List of SuperlinkedSearchResult objects sorted by similarity score
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
                # Filter by tempo range with tolerance
                filter_conditions.append(
                    FieldCondition(
                        key="tempo",
                        range=Range(gte=tempo_range[0], lte=tempo_range[1])
                    )
                )
            
            if difficulty_range:
                # Filter by difficulty score range
                filter_conditions.append(
                    FieldCondition(
                        key="difficulty_score",
                        range=Range(gte=difficulty_range[0], lte=difficulty_range[1])
                    )
                )
            
            if energy_level:
                filter_conditions.append(
                    FieldCondition(
                        key="energy_level",
                        match=MatchValue(value=energy_level)
                    )
                )
            
            if role_focus:
                filter_conditions.append(
                    FieldCondition(
                        key="role_focus",
                        match=MatchValue(value=role_focus)
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
            
            # Convert to SuperlinkedSearchResult objects
            results = []
            for result in search_results:
                payload = result.payload
                search_result = SuperlinkedSearchResult(
                    clip_id=payload["clip_id"],
                    move_label=payload["move_label"],
                    move_description=payload["move_description"],
                    tempo=payload["tempo"],
                    difficulty_score=payload["difficulty_score"],
                    energy_level=payload["energy_level"],
                    role_focus=payload["role_focus"],
                    video_path=payload["video_path"],
                    notes=payload["notes"],
                    similarity_score=result.score,
                    transition_compatibility=payload["transition_compatibility"]
                )
                results.append(search_result)
            
            # Update statistics
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            self.stats.search_requests += 1
            self.stats.avg_search_time_ms = (
                (self.stats.avg_search_time_ms * (self.stats.search_requests - 1) + search_time) /
                self.stats.search_requests
            )
            
            logger.debug(f"Superlinked vector search completed: {len(results)} results in {search_time:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Superlinked vector search failed: {e}")
            raise
    
    def search_by_superlinked_query(self,
                                  query_embedding: np.ndarray,
                                  tempo: float,
                                  difficulty_score: float,
                                  energy_level: str,
                                  role_focus: str,
                                  limit: int = 10,
                                  tempo_tolerance: float = 10.0,
                                  difficulty_tolerance: float = 0.5) -> List[SuperlinkedSearchResult]:
        """
        Search for moves using Superlinked query embedding with preserved linear relationships.
        
        Args:
            query_embedding: Superlinked query embedding vector
            tempo: Target tempo in BPM
            difficulty_score: Target difficulty score (1.0-3.0)
            energy_level: Target energy level
            role_focus: Target role focus
            limit: Maximum number of results
            tempo_tolerance: Tempo tolerance in BPM
            difficulty_tolerance: Difficulty score tolerance
            
        Returns:
            List of compatible moves with preserved linear relationships
        """
        # Define tempo range with preserved linear relationships
        tempo_range = (tempo - tempo_tolerance, tempo + tempo_tolerance)
        
        # Define difficulty range with preserved linear relationships
        difficulty_range = (
            max(1.0, difficulty_score - difficulty_tolerance),
            min(3.0, difficulty_score + difficulty_tolerance)
        )
        
        return self.search_superlinked_moves(
            query_embedding=query_embedding,
            limit=limit,
            tempo_range=tempo_range,
            difficulty_range=difficulty_range,
            energy_level=energy_level,
            role_focus=role_focus
        )
    
    def get_move_by_clip_id(self, clip_id: str) -> Optional[SuperlinkedSearchResult]:
        """
        Get a specific move by its clip ID.
        
        Args:
            clip_id: Move clip identifier
            
        Returns:
            SuperlinkedSearchResult if found, None otherwise
        """
        try:
            # Search by clip_id in payload
            search_results = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="clip_id",
                            match=MatchValue(value=clip_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=True
            )
            
            if search_results[0]:  # Points found
                result = search_results[0][0]  # First point
                payload = result.payload
                return SuperlinkedSearchResult(
                    clip_id=payload["clip_id"],
                    move_label=payload["move_label"],
                    move_description=payload["move_description"],
                    tempo=payload["tempo"],
                    difficulty_score=payload["difficulty_score"],
                    energy_level=payload["energy_level"],
                    role_focus=payload["role_focus"],
                    video_path=payload["video_path"],
                    notes=payload["notes"],
                    similarity_score=1.0,  # Exact match
                    transition_compatibility=payload["transition_compatibility"],
                    embedding=np.array(result.vector) if result.vector else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get move by clip ID {clip_id}: {e}")
            return None
    
    def delete_move(self, clip_id: str) -> bool:
        """
        Delete a move from the collection.
        
        Args:
            clip_id: Move clip identifier
            
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
                            key="clip_id",
                            match=MatchValue(value=clip_id)
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
                logger.info(f"Deleted move: {clip_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete move {clip_id}: {e}")
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
                    self.search_superlinked_moves(dummy_vector, limit=1)
                    health_status["can_search"] = True
                except Exception as e:
                    health_status["error_message"] = f"Search test failed: {e}"
                
                # Test store (create and delete dummy point)
                try:
                    import uuid
                    test_id = str(uuid.uuid4())  # Use UUID for point ID
                    dummy_point = PointStruct(
                        id=test_id,
                        vector=dummy_vector.tolist(),
                        payload={"test": True}
                    )
                    
                    self.client.upsert(
                        collection_name=self.config.collection_name,
                        points=[dummy_point]
                    )
                    
                    self.client.delete(
                        collection_name=self.config.collection_name,
                        points_selector=models.PointIdsList(points=[test_id])
                    )
                    
                    health_status["can_store"] = True
                    
                except Exception as e:
                    health_status["error_message"] = f"Store test failed: {e}"
            
        except Exception as e:
            health_status["error_message"] = f"Health check failed: {e}"
        
        return health_status
    
    def migrate_superlinked_embeddings(self, moves_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Migrate Superlinked move embeddings to Qdrant.
        
        Args:
            moves_data: List of move data dictionaries with Superlinked embeddings
            
        Returns:
            Migration summary
        """
        logger.info(f"Starting migration of {len(moves_data)} Superlinked move embeddings to Qdrant")
        
        start_time = time.time()
        successful_migrations = 0
        failed_migrations = 0
        
        # Clear existing data
        self.clear_collection()
        
        # Batch store all moves
        try:
            point_ids = self.batch_store_superlinked_moves(moves_data)
            successful_migrations = len(point_ids)
            
        except Exception as e:
            logger.error(f"Batch migration failed: {e}")
            
            # Fallback to individual storage
            for move_data in moves_data:
                try:
                    self.store_superlinked_move(move_data)
                    successful_migrations += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate {move_data.get('clip_id', 'unknown')}: {e}")
                    failed_migrations += 1
        
        migration_time = time.time() - start_time
        
        summary = {
            "total_moves": len(moves_data),
            "successful_migrations": successful_migrations,
            "failed_migrations": failed_migrations,
            "migration_time_seconds": migration_time,
            "final_collection_size": self.stats.total_points
        }
        
        logger.info(f"Superlinked migration completed: {summary}")
        return summary


# Docker setup removed - now using Qdrant Cloud deployment


# Fallback implementation when Qdrant is not available
class MockSuperlinkedQdrantService:
    """Mock implementation for when Qdrant is not available."""
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        self.config = config or QdrantConfig()
        logger.warning("Qdrant not available, using mock Superlinked implementation")
    
    def store_superlinked_move(self, move_data: Dict[str, Any]) -> str:
        return "mock_id"
    
    def batch_store_superlinked_moves(self, moves_data: List[Dict[str, Any]]) -> List[str]:
        return ["mock_id"] * len(moves_data)
    
    def search_superlinked_moves(self, query_embedding: np.ndarray, limit: int = 10, **kwargs) -> List[SuperlinkedSearchResult]:
        return []
    
    def search_by_superlinked_query(self, query_embedding: np.ndarray, **kwargs) -> List[SuperlinkedSearchResult]:
        return []
    
    def health_check(self) -> Dict[str, Any]:
        return {"qdrant_available": False, "error_message": "Qdrant client not installed"}
    
    def get_statistics(self) -> QdrantStats:
        return QdrantStats()
    
    def get_collection_info(self) -> Dict[str, Any]:
        return {"points_count": 0}
    
    def migrate_superlinked_embeddings(self, moves_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"error": "Qdrant not available"}
    
    def clear_collection(self) -> bool:
        return True


# Factory function to create appropriate service
def create_superlinked_qdrant_service(config: Optional[QdrantConfig] = None) -> Union[SuperlinkedQdrantService, MockSuperlinkedQdrantService]:
    """
    Create Superlinked Qdrant service or mock if not available.
    
    Args:
        config: Qdrant configuration
        
    Returns:
        SuperlinkedQdrantService or MockSuperlinkedQdrantService
    """
    if QDRANT_AVAILABLE:
        try:
            return SuperlinkedQdrantService(config)
        except Exception as e:
            logger.warning(f"Failed to create Superlinked Qdrant service: {e}, using mock")
            return MockSuperlinkedQdrantService(config)
    else:
        return MockSuperlinkedQdrantService(config)


# Backward compatibility - keep old factory function but redirect to new service
def create_qdrant_service(config: Optional[QdrantConfig] = None) -> Union[SuperlinkedQdrantService, MockSuperlinkedQdrantService]:
    """
    Create Qdrant service (now using Superlinked embeddings).
    
    Args:
        config: Qdrant configuration
        
    Returns:
        SuperlinkedQdrantService or MockSuperlinkedQdrantService
    """
    logger.info("Creating Superlinked-enabled Qdrant service")
    return create_superlinked_qdrant_service(config)