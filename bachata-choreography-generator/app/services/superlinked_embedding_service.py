"""
Superlinked-Inspired Embedding Service for Bachata Choreography Generator

This service implements specialized embedding spaces inspired by Superlinked concepts
but using a simpler, more direct approach with sentence-transformers and numpy.
It creates unified vector representations for moves and music that preserve semantic
and numerical relationships through specialized embedding spaces.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingSpace:
    """Base class for different types of embedding spaces."""
    name: str
    dimension: int
    weight: float = 1.0


@dataclass
class TextEmbeddingSpace(EmbeddingSpace):
    """Text similarity space for semantic understanding."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __post_init__(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            logger.warning("sentence-transformers not available, using dummy embeddings")
            self.model = None
            self.dimension = 384  # Default dimension


@dataclass
class NumberEmbeddingSpace(EmbeddingSpace):
    """Number space that preserves linear relationships."""
    min_value: float = 0.0
    max_value: float = 1.0
    
    def __post_init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit([[self.min_value], [self.max_value]])
        self.dimension = 8  # Use 8 dimensions for number encoding


@dataclass
class CategoricalEmbeddingSpace(EmbeddingSpace):
    """Categorical space for discrete values with similarity."""
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        self.encoder = LabelEncoder()
        if self.categories:
            self.encoder.fit(self.categories)
            self.dimension = len(self.categories)
        else:
            self.dimension = 0


@dataclass
class TransitionEmbeddingSpace(EmbeddingSpace):
    """Custom space for transition compatibility patterns."""
    
    def __post_init__(self):
        self.dimension = 64  # Fixed dimension for transition patterns


class SuperlinkedEmbeddingService:
    """
    Service for creating and managing Superlinked-inspired embedding spaces for Bachata moves.
    
    This service creates specialized embedding spaces that preserve different types of
    relationships:
    - TextEmbeddingSpace: For semantic understanding of move descriptions
    - NumberEmbeddingSpace: For tempo (BPM) and difficulty with linear relationships  
    - CategoricalEmbeddingSpace: For energy levels and role focus
    - TransitionEmbeddingSpace: For transition patterns and move compatibility
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.annotations_path = self.data_dir / "bachata_annotations.json"
        
        # Initialize embedding spaces
        self._setup_embedding_spaces()
        
        # Load and process move data
        self.move_data = self._load_move_annotations()
        self.transition_graph = self._build_transition_compatibility_graph()
        
        # Storage for computed embeddings
        self.move_embeddings = {}
        self.move_index = {}
        
        logger.info("SuperlinkedEmbeddingService initialized with specialized embedding spaces")
    
    def _setup_embedding_spaces(self):
        """Create specialized embedding spaces for different data types."""
        
        # 1. TextEmbeddingSpace for semantic move understanding
        self.text_space = TextEmbeddingSpace(
            name="move_semantic_space",
            dimension=384,  # Will be updated after model loading
            weight=0.3
        )
        
        # 2. NumberEmbeddingSpace for tempo (BPM) with preserved linear relationships
        self.tempo_space = NumberEmbeddingSpace(
            name="tempo_space",
            dimension=8,
            min_value=90.0,
            max_value=150.0,
            weight=0.25
        )
        
        # 3. NumberEmbeddingSpace for difficulty with preserved linear relationships
        self.difficulty_space = NumberEmbeddingSpace(
            name="difficulty_space", 
            dimension=8,
            min_value=1.0,
            max_value=3.0,
            weight=0.15
        )
        
        # 4. CategoricalEmbeddingSpace for energy levels
        self.energy_space = CategoricalEmbeddingSpace(
            name="energy_space",
            dimension=3,
            categories=["low", "medium", "high"],
            weight=0.15
        )
        
        # 5. CategoricalEmbeddingSpace for role focus
        self.role_space = CategoricalEmbeddingSpace(
            name="role_space",
            dimension=3,
            categories=["lead_focus", "follow_focus", "both"],
            weight=0.1
        )
        
        # 6. TransitionEmbeddingSpace for transition patterns
        self.transition_space = TransitionEmbeddingSpace(
            name="transition_space",
            dimension=64,
            weight=0.05
        )
        
        # Calculate total embedding dimension
        self.total_dimension = (
            self.text_space.dimension + 
            self.tempo_space.dimension + 
            self.difficulty_space.dimension +
            self.energy_space.dimension + 
            self.role_space.dimension + 
            self.transition_space.dimension
        )
        
        logger.info(f"Created 6 specialized embedding spaces with total dimension: {self.total_dimension}")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using sentence transformer."""
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.text_space.model:
            return self.text_space.model.encode([text])[0]
        else:
            # Fallback: simple hash-based encoding
            hash_val = hash(text)
            vector = np.zeros(384)
            for i in range(min(10, len(text))):
                pos = (hash_val + ord(text[i])) % 384
                vector[pos] = 1.0
            return vector
    
    def _encode_number(self, value: float, space: NumberEmbeddingSpace) -> np.ndarray:
        """Encode number preserving linear relationships."""
        # Normalize the value
        normalized = space.scaler.transform([[value]])[0][0]
        
        # Create a distributed representation
        vector = np.zeros(space.dimension)
        
        # Use multiple basis functions to preserve linear relationships
        for i in range(space.dimension):
            # Sine and cosine basis functions at different frequencies
            freq = (i + 1) * np.pi / space.dimension
            if i % 2 == 0:
                vector[i] = np.sin(freq * normalized)
            else:
                vector[i] = np.cos(freq * normalized)
        
        return vector
    
    def _encode_categorical(self, category: str, space: CategoricalEmbeddingSpace) -> np.ndarray:
        """Encode categorical value with similarity relationships."""
        vector = np.zeros(space.dimension)
        
        try:
            # One-hot encoding with similarity relationships
            idx = space.encoder.transform([category])[0]
            vector[idx] = 1.0
            
            # Add similarity to adjacent categories for ordinal relationships
            if space.name == "energy_space":
                # Energy levels have ordinal relationships: low < medium < high
                if category == "medium":
                    # Medium is similar to both low and high
                    vector[space.encoder.transform(["low"])[0]] = 0.3
                    vector[space.encoder.transform(["high"])[0]] = 0.3
                elif category == "low":
                    # Low is somewhat similar to medium
                    vector[space.encoder.transform(["medium"])[0]] = 0.2
                elif category == "high":
                    # High is somewhat similar to medium
                    vector[space.encoder.transform(["medium"])[0]] = 0.2
                    
        except (ValueError, KeyError):
            # Unknown category, use zero vector
            pass
            
        return vector
    
    def _encode_transition_patterns(self, compatible_moves: List[str]) -> np.ndarray:
        """Encode transition compatibility patterns."""
        vector = np.zeros(self.transition_space.dimension)
        
        # Hash each compatible move to multiple positions for robustness
        for move in compatible_moves:
            if isinstance(move, str):
                # Use multiple hash functions for better distribution
                for seed in [1, 7, 13]:
                    hash_pos = (hash(move + str(seed))) % self.transition_space.dimension
                    vector[hash_pos] = 1.0
        
        # Normalize to unit vector if not zero
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _load_move_annotations(self) -> Dict[str, Any]:
        """Load move annotations from JSON file."""
        try:
            with open(self.annotations_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data.get('clips', []))} move annotations")
            return data
        except FileNotFoundError:
            logger.error(f"Annotations file not found: {self.annotations_path}")
            return {"clips": []}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing annotations JSON: {e}")
            return {"clips": []}
    
    def _build_transition_compatibility_graph(self) -> Dict[str, List[str]]:
        """
        Build a transition compatibility graph based on move characteristics.
        
        This creates a knowledge graph of which moves transition well together
        based on energy levels, difficulty, and move types.
        """
        clips = self.move_data.get("clips", [])
        transition_graph = {}
        
        for clip in clips:
            clip_id = clip["clip_id"]
            move_label = clip["move_label"]
            energy = clip["energy_level"]
            difficulty = clip["difficulty"]
            
            # Find compatible moves based on rules
            compatible_moves = []
            
            for other_clip in clips:
                if other_clip["clip_id"] == clip_id:
                    continue
                
                other_energy = other_clip["energy_level"]
                other_difficulty = other_clip["difficulty"]
                other_label = other_clip["move_label"]
                
                # Compatibility rules
                is_compatible = False
                
                # Same energy level moves are always compatible
                if energy == other_energy:
                    is_compatible = True
                
                # Adjacent energy levels are compatible
                energy_levels = ["low", "medium", "high"]
                if energy in energy_levels and other_energy in energy_levels:
                    energy_diff = abs(energy_levels.index(energy) - energy_levels.index(other_energy))
                    if energy_diff <= 1:
                        is_compatible = True
                
                # Similar difficulty levels are compatible
                difficulty_levels = ["beginner", "intermediate", "advanced"]
                if difficulty in difficulty_levels and other_difficulty in difficulty_levels:
                    diff_diff = abs(difficulty_levels.index(difficulty) - difficulty_levels.index(other_difficulty))
                    if diff_diff <= 1:
                        is_compatible = True
                
                # Specific move type compatibility rules
                if move_label == "basic_step":
                    # Basic steps can transition to anything
                    is_compatible = True
                elif move_label in ["cross_body_lead", "double_cross_body_lead"]:
                    # Cross body leads work well with turns
                    if "turn" in other_label or other_label == "basic_step":
                        is_compatible = True
                elif "turn" in move_label:
                    # Turns work well with basic steps and other turns
                    if other_label == "basic_step" or "turn" in other_label:
                        is_compatible = True
                
                if is_compatible:
                    compatible_moves.append(other_clip["clip_id"])
            
            transition_graph[clip_id] = compatible_moves
        
        logger.info(f"Built transition compatibility graph with {len(transition_graph)} nodes")
        return transition_graph
    
    def _convert_difficulty_to_score(self, difficulty: str) -> float:
        """Convert difficulty string to numerical score."""
        difficulty_map = {
            "beginner": 1.0,
            "intermediate": 2.0,
            "advanced": 3.0
        }
        return difficulty_map.get(difficulty.lower(), 2.0)
    
    def _generate_move_description(self, clip: Dict[str, Any]) -> str:
        """Generate a descriptive text for the move based on its attributes."""
        move_label = clip["move_label"].replace("_", " ")
        energy = clip["energy_level"]
        difficulty = clip["difficulty"]
        notes = clip.get("notes", "")
        
        description = f"{difficulty} {energy} energy {move_label}"
        if notes:
            description += f" - {notes}"
        
        return description
    
    def generate_move_embedding(self, clip: Dict[str, Any]) -> np.ndarray:
        """
        Generate unified embedding for a move by combining all embedding spaces.
        
        Args:
            clip: Move clip data dictionary
            
        Returns:
            Unified embedding vector
        """
        clip_id = clip["clip_id"]
        
        # Generate description and get transition compatibility
        move_description = self._generate_move_description(clip)
        compatible_moves = self.transition_graph.get(clip_id, [])
        
        # Encode each component
        text_embedding = self._encode_text(move_description)
        tempo_embedding = self._encode_number(float(clip["estimated_tempo"]), self.tempo_space)
        difficulty_embedding = self._encode_number(self._convert_difficulty_to_score(clip["difficulty"]), self.difficulty_space)
        energy_embedding = self._encode_categorical(clip["energy_level"], self.energy_space)
        role_embedding = self._encode_categorical(clip["lead_follow_roles"], self.role_space)
        transition_embedding = self._encode_transition_patterns(compatible_moves)
        
        # Combine embeddings with weights
        weighted_embeddings = [
            text_embedding * self.text_space.weight,
            tempo_embedding * self.tempo_space.weight,
            difficulty_embedding * self.difficulty_space.weight,
            energy_embedding * self.energy_space.weight,
            role_embedding * self.role_space.weight,
            transition_embedding * self.transition_space.weight
        ]
        
        # Concatenate all embeddings
        unified_embedding = np.concatenate(weighted_embeddings)
        
        return unified_embedding
    
    def prepare_move_data_for_indexing(self) -> List[Dict[str, Any]]:
        """
        Prepare move data and generate embeddings.
        
        Returns:
            List of dictionaries with embeddings
        """
        clips = self.move_data.get("clips", [])
        prepared_data = []
        
        for clip in clips:
            clip_id = clip["clip_id"]
            
            # Generate unified embedding
            embedding = self.generate_move_embedding(clip)
            
            prepared_clip = {
                "clip_id": clip_id,
                "move_label": clip["move_label"],
                "move_description": self._generate_move_description(clip),
                "tempo": float(clip["estimated_tempo"]),
                "difficulty_score": self._convert_difficulty_to_score(clip["difficulty"]),
                "energy_level": clip["energy_level"],
                "role_focus": clip["lead_follow_roles"],
                "video_path": clip["video_path"],
                "notes": clip.get("notes", ""),
                "embedding": embedding,
                "transition_compatibility": self.transition_graph.get(clip_id, [])
            }
            
            prepared_data.append(prepared_clip)
        
        logger.info(f"Prepared {len(prepared_data)} moves with embeddings")
        return prepared_data
    
    def index_moves(self) -> None:
        """Index all moves into the embedding system."""
        prepared_data = self.prepare_move_data_for_indexing()
        
        # Store embeddings and create index
        embeddings_matrix = []
        for i, move_data in enumerate(prepared_data):
            clip_id = move_data["clip_id"]
            self.move_embeddings[clip_id] = move_data["embedding"]
            self.move_index[clip_id] = i
            embeddings_matrix.append(move_data["embedding"])
        
        # Store as numpy array for efficient similarity computation
        self.embeddings_matrix = np.array(embeddings_matrix)
        self.indexed_moves = prepared_data
        
        logger.info(f"Indexed {len(prepared_data)} moves with {self.total_dimension}D embeddings")
    
    def generate_query_embedding(self, 
                               description: str = "",
                               target_tempo: float = 120.0,
                               difficulty_level: str = "intermediate",
                               energy_level: str = "medium",
                               role_focus: str = "both",
                               weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Generate a query embedding using the same spaces as move embeddings.
        
        Args:
            description: Natural language description
            target_tempo: Target BPM
            difficulty_level: Desired difficulty
            energy_level: Desired energy
            role_focus: Role focus
            weights: Custom weights for different spaces
            
        Returns:
            Query embedding vector
        """
        if weights is None:
            weights = {
                "text": self.text_space.weight,
                "tempo": self.tempo_space.weight,
                "difficulty": self.difficulty_space.weight,
                "energy": self.energy_space.weight,
                "role": self.role_space.weight,
                "transition": self.transition_space.weight
            }
        
        # Encode each component
        text_embedding = self._encode_text(description) if description else np.zeros(self.text_space.dimension)
        tempo_embedding = self._encode_number(target_tempo, self.tempo_space)
        difficulty_embedding = self._encode_number(self._convert_difficulty_to_score(difficulty_level), self.difficulty_space)
        energy_embedding = self._encode_categorical(energy_level, self.energy_space)
        role_embedding = self._encode_categorical(role_focus, self.role_space)
        transition_embedding = np.zeros(self.transition_space.dimension)  # No specific transition for query
        
        # Combine embeddings with weights
        weighted_embeddings = [
            text_embedding * weights["text"],
            tempo_embedding * weights["tempo"],
            difficulty_embedding * weights["difficulty"],
            energy_embedding * weights["energy"],
            role_embedding * weights["role"],
            transition_embedding * weights["transition"]
        ]
        
        # Concatenate all embeddings
        query_embedding = np.concatenate(weighted_embeddings)
        
        return query_embedding
    
    def search_moves(self, 
                    description: str = "",
                    target_tempo: float = 120.0,
                    difficulty_level: str = "intermediate",
                    energy_level: str = "medium",
                    role_focus: str = "both",
                    weights: Optional[Dict[str, float]] = None,
                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for moves using unified embedding similarity.
        
        Args:
            description: Natural language description
            target_tempo: Target BPM
            difficulty_level: Desired difficulty
            energy_level: Desired energy
            role_focus: Role focus
            weights: Custom weights for different spaces
            limit: Maximum number of results
            
        Returns:
            List of matching moves with similarity scores
        """
        if not hasattr(self, 'embeddings_matrix'):
            logger.error("Moves not indexed yet. Call index_moves() first.")
            return []
        
        # Generate query embedding
        query_embedding = self.generate_query_embedding(
            description, target_tempo, difficulty_level, energy_level, role_focus, weights
        )
        
        # Compute cosine similarity with all move embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        # Format results
        results = []
        for idx in top_indices:
            move_data = self.indexed_moves[idx].copy()
            move_data["similarity_score"] = float(similarities[idx])
            # Remove the embedding from the result to keep it clean
            move_data.pop("embedding", None)
            results.append(move_data)
        
        logger.info(f"Search returned {len(results)} results")
        return results
    
    def search_semantic(self, description: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search using only semantic similarity."""
        return self.search_moves(
            description=description,
            weights={"text": 1.0, "tempo": 0.0, "difficulty": 0.0, "energy": 0.0, "role": 0.0, "transition": 0.0},
            limit=limit
        )
    
    def search_tempo(self, target_tempo: float, limit: int = 10) -> List[Dict[str, Any]]:
        """Search using only tempo similarity."""
        return self.search_moves(
            target_tempo=target_tempo,
            weights={"text": 0.0, "tempo": 1.0, "difficulty": 0.0, "energy": 0.0, "role": 0.0, "transition": 0.0},
            limit=limit
        )
    
    def get_move_embedding(self, clip_id: str) -> Optional[np.ndarray]:
        """
        Get the unified embedding vector for a specific move.
        
        Args:
            clip_id: ID of the move clip
            
        Returns:
            Embedding vector or None if not found
        """
        return self.move_embeddings.get(clip_id)
    
    def get_transition_compatibility(self, clip_id: str) -> List[str]:
        """
        Get the list of moves that are compatible for transitions from the given move.
        
        Args:
            clip_id: ID of the source move
            
        Returns:
            List of compatible move IDs
        """
        return self.transition_graph.get(clip_id, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding service."""
        clips = self.move_data.get("clips", [])
        
        # Count moves by category
        move_counts = {}
        energy_counts = {"low": 0, "medium": 0, "high": 0}
        difficulty_counts = {"beginner": 0, "intermediate": 0, "advanced": 0}
        role_counts = {"lead_focus": 0, "follow_focus": 0, "both": 0}
        
        for clip in clips:
            move_label = clip["move_label"]
            move_counts[move_label] = move_counts.get(move_label, 0) + 1
            
            energy_counts[clip["energy_level"]] += 1
            difficulty_counts[clip["difficulty"]] += 1
            role_counts[clip["lead_follow_roles"]] += 1
        
        return {
            "total_moves": len(clips),
            "move_categories": len(move_counts),
            "move_counts": move_counts,
            "energy_distribution": energy_counts,
            "difficulty_distribution": difficulty_counts,
            "role_distribution": role_counts,
            "transition_graph_size": len(self.transition_graph),
            "embedding_spaces": {
                "text_space": "TextSimilaritySpace (semantic understanding)",
                "tempo_space": "NumberSpace (90-150 BPM)",
                "difficulty_space": "NumberSpace (1-3 scale)",
                "energy_space": "CategoricalSimilaritySpace (low/medium/high)",
                "role_space": "CategoricalSimilaritySpace (lead/follow/both)",
                "transition_space": "CustomSpace (compatibility patterns)"
            }
        }


def create_superlinked_service(data_dir: str = "data") -> SuperlinkedEmbeddingService:
    """
    Factory function to create and initialize a SuperlinkedEmbeddingService.
    
    Args:
        data_dir: Directory containing the bachata_annotations.json file
        
    Returns:
        Initialized SuperlinkedEmbeddingService
    """
    service = SuperlinkedEmbeddingService(data_dir)
    service.index_moves()
    return service


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create the service
    service = create_superlinked_service("data")
    
    # Print statistics
    stats = service.get_stats()
    print("\nSuperlinked-Inspired Embedding Service Statistics:")
    print(f"Total moves: {stats['total_moves']}")
    print(f"Move categories: {stats['move_categories']}")
    print(f"Embedding spaces: {len(stats['embedding_spaces'])}")
    print(f"Total embedding dimension: {service.total_dimension}")
    
    # Test semantic search
    results = service.search_semantic("energetic basic steps for beginners")
    
    print(f"\nSemantic search results: {len(results)}")
    for result in results[:3]:
        print(f"- {result['clip_id']}: {result['move_description']} (score: {result['similarity_score']:.3f})")
    
    # Test tempo-based search
    results = service.search_tempo(120.0)
    
    print(f"\nTempo search results (120 BPM): {len(results)}")
    for result in results[:3]:
        print(f"- {result['clip_id']}: {result['tempo']} BPM (score: {result['similarity_score']:.3f})")
    
    # Test multi-factor search
    results = service.search_moves(
        description="smooth intermediate moves",
        target_tempo=115.0,
        difficulty_level="intermediate",
        energy_level="medium",
        role_focus="both"
    )
    
    print(f"\nMulti-factor search results: {len(results)}")
    for result in results[:3]:
        print(f"- {result['clip_id']}: {result['move_description']} (score: {result['similarity_score']:.3f})")
    
    # Test transition compatibility
    print(f"\nTransition compatibility for basic_step_1:")
    compatible = service.get_transition_compatibility("basic_step_1")
    print(f"Compatible moves: {compatible[:5]}...")  # Show first 5