"""
SuperlinkedRecommendationEngine - Unified vector-based recommendation system with Qdrant integration.

This engine replaces the complex multi-factor scoring algorithm with unified Superlinked vectors
that combine all move and music features into a single 512-dimensional embedding space.
It supports natural language queries and dynamic query-time weights for personalized choreography.
Now integrated with Qdrant as the primary vector database.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time
import re
import random

from .superlinked_embedding_service import SuperlinkedEmbeddingService, create_superlinked_service
from .qdrant_service import (
    create_superlinked_qdrant_service, SuperlinkedQdrantService, MockSuperlinkedQdrantService,
    QdrantConfig, SuperlinkedSearchResult
)
from .music_analyzer import MusicFeatures
from .move_analyzer import MoveAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class SuperlinkedMoveCandidate:
    """Simplified move candidate for Superlinked-based recommendations."""
    move_id: str
    video_path: str
    move_label: str
    move_description: str
    tempo: float
    difficulty_score: float
    energy_level: str
    role_focus: str
    notes: str
    embedding: np.ndarray
    transition_compatibility: List[str]


@dataclass
class SuperlinkedRecommendationScore:
    """Unified recommendation score using Superlinked similarity."""
    move_candidate: SuperlinkedMoveCandidate
    similarity_score: float
    explanation: str


@dataclass
class NaturalLanguageQuery:
    """Parsed natural language query for choreography generation."""
    description: str
    target_tempo: Optional[float] = None
    difficulty_level: Optional[str] = None
    energy_level: Optional[str] = None
    role_focus: Optional[str] = None
    move_types: List[str] = None
    weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.move_types is None:
            self.move_types = []


class SuperlinkedRecommendationEngine:
    """
    Unified vector-based recommendation system using Superlinked-inspired embeddings with Qdrant integration.
    
    This engine eliminates the complex multi-factor scoring (audio 40%, tempo 30%, 
    energy 20%, difficulty 10%) and replaces it with a single unified 512-dimensional
    vector similarity search that preserves all relationships through specialized
    embedding spaces. Now uses Qdrant as the primary vector database.
    """
    
    def __init__(self, data_dir: str = "data", qdrant_config: Optional[QdrantConfig] = None):
        """
        Initialize the SuperlinkedRecommendationEngine with Qdrant integration.
        
        Args:
            data_dir: Directory containing move annotations
            qdrant_config: Optional Qdrant configuration
        """
        self.data_dir = data_dir
        
        # Initialize the Superlinked embedding service
        self.embedding_service = create_superlinked_service(data_dir)
        
        # Initialize Qdrant service for vector storage and search
        # Use environment-based configuration if no config provided
        if qdrant_config is None:
            qdrant_config = QdrantConfig.from_env()
        
        self.qdrant_service = create_superlinked_qdrant_service(qdrant_config)
        self.is_qdrant_available = not isinstance(self.qdrant_service, MockSuperlinkedQdrantService)
        
        # Cache for move candidates (fallback when Qdrant is not available)
        self._move_candidates_cache = {}
        self._music_embeddings_cache = {}
        
        # Performance tracking
        self.performance_stats = {
            'unified_searches': 0,
            'avg_search_time_ms': 0.0,
            'natural_language_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'qdrant_searches': 0,
            'fallback_searches': 0
        }
        
        # Auto-populate Qdrant if available and empty
        if self.is_qdrant_available:
            self._auto_populate_qdrant()
        
        logger.info(f"SuperlinkedRecommendationEngine initialized with {self.embedding_service.total_dimension}D unified embeddings and Qdrant integration: {self.is_qdrant_available}")
    
    def _auto_populate_qdrant(self) -> None:
        """Auto-populate Qdrant with Superlinked embeddings if collection is empty."""
        try:
            collection_info = self.qdrant_service.get_collection_info()
            points_count = collection_info.get("points_count", 0)
            
            if points_count == 0:
                logger.info("Qdrant collection is empty - populating with Superlinked embeddings")
                moves_data = self.embedding_service.prepare_move_data_for_indexing()
                migration_summary = self.qdrant_service.migrate_superlinked_embeddings(moves_data)
                logger.info(f"Populated Qdrant with {migration_summary.get('successful_migrations', 0)} Superlinked embeddings")
            else:
                logger.info(f"Qdrant collection already contains {points_count} embeddings")
                
        except Exception as e:
            logger.warning(f"Failed to auto-populate Qdrant: {e}")
    

    def populate_qdrant_from_candidates(self, move_candidates: List[Any]) -> Dict[str, Any]:
        """
        Populate Qdrant with move candidates (compatibility method).
        
        Args:
            move_candidates: List of move candidates (ignored - uses Superlinked data)
            
        Returns:
            Migration summary
        """
        if not self.is_qdrant_available:
            return {"error": "Qdrant not available"}
        
        try:
            moves_data = self.embedding_service.prepare_move_data_for_indexing()
            return self.qdrant_service.migrate_superlinked_embeddings(moves_data)
        except Exception as e:
            return {"error": str(e)}
    
    def recommend_moves(self, 
                       music_features: MusicFeatures,
                       target_difficulty: str = "intermediate",
                       target_energy: Optional[str] = None,
                       role_focus: str = "both",
                       description: str = "",
                       custom_weights: Optional[Dict[str, float]] = None,
                       top_k: int = 10,
                       enable_transition_awareness: bool = True,
                       diversity_factor: float = 0.3,
                       randomization_seed: Optional[int] = None) -> List[SuperlinkedRecommendationScore]:
        """
        Generate move recommendations using unified Superlinked vector similarity with Qdrant.
        
        This method replaces the complex multi-factor scoring with a single vector
        similarity search that naturally handles all factors through the embedding space.
        Now uses Qdrant as the primary vector database for optimized search.
        
        Args:
            music_features: Analyzed music features
            target_difficulty: Desired difficulty level
            target_energy: Desired energy level (auto-detected if None)
            role_focus: Role focus preference
            description: Natural language description
            custom_weights: Custom weights for embedding spaces
            top_k: Number of recommendations to return
            enable_transition_awareness: Enable transition compatibility
            diversity_factor: Factor for diversity injection (0.0 = no diversity, 1.0 = max diversity)
            randomization_seed: Seed for reproducible randomization (None = random)
            
        Returns:
            List of SuperlinkedRecommendationScore objects with diversity
        """
        start_time = time.time()
        
        # Auto-detect energy level if not provided
        if target_energy is None:
            target_energy = self._detect_music_energy_level(music_features)
            logger.info(f"Auto-detected target energy level: {target_energy}")
        
        # Generate enhanced description from music features if not provided
        if not description:
            description = self._generate_music_description(music_features, target_difficulty, target_energy)
        
        # Generate query embedding using Superlinked service
        difficulty_score = self._convert_difficulty_to_score(target_difficulty)
        query_embedding = self.embedding_service.generate_query_embedding(
            description=description,
            target_tempo=music_features.tempo,
            difficulty_level=target_difficulty,
            energy_level=target_energy,
            role_focus=role_focus,
            weights=custom_weights
        )
        
        # Search using Qdrant if available, otherwise fallback to in-memory search
        # Request more results than needed to enable diversity selection
        search_limit = max(top_k * 5, 30)  # Get 5x more results for better diversity selection
        
        if self.is_qdrant_available:
            # Use Qdrant for optimized vector search with preserved linear relationships
            search_results = self.qdrant_service.search_by_superlinked_query(
                query_embedding=query_embedding,
                tempo=music_features.tempo,
                difficulty_score=difficulty_score,
                energy_level=target_energy,
                role_focus=role_focus,
                limit=search_limit,  # Request more results for diversity
                tempo_tolerance=20.0,  # Increased tolerance for more diverse results
                difficulty_tolerance=1.0  # Increased tolerance for more diverse results
            )
            
            # Convert Qdrant results to SuperlinkedRecommendationScore objects
            recommendations = []
            for result in search_results:
                candidate = self._create_move_candidate_from_qdrant_result(result)
                explanation = self._generate_recommendation_explanation_from_qdrant(result, music_features)
                
                score = SuperlinkedRecommendationScore(
                    move_candidate=candidate,
                    similarity_score=result.similarity_score,
                    explanation=explanation
                )
                recommendations.append(score)
            
            self.performance_stats['qdrant_searches'] += 1
            logger.debug(f"Used Qdrant for vector search")
            
        else:
            # Fallback to in-memory search using embedding service
            search_results = self.embedding_service.search_moves(
                description=description,
                target_tempo=music_features.tempo,
                difficulty_level=target_difficulty,
                energy_level=target_energy,
                role_focus=role_focus,
                weights=custom_weights,
                limit=search_limit  # Request more results for diversity
            )
            
            # Convert to SuperlinkedRecommendationScore objects
            recommendations = []
            for result in search_results:
                candidate = self._create_move_candidate_from_result(result)
                explanation = self._generate_recommendation_explanation(result, music_features)
                
                score = SuperlinkedRecommendationScore(
                    move_candidate=candidate,
                    similarity_score=result["similarity_score"],
                    explanation=explanation
                )
                recommendations.append(score)
            
            self.performance_stats['fallback_searches'] += 1
            logger.debug(f"Used fallback in-memory search")
        
        # Apply diversity selection to avoid repetitive choreographies
        if diversity_factor > 0.0 and len(recommendations) > top_k:
            recommendations = self._apply_diversity_selection(
                recommendations, 
                top_k, 
                diversity_factor, 
                randomization_seed
            )
        else:
            # Just take the top results if no diversity requested
            recommendations = recommendations[:top_k]
        
        # Update performance stats
        search_time = (time.time() - start_time) * 1000
        self.performance_stats['unified_searches'] += 1
        self.performance_stats['avg_search_time_ms'] = (
            (self.performance_stats['avg_search_time_ms'] * (self.performance_stats['unified_searches'] - 1) + search_time) /
            self.performance_stats['unified_searches']
        )
        
        logger.info(f"Unified vector search completed in {search_time:.2f}ms: {[f'{s.move_candidate.move_label}={s.similarity_score:.3f}' for s in recommendations[:3]]}")
        
        return recommendations
    
    def _apply_diversity_selection(self, 
                                 recommendations: List[SuperlinkedRecommendationScore],
                                 top_k: int,
                                 diversity_factor: float,
                                 randomization_seed: Optional[int] = None) -> List[SuperlinkedRecommendationScore]:
        """
        Apply diversity selection to avoid repetitive choreographies.
        
        Uses a combination of:
        1. Move category diversity (avoid too many of the same type)
        2. Randomized selection from top candidates
        3. Similarity score preservation
        
        Args:
            recommendations: All available recommendations
            top_k: Number of final recommendations to return
            diversity_factor: How much diversity to inject (0.0-1.0)
            randomization_seed: Seed for reproducible randomization
            
        Returns:
            Diversified list of recommendations
        """
        import random
        from collections import defaultdict
        
        if randomization_seed is not None:
            random.seed(randomization_seed)
        
        if len(recommendations) <= top_k:
            return recommendations
        
        # Strategy 1: Category-based diversity
        category_counts = defaultdict(int)
        selected_recommendations = []
        remaining_recommendations = recommendations.copy()
        
        # First pass: Select diverse categories with high scores
        max_per_category = max(1, top_k // 4)  # Allow max 25% of same category
        
        for rec in recommendations:
            move_category = rec.move_candidate.move_label
            
            if (len(selected_recommendations) < top_k and 
                category_counts[move_category] < max_per_category):
                
                selected_recommendations.append(rec)
                category_counts[move_category] += 1
                remaining_recommendations.remove(rec)
        
        # Second pass: Fill remaining slots with randomized selection
        remaining_slots = top_k - len(selected_recommendations)
        
        if remaining_slots > 0 and remaining_recommendations:
            # Apply diversity factor to determine selection strategy
            if diversity_factor >= 0.7:
                # High diversity: Random selection from top 50%
                top_candidates = remaining_recommendations[:len(remaining_recommendations)//2]
                additional = random.sample(top_candidates, min(remaining_slots, len(top_candidates)))
            elif diversity_factor >= 0.3:
                # Medium diversity: Weighted random selection favoring higher scores
                weights = [1.0 / (i + 1) for i in range(len(remaining_recommendations))]
                additional = random.choices(remaining_recommendations, weights=weights, k=remaining_slots)
            else:
                # Low diversity: Just take the top remaining
                additional = remaining_recommendations[:remaining_slots]
            
            selected_recommendations.extend(additional)
        
        # Sort by similarity score to maintain quality
        selected_recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.debug(f"Applied diversity selection: {len(recommendations)} -> {len(selected_recommendations)} moves")
        logger.debug(f"Category distribution: {dict(category_counts)}")
        
        return selected_recommendations[:top_k]
    
    def search_semantic_moves(self,
                            semantic_query: str,
                            music_features: MusicFeatures,
                            top_k: int = 10,
                            semantic_weight: float = 0.8) -> List[SuperlinkedRecommendationScore]:
        """
        Perform semantic move search using TextSimilaritySpace for move descriptions.
        
        This method emphasizes semantic understanding of move characteristics
        and descriptions, using the TextSimilaritySpace to find moves that
        match the semantic intent of the query.
        
        Args:
            semantic_query: Natural language description of desired moves
            music_features: Music features for context
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0.0-1.0)
            
        Returns:
            List of semantically matching moves
        """
        start_time = time.time()
        
        # Create custom weights that emphasize semantic similarity
        semantic_weights = {
            "text": semantic_weight,
            "tempo": (1.0 - semantic_weight) * 0.4,
            "difficulty": (1.0 - semantic_weight) * 0.2,
            "energy": (1.0 - semantic_weight) * 0.2,
            "role": (1.0 - semantic_weight) * 0.1,
            "transition": (1.0 - semantic_weight) * 0.1
        }
        
        # Generate recommendations with semantic emphasis
        recommendations = self.recommend_moves(
            music_features=music_features,
            description=semantic_query,
            custom_weights=semantic_weights,
            top_k=top_k,
            enable_transition_awareness=False
        )
        
        # Update performance stats
        search_time = (time.time() - start_time) * 1000
        logger.info(f"Semantic search for '{semantic_query}' completed in {search_time:.2f}ms")
        
        return recommendations

    def search_tempo_aware_moves(self,
                               target_tempo: float,
                               music_features: MusicFeatures,
                               tempo_tolerance: float = 10.0,
                               top_k: int = 10) -> List[SuperlinkedRecommendationScore]:
        """
        Perform tempo-aware search that properly handles BPM linear relationships.
        
        This method uses the NumberSpace for tempo to ensure that moves with
        tempos closer to the target (e.g., 125 BPM closer to 130 than 90) are
        ranked higher, preserving linear relationships.
        
        Args:
            target_tempo: Target BPM for the search
            music_features: Music features (tempo will be overridden)
            tempo_tolerance: BPM tolerance for filtering
            top_k: Number of results to return
            
        Returns:
            List of tempo-compatible moves
        """
        start_time = time.time()
        
        # Create custom weights that emphasize tempo matching
        tempo_weights = {
            "text": 0.1,
            "tempo": 0.7,  # High weight for tempo matching
            "difficulty": 0.1,
            "energy": 0.05,
            "role": 0.05,
            "transition": 0.0
        }
        
        # Override music features tempo for search
        modified_music_features = MusicFeatures(
            tempo=target_tempo,
            beat_positions=music_features.beat_positions,
            duration=music_features.duration,
            mfcc_features=music_features.mfcc_features,
            chroma_features=music_features.chroma_features,
            spectral_centroid=music_features.spectral_centroid,
            zero_crossing_rate=music_features.zero_crossing_rate,
            rms_energy=music_features.rms_energy,
            harmonic_component=music_features.harmonic_component,
            percussive_component=music_features.percussive_component,
            energy_profile=music_features.energy_profile,
            tempo_confidence=music_features.tempo_confidence,
            sections=music_features.sections,
            rhythm_pattern_strength=music_features.rhythm_pattern_strength,
            syncopation_level=music_features.syncopation_level,
            audio_embedding=music_features.audio_embedding
        )
        
        # Generate recommendations with tempo emphasis
        recommendations = self.recommend_moves(
            music_features=modified_music_features,
            description=f"moves compatible with {target_tempo} BPM",
            custom_weights=tempo_weights,
            top_k=top_k * 2,  # Get more results for filtering
            enable_transition_awareness=False
        )
        
        # Filter by tempo tolerance and re-rank
        filtered_recommendations = []
        for rec in recommendations:
            move_tempo = rec.move_candidate.tempo
            tempo_diff = abs(move_tempo - target_tempo)
            
            if tempo_diff <= tempo_tolerance:
                # Adjust score based on tempo proximity (linear relationship)
                tempo_proximity = 1.0 - (tempo_diff / tempo_tolerance)
                rec.similarity_score = rec.similarity_score * 0.3 + tempo_proximity * 0.7
                filtered_recommendations.append(rec)
        
        # Sort by adjusted score and take top_k
        filtered_recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
        final_recommendations = filtered_recommendations[:top_k]
        
        # Update performance stats
        search_time = (time.time() - start_time) * 1000
        logger.info(f"Tempo-aware search for {target_tempo} BPM (±{tempo_tolerance}) completed in {search_time:.2f}ms")
        
        return final_recommendations

    def search_categorical_filtered_moves(self,
                                        music_features: MusicFeatures,
                                        energy_level: str,
                                        role_focus: str,
                                        difficulty_level: Optional[str] = None,
                                        top_k: int = 10) -> List[SuperlinkedRecommendationScore]:
        """
        Search with categorical filtering for energy level and role focus relationships.
        
        This method uses CategoricalSimilaritySpace to understand relationships
        between energy levels (low/medium/high) and role focus (lead/follow/both).
        
        Args:
            music_features: Music features for context
            energy_level: Required energy level (low/medium/high)
            role_focus: Required role focus (lead_focus/follow_focus/both)
            difficulty_level: Optional difficulty filter
            top_k: Number of results to return
            
        Returns:
            List of categorically filtered moves
        """
        start_time = time.time()
        
        # Create custom weights that emphasize categorical matching
        categorical_weights = {
            "text": 0.2,
            "tempo": 0.3,
            "difficulty": 0.2 if difficulty_level else 0.1,
            "energy": 0.25,  # High weight for energy matching
            "role": 0.15,    # High weight for role matching
            "transition": 0.1
        }
        
        # Generate recommendations with categorical emphasis
        recommendations = self.recommend_moves(
            music_features=music_features,
            target_difficulty=difficulty_level or "intermediate",
            target_energy=energy_level,
            role_focus=role_focus,
            description=f"{energy_level} energy moves for {role_focus}",
            custom_weights=categorical_weights,
            top_k=top_k * 2,  # Get more results for strict filtering
            enable_transition_awareness=False
        )
        
        # Apply strict categorical filtering
        filtered_recommendations = []
        for rec in recommendations:
            candidate = rec.move_candidate
            
            # Energy level matching (exact or adjacent)
            energy_match = self._calculate_energy_compatibility(energy_level, candidate.energy_level)
            
            # Role focus matching
            role_match = self._calculate_role_compatibility(role_focus, candidate.role_focus)
            
            # Difficulty matching (if specified)
            difficulty_match = 1.0
            if difficulty_level:
                difficulty_match = self._calculate_difficulty_compatibility(difficulty_level, candidate.difficulty_score)
            
            # Combined categorical score
            categorical_score = (energy_match * 0.4 + role_match * 0.4 + difficulty_match * 0.2)
            
            # Only include moves with good categorical match
            if categorical_score >= 0.6:
                # Adjust similarity score based on categorical compatibility
                rec.similarity_score = rec.similarity_score * 0.5 + categorical_score * 0.5
                filtered_recommendations.append(rec)
        
        # Sort by adjusted score and take top_k
        filtered_recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
        final_recommendations = filtered_recommendations[:top_k]
        
        # Update performance stats
        search_time = (time.time() - start_time) * 1000
        logger.info(f"Categorical search for {energy_level}/{role_focus} completed in {search_time:.2f}ms")
        
        return final_recommendations

    def recommend_with_natural_language(self, 
                                      query: str,
                                      music_features: MusicFeatures,
                                      top_k: int = 10) -> List[SuperlinkedRecommendationScore]:
        """
        Generate recommendations using natural language queries.
        
        Examples:
        - "energetic intermediate moves for 125 BPM song"
        - "smooth beginner basic steps with low energy"
        - "advanced turns and styling for fast bachata"
        
        Args:
            query: Natural language query
            music_features: Analyzed music features
            top_k: Number of recommendations to return
            
        Returns:
            List of SuperlinkedRecommendationScore objects
        """
        start_time = time.time()
        
        # Parse natural language query
        parsed_query = self._parse_natural_language_query(query, music_features)
        
        # Generate recommendations using parsed parameters
        recommendations = self.recommend_moves(
            music_features=music_features,
            target_difficulty=parsed_query.difficulty_level or "intermediate",
            target_energy=parsed_query.energy_level,
            role_focus=parsed_query.role_focus or "both",
            description=parsed_query.description,
            custom_weights=parsed_query.weights,
            top_k=top_k
        )
        
        # Update performance stats
        self.performance_stats['natural_language_queries'] += 1
        
        search_time = (time.time() - start_time) * 1000
        logger.info(f"Natural language query '{query}' processed in {search_time:.2f}ms")
        
        return recommendations
    
    def _parse_natural_language_query(self, query: str, music_features: MusicFeatures) -> NaturalLanguageQuery:
        """
        Parse natural language query into structured parameters.
        
        Args:
            query: Natural language query string
            music_features: Music features for context
            
        Returns:
            Parsed NaturalLanguageQuery object
        """
        query_lower = query.lower()
        
        # Extract difficulty level
        difficulty_level = None
        if "beginner" in query_lower or "basic" in query_lower or "easy" in query_lower:
            difficulty_level = "beginner"
        elif "advanced" in query_lower or "complex" in query_lower or "difficult" in query_lower:
            difficulty_level = "advanced"
        elif "intermediate" in query_lower or "medium" in query_lower:
            difficulty_level = "intermediate"
        
        # Extract energy level
        energy_level = None
        if "energetic" in query_lower or "high energy" in query_lower or "fast" in query_lower or "dynamic" in query_lower:
            energy_level = "high"
        elif "calm" in query_lower or "low energy" in query_lower or "slow" in query_lower or "gentle" in query_lower:
            energy_level = "low"
        elif "medium" in query_lower or "moderate" in query_lower or "balanced" in query_lower:
            energy_level = "medium"
        elif "smooth" in query_lower or "flowing" in query_lower:
            energy_level = "medium"
        
        # Extract role focus
        role_focus = "both"
        if "lead" in query_lower and "follow" not in query_lower:
            role_focus = "lead_focus"
        elif "follow" in query_lower and "lead" not in query_lower:
            role_focus = "follow_focus"
        
        # Extract move types
        move_types = []
        move_keywords = {
            "basic": ["basic", "step", "foundation"],
            "turn": ["turn", "spin", "rotation"],
            "cross_body": ["cross", "body", "lead"],
            "styling": ["styling", "arm", "hand"],
            "dip": ["dip", "lean"],
            "footwork": ["footwork", "feet", "step"]
        }
        
        for move_type, keywords in move_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                move_types.append(move_type)
        
        # Extract tempo if mentioned
        target_tempo = None
        tempo_match = re.search(r'(\d+)\s*bpm', query_lower)
        if tempo_match:
            target_tempo = float(tempo_match.group(1))
        else:
            # Use music tempo as default
            target_tempo = music_features.tempo
        
        # Determine custom weights based on query emphasis
        weights = None
        if "musical" in query_lower or "rhythm" in query_lower or "beat" in query_lower:
            # Emphasize musical compatibility
            weights = {"text": 0.4, "tempo": 0.4, "difficulty": 0.1, "energy": 0.1, "role": 0.0, "transition": 0.0}
        elif "tempo" in query_lower or "bpm" in query_lower or "speed" in query_lower:
            # Emphasize tempo matching
            weights = {"text": 0.2, "tempo": 0.5, "difficulty": 0.1, "energy": 0.2, "role": 0.0, "transition": 0.0}
        elif "flow" in query_lower or "transition" in query_lower or "sequence" in query_lower:
            # Emphasize transitions
            weights = {"text": 0.3, "tempo": 0.2, "difficulty": 0.1, "energy": 0.2, "role": 0.0, "transition": 0.2}
        
        return NaturalLanguageQuery(
            description=query,
            target_tempo=target_tempo,
            difficulty_level=difficulty_level,
            energy_level=energy_level,
            role_focus=role_focus,
            move_types=move_types,
            weights=weights
        )
    
    def _detect_music_energy_level(self, music_features: MusicFeatures) -> str:
        """Auto-detect energy level from music features."""
        # Use multiple indicators for energy detection
        
        # 1. RMS energy statistics
        avg_rms = np.mean(music_features.rms_energy)
        
        # 2. Tempo (higher tempo often indicates higher energy)
        tempo_factor = music_features.tempo / 130.0  # Normalize around typical Bachata tempo
        
        # 3. Spectral centroid (brightness indicator)
        avg_spectral_centroid = np.mean(music_features.spectral_centroid)
        brightness_factor = avg_spectral_centroid / 2000.0  # Normalize
        
        # 4. Percussive component strength
        percussive_strength = np.mean(np.abs(music_features.percussive_component))
        
        # 5. Energy profile variance (dynamic range)
        energy_variance = np.var(music_features.energy_profile)
        
        # Combine indicators
        energy_score = (
            0.3 * min(1.0, avg_rms * 10) +  # RMS energy
            0.2 * min(1.0, tempo_factor) +   # Tempo factor
            0.2 * min(1.0, brightness_factor) +  # Brightness
            0.2 * min(1.0, percussive_strength * 5) +  # Percussion
            0.1 * min(1.0, energy_variance * 100)  # Dynamics
        )
        
        # Map to energy levels
        if energy_score < 0.35:
            return "low"
        elif energy_score < 0.65:
            return "medium"
        else:
            return "high"
    
    def _generate_music_description(self, 
                                  music_features: MusicFeatures, 
                                  difficulty: str, 
                                  energy: str) -> str:
        """Generate a descriptive text for the music based on its features."""
        tempo_desc = "slow" if music_features.tempo < 110 else "fast" if music_features.tempo > 130 else "medium tempo"
        
        # Analyze musical characteristics
        avg_spectral_centroid = np.mean(music_features.spectral_centroid)
        brightness = "bright" if avg_spectral_centroid > 2000 else "warm"
        
        percussive_strength = np.mean(np.abs(music_features.percussive_component))
        rhythm_desc = "strong rhythm" if percussive_strength > 0.1 else "gentle rhythm"
        
        description = f"{energy} energy {difficulty} bachata with {tempo_desc} {brightness} sound and {rhythm_desc}"
        
        return description
    
    def _create_move_candidate_from_result(self, result: Dict[str, Any]) -> SuperlinkedMoveCandidate:
        """Create SuperlinkedMoveCandidate from search result."""
        return SuperlinkedMoveCandidate(
            move_id=result["clip_id"],
            video_path=result["video_path"],
            move_label=result["move_label"],
            move_description=result["move_description"],
            tempo=result["tempo"],
            difficulty_score=result["difficulty_score"],
            energy_level=result["energy_level"],
            role_focus=result["role_focus"],
            notes=result["notes"],
            embedding=np.array([]),  # Not needed for results
            transition_compatibility=result["transition_compatibility"]
        )
    
    def _generate_recommendation_explanation(self, 
                                           result: Dict[str, Any], 
                                           music_features: MusicFeatures) -> str:
        """Generate human-readable explanation for the recommendation."""
        similarity = result["similarity_score"]
        tempo_diff = abs(result["tempo"] - music_features.tempo)
        
        # Generate explanation based on similarity score and characteristics
        if similarity > 0.8:
            quality = "Excellent"
        elif similarity > 0.6:
            quality = "Good"
        elif similarity > 0.4:
            quality = "Moderate"
        else:
            quality = "Basic"
        
        tempo_desc = ""
        if tempo_diff <= 5:
            tempo_desc = "perfect tempo match"
        elif tempo_diff <= 10:
            tempo_desc = f"close tempo (±{tempo_diff:.0f} BPM)"
        else:
            tempo_desc = f"tempo adaptation needed (±{tempo_diff:.0f} BPM)"
        
        explanation = f"{quality} match with {tempo_desc}, {result['energy_level']} energy {result['move_label']}"
        
        if result["notes"]:
            explanation += f" - {result['notes']}"
        
        return explanation
    
    def get_move_transitions(self, move_id: str) -> List[str]:
        """Get compatible moves for transitions."""
        return self.embedding_service.get_transition_compatibility(move_id)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the recommendation engine."""
        embedding_stats = self.embedding_service.get_stats()
        
        return {
            "engine_type": "SuperlinkedRecommendationEngine",
            "unified_vector_dimension": self.embedding_service.total_dimension,
            "embedding_spaces": len(embedding_stats["embedding_spaces"]),
            "total_moves": embedding_stats["total_moves"],
            "performance": self.performance_stats,
            "embedding_service_stats": embedding_stats
        }
    
    def create_personalized_weights(self, 
                                  user_preferences: Dict[str, float]) -> Dict[str, float]:
        """
        Create personalized weights for embedding spaces based on user preferences.
        
        Args:
            user_preferences: Dictionary with preference weights
                - musicality: 0.0-1.0 (emphasis on musical matching)
                - tempo_precision: 0.0-1.0 (emphasis on exact tempo)
                - difficulty_match: 0.0-1.0 (emphasis on difficulty matching)
                - energy_match: 0.0-1.0 (emphasis on energy matching)
                - flow: 0.0-1.0 (emphasis on transitions)
                
        Returns:
            Weights dictionary for embedding spaces
        """
        # Default weights
        base_weights = {
            "text": 0.3,
            "tempo": 0.25,
            "difficulty": 0.15,
            "energy": 0.15,
            "role": 0.1,
            "transition": 0.05
        }
        
        # Adjust based on preferences
        if "musicality" in user_preferences:
            base_weights["text"] *= (1.0 + user_preferences["musicality"])
        
        if "tempo_precision" in user_preferences:
            base_weights["tempo"] *= (1.0 + user_preferences["tempo_precision"])
        
        if "difficulty_match" in user_preferences:
            base_weights["difficulty"] *= (1.0 + user_preferences["difficulty_match"])
        
        if "energy_match" in user_preferences:
            base_weights["energy"] *= (1.0 + user_preferences["energy_match"])
        
        if "flow" in user_preferences:
            base_weights["transition"] *= (1.0 + user_preferences["flow"])
        
        # Normalize weights to sum to 1.0
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        return normalized_weights
    
    def generate_transition_aware_choreography(self,
                                             music_features: MusicFeatures,
                                             sequence_length: int,
                                             target_difficulty: str = "intermediate",
                                             target_energy: Optional[str] = None,
                                             role_focus: str = "both",
                                             diversity_factor: float = 0.3,
                                             transition_weight: float = 0.4) -> List[SuperlinkedRecommendationScore]:
        """
        Generate a complete choreography sequence using transition-aware selection.
        
        This method uses the CustomSpace for move flow patterns to create smooth
        transitions between moves, ensuring the choreography flows naturally.
        
        Args:
            music_features: Analyzed music features
            sequence_length: Number of moves in the choreography
            target_difficulty: Desired difficulty level
            target_energy: Desired energy level (auto-detected if None)
            role_focus: Role focus preference
            diversity_factor: 0.0-1.0, controls move diversity
            transition_weight: 0.0-1.0, weight for transition compatibility
            
        Returns:
            List of selected moves in choreography order
        """
        start_time = time.time()
        
        # Auto-detect energy level if not provided
        if target_energy is None:
            target_energy = self._detect_music_energy_level(music_features)
        
        selected_moves = []
        used_moves = set()
        
        # Generate first move without transition constraints
        first_move_candidates = self.recommend_moves(
            music_features=music_features,
            target_difficulty=target_difficulty,
            target_energy=target_energy,
            role_focus=role_focus,
            description="opening move for choreography",
            top_k=20,
            enable_transition_awareness=False
        )
        
        if not first_move_candidates:
            logger.warning("No candidates found for first move")
            return []
        
        # Select first move (prefer basic steps for opening)
        first_move = None
        for candidate in first_move_candidates:
            if "basic" in candidate.move_candidate.move_label.lower():
                first_move = candidate
                break
        
        if first_move is None:
            first_move = first_move_candidates[0]
        
        selected_moves.append(first_move)
        used_moves.add(first_move.move_candidate.move_id)
        
        # Generate subsequent moves with transition awareness
        for i in range(1, sequence_length):
            previous_move = selected_moves[-1]
            
            # Get transition-compatible moves
            compatible_moves = self.get_move_transitions(previous_move.move_candidate.move_id)
            
            # Get general recommendations
            all_candidates = self.recommend_moves(
                music_features=music_features,
                target_difficulty=target_difficulty,
                target_energy=target_energy,
                role_focus=role_focus,
                description=f"move {i+1} following {previous_move.move_candidate.move_label}",
                top_k=30,
                enable_transition_awareness=True
            )
            
            # Score candidates based on transition compatibility and diversity
            scored_candidates = []
            for candidate in all_candidates:
                move_id = candidate.move_candidate.move_id
                
                # Base similarity score
                score = candidate.similarity_score
                
                # Transition compatibility bonus
                if move_id in compatible_moves:
                    score += transition_weight * 0.3  # Boost for compatible transitions
                
                # Diversity penalty for recently used moves
                if move_id in used_moves:
                    score *= (1.0 - diversity_factor)
                
                # Prefer different move types for variety
                if i > 1:
                    recent_labels = [m.move_candidate.move_label for m in selected_moves[-2:]]
                    if candidate.move_candidate.move_label not in recent_labels:
                        score += diversity_factor * 0.2
                
                scored_candidates.append((candidate, score))
            
            # Sort by adjusted score and select best
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            if scored_candidates:
                next_move = scored_candidates[0][0]
                selected_moves.append(next_move)
                used_moves.add(next_move.move_candidate.move_id)
            else:
                logger.warning(f"No suitable candidates found for position {i+1}")
                break
        
        # Update performance stats
        generation_time = (time.time() - start_time) * 1000
        logger.info(f"Generated transition-aware choreography with {len(selected_moves)} moves in {generation_time:.2f}ms")
        
        return selected_moves

    def batch_recommend_for_sequence(self, 
                                   music_features: MusicFeatures,
                                   sequence_length: int,
                                   diversity_factor: float = 0.3,
                                   **kwargs) -> List[List[SuperlinkedRecommendationScore]]:
        """
        Generate recommendations for a sequence of moves with diversity control.
        
        Args:
            music_features: Analyzed music features
            sequence_length: Number of moves in sequence
            diversity_factor: 0.0-1.0, higher values increase diversity
            **kwargs: Additional parameters for recommend_moves
            
        Returns:
            List of recommendation lists for each position in sequence
        """
        all_recommendations = []
        used_moves = set()
        
        for i in range(sequence_length):
            # Get recommendations
            recommendations = self.recommend_moves(music_features, **kwargs)
            
            # Apply diversity filtering
            if diversity_factor > 0 and used_moves:
                filtered_recommendations = []
                for rec in recommendations:
                    if rec.move_candidate.move_id not in used_moves:
                        filtered_recommendations.append(rec)
                    elif len(filtered_recommendations) < 3:  # Keep some options
                        # Reduce score based on diversity factor
                        rec.similarity_score *= (1.0 - diversity_factor)
                        filtered_recommendations.append(rec)
                
                recommendations = filtered_recommendations
            
            # Add top move to used set
            if recommendations:
                used_moves.add(recommendations[0].move_candidate.move_id)
            
            all_recommendations.append(recommendations)
        
        return all_recommendations
    
    def _create_move_candidate_from_qdrant_result(self, result: SuperlinkedSearchResult) -> SuperlinkedMoveCandidate:
        """Create SuperlinkedMoveCandidate from Qdrant search result."""
        return SuperlinkedMoveCandidate(
            move_id=result.clip_id,
            video_path=result.video_path,
            move_label=result.move_label,
            move_description=result.move_description,
            tempo=result.tempo,
            difficulty_score=result.difficulty_score,
            energy_level=result.energy_level,
            role_focus=result.role_focus,
            notes=result.notes,
            embedding=result.embedding if result.embedding is not None else np.array([]),
            transition_compatibility=result.transition_compatibility
        )
    
    def _generate_recommendation_explanation_from_qdrant(self, 
                                                       result: SuperlinkedSearchResult, 
                                                       music_features: MusicFeatures) -> str:
        """Generate human-readable explanation for Qdrant recommendation."""
        similarity = result.similarity_score
        tempo_diff = abs(result.tempo - music_features.tempo)
        
        # Generate explanation based on similarity score and characteristics
        if similarity > 0.8:
            quality = "Excellent"
        elif similarity > 0.6:
            quality = "Good"
        elif similarity > 0.4:
            quality = "Moderate"
        else:
            quality = "Basic"
        
        tempo_desc = ""
        if tempo_diff <= 5:
            tempo_desc = "perfect tempo match"
        elif tempo_diff <= 10:
            tempo_desc = f"close tempo (±{tempo_diff:.0f} BPM)"
        else:
            tempo_desc = f"tempo adaptation needed (±{tempo_diff:.0f} BPM)"
        
        explanation = f"{quality} match with {tempo_desc}, {result.energy_level} energy {result.move_label}"
        
        if result.notes:
            explanation += f" - {result.notes}"
        
        return explanation
    
    def _convert_difficulty_to_score(self, difficulty: str) -> float:
        """Convert difficulty string to numerical score."""
        difficulty_map = {
            "beginner": 1.0,
            "intermediate": 2.0,
            "advanced": 3.0
        }
        return difficulty_map.get(difficulty.lower(), 2.0)

    def _calculate_energy_compatibility(self, target_energy: str, move_energy: str) -> float:
        """
        Calculate energy level compatibility using CategoricalSimilaritySpace logic.
        
        Args:
            target_energy: Target energy level
            move_energy: Move's energy level
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        energy_levels = ["low", "medium", "high"]
        
        if target_energy not in energy_levels or move_energy not in energy_levels:
            return 0.5  # Default compatibility
        
        target_idx = energy_levels.index(target_energy)
        move_idx = energy_levels.index(move_energy)
        
        # Calculate compatibility based on distance
        distance = abs(target_idx - move_idx)
        
        if distance == 0:
            return 1.0  # Perfect match
        elif distance == 1:
            return 0.7  # Adjacent levels are compatible
        else:
            return 0.3  # Opposite levels have low compatibility

    def _calculate_role_compatibility(self, target_role: str, move_role: str) -> float:
        """
        Calculate role focus compatibility using CategoricalSimilaritySpace logic.
        
        Args:
            target_role: Target role focus
            move_role: Move's role focus
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        # "both" is compatible with everything
        if target_role == "both" or move_role == "both":
            return 1.0
        
        # Exact match
        if target_role == move_role:
            return 1.0
        
        # Different specific roles have moderate compatibility
        return 0.6

    def _calculate_difficulty_compatibility(self, target_difficulty: str, move_difficulty_score: float) -> float:
        """
        Calculate difficulty compatibility using NumberSpace logic.
        
        Args:
            target_difficulty: Target difficulty level
            move_difficulty_score: Move's difficulty score (1.0-3.0)
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        target_score = self._convert_difficulty_to_score(target_difficulty)
        
        # Calculate linear distance
        distance = abs(target_score - move_difficulty_score)
        
        # Convert distance to compatibility (max distance is 2.0)
        compatibility = 1.0 - (distance / 2.0)
        
        return max(0.0, compatibility)


    def generate_complete_choreography_with_transitions(self,
                                                      music_features: MusicFeatures,
                                                      choreography_length: int = 8,
                                                      target_difficulty: str = "intermediate",
                                                      style_preference: str = "balanced",
                                                      natural_language_description: str = "") -> Dict[str, Any]:
        """
        Generate a complete choreography using all SuperlinkedRecommendationEngine capabilities.
        
        This method demonstrates the complete replacement of the complex RecommendationEngine
        by using unified vector similarity search with specialized embedding spaces for:
        - Semantic move understanding (TextSimilaritySpace)
        - Tempo-aware search with linear BPM relationships (NumberSpace)
        - Categorical filtering for energy and role focus (CategoricalSimilaritySpace)
        - Transition-aware choreography generation (CustomSpace)
        
        Args:
            music_features: Analyzed music features
            choreography_length: Number of moves in the choreography
            target_difficulty: Desired difficulty level
            style_preference: Style preference (balanced/energetic/smooth/technical)
            natural_language_description: Optional natural language description
            
        Returns:
            Dictionary with complete choreography and analysis
        """
        start_time = time.time()
        
        # Auto-detect energy level from music
        detected_energy = self._detect_music_energy_level(music_features)
        
        # Create style-specific parameters
        style_params = self._get_style_parameters(style_preference, detected_energy)
        
        # Generate enhanced description if not provided
        if not natural_language_description:
            natural_language_description = self._generate_enhanced_music_description(
                music_features, target_difficulty, detected_energy, style_preference
            )
        
        # Phase 1: Semantic search for opening moves
        logger.info("Phase 1: Semantic search for opening moves")
        opening_candidates = self.search_semantic_moves(
            semantic_query=f"opening {natural_language_description}",
            music_features=music_features,
            top_k=10,
            semantic_weight=0.8
        )
        
        # Phase 2: Tempo-aware search for rhythm matching
        logger.info("Phase 2: Tempo-aware search for rhythm matching")
        tempo_candidates = self.search_tempo_aware_moves(
            target_tempo=music_features.tempo,
            music_features=music_features,
            tempo_tolerance=style_params["tempo_tolerance"],
            top_k=15
        )
        
        # Phase 3: Categorical filtering for energy and role consistency
        logger.info("Phase 3: Categorical filtering for energy and role consistency")
        categorical_candidates = self.search_categorical_filtered_moves(
            music_features=music_features,
            energy_level=detected_energy,
            role_focus=style_params["role_focus"],
            difficulty_level=target_difficulty,
            top_k=20
        )
        
        # Phase 4: Generate transition-aware choreography
        logger.info("Phase 4: Generating transition-aware choreography")
        choreography_sequence = self.generate_transition_aware_choreography(
            music_features=music_features,
            sequence_length=choreography_length,
            target_difficulty=target_difficulty,
            target_energy=detected_energy,
            role_focus=style_params["role_focus"],
            diversity_factor=style_params["diversity_factor"],
            transition_weight=style_params["transition_weight"]
        )
        
        # Phase 5: Analyze choreography quality
        quality_analysis = self._analyze_choreography_quality(
            choreography_sequence, music_features, style_params
        )
        
        # Compile results
        total_time = (time.time() - start_time) * 1000
        
        result = {
            "choreography_sequence": choreography_sequence,
            "choreography_analysis": {
                "total_moves": len(choreography_sequence),
                "average_similarity_score": np.mean([m.similarity_score for m in choreography_sequence]) if choreography_sequence else 0.0,
                "difficulty_distribution": self._analyze_difficulty_distribution(choreography_sequence),
                "energy_consistency": self._analyze_energy_consistency(choreography_sequence, detected_energy),
                "transition_quality": quality_analysis["transition_quality"],
                "tempo_compatibility": quality_analysis["tempo_compatibility"],
                "style_adherence": quality_analysis["style_adherence"]
            },
            "search_phases": {
                "semantic_candidates": len(opening_candidates),
                "tempo_candidates": len(tempo_candidates),
                "categorical_candidates": len(categorical_candidates),
                "final_sequence_length": len(choreography_sequence)
            },
            "generation_metadata": {
                "target_difficulty": target_difficulty,
                "detected_energy": detected_energy,
                "style_preference": style_preference,
                "natural_language_description": natural_language_description,
                "music_tempo": music_features.tempo,
                "generation_time_ms": total_time
            },
            "performance_stats": self.get_performance_stats()
        }
        
        logger.info(f"Complete choreography generation finished in {total_time:.2f}ms")
        return result

    def _get_style_parameters(self, style_preference: str, detected_energy: str) -> Dict[str, Any]:
        """Get style-specific parameters for choreography generation."""
        base_params = {
            "tempo_tolerance": 10.0,
            "role_focus": "both",
            "diversity_factor": 0.3,
            "transition_weight": 0.4
        }
        
        if style_preference == "energetic":
            return {
                **base_params,
                "tempo_tolerance": 15.0,
                "diversity_factor": 0.5,
                "transition_weight": 0.3
            }
        elif style_preference == "smooth":
            return {
                **base_params,
                "tempo_tolerance": 8.0,
                "diversity_factor": 0.2,
                "transition_weight": 0.6
            }
        elif style_preference == "technical":
            return {
                **base_params,
                "tempo_tolerance": 5.0,
                "diversity_factor": 0.4,
                "transition_weight": 0.5
            }
        else:  # balanced
            return base_params

    def _generate_enhanced_music_description(self, music_features: MusicFeatures, 
                                           difficulty: str, energy: str, style: str) -> str:
        """Generate enhanced description incorporating all musical characteristics."""
        tempo_desc = "slow" if music_features.tempo < 110 else "fast" if music_features.tempo > 130 else "medium tempo"
        
        # Analyze additional musical characteristics
        avg_spectral_centroid = np.mean(music_features.spectral_centroid)
        brightness = "bright" if avg_spectral_centroid > 2000 else "warm"
        
        percussive_strength = np.mean(np.abs(music_features.percussive_component))
        rhythm_desc = "strong rhythm" if percussive_strength > 0.1 else "gentle rhythm"
        
        # Incorporate style preference
        style_desc = {
            "energetic": "dynamic and expressive",
            "smooth": "flowing and connected", 
            "technical": "precise and complex",
            "balanced": "well-rounded"
        }.get(style, "versatile")
        
        description = f"{style_desc} {difficulty} {energy} energy bachata with {tempo_desc} {brightness} sound and {rhythm_desc}"
        
        return description

    def _analyze_choreography_quality(self, choreography: List[SuperlinkedRecommendationScore], 
                                    music_features: MusicFeatures, style_params: Dict[str, Any]) -> Dict[str, float]:
        """Analyze the quality of the generated choreography."""
        if not choreography:
            return {"transition_quality": 0.0, "tempo_compatibility": 0.0, "style_adherence": 0.0}
        
        # Analyze transition quality
        transition_scores = []
        for i in range(len(choreography) - 1):
            current_move = choreography[i].move_candidate
            next_move = choreography[i + 1].move_candidate
            
            compatible_moves = self.get_move_transitions(current_move.move_id)
            if next_move.move_id in compatible_moves:
                transition_scores.append(1.0)
            else:
                # Calculate compatibility based on energy and difficulty similarity
                energy_compat = self._calculate_energy_compatibility(current_move.energy_level, next_move.energy_level)
                diff_compat = self._calculate_difficulty_compatibility("intermediate", abs(current_move.difficulty_score - next_move.difficulty_score))
                transition_scores.append((energy_compat + diff_compat) / 2.0)
        
        transition_quality = np.mean(transition_scores) if transition_scores else 0.0
        
        # Analyze tempo compatibility
        tempo_scores = []
        for move in choreography:
            tempo_diff = abs(move.move_candidate.tempo - music_features.tempo)
            tempo_score = max(0.0, 1.0 - (tempo_diff / style_params["tempo_tolerance"]))
            tempo_scores.append(tempo_score)
        
        tempo_compatibility = np.mean(tempo_scores)
        
        # Analyze style adherence (based on similarity scores)
        style_adherence = np.mean([move.similarity_score for move in choreography])
        
        return {
            "transition_quality": transition_quality,
            "tempo_compatibility": tempo_compatibility,
            "style_adherence": style_adherence
        }

    def _analyze_difficulty_distribution(self, choreography: List[SuperlinkedRecommendationScore]) -> Dict[str, int]:
        """Analyze the distribution of difficulty levels in the choreography."""
        distribution = {"beginner": 0, "intermediate": 0, "advanced": 0}
        
        for move in choreography:
            score = move.move_candidate.difficulty_score
            if score <= 1.5:
                distribution["beginner"] += 1
            elif score <= 2.5:
                distribution["intermediate"] += 1
            else:
                distribution["advanced"] += 1
        
        return distribution

    def _analyze_energy_consistency(self, choreography: List[SuperlinkedRecommendationScore], target_energy: str) -> float:
        """Analyze how consistent the energy levels are with the target."""
        if not choreography:
            return 0.0
        
        consistency_scores = []
        for move in choreography:
            consistency = self._calculate_energy_compatibility(target_energy, move.move_candidate.energy_level)
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores)


def create_superlinked_recommendation_engine(data_dir: str = "data", qdrant_config: Optional[QdrantConfig] = None) -> SuperlinkedRecommendationEngine:
    """
    Factory function to create a SuperlinkedRecommendationEngine with Qdrant integration.
    
    Args:
        data_dir: Directory containing move annotations
        qdrant_config: Optional Qdrant configuration
        
    Returns:
        Initialized SuperlinkedRecommendationEngine with Qdrant integration
    """
    return SuperlinkedRecommendationEngine(data_dir, qdrant_config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create the engine
    engine = create_superlinked_recommendation_engine("data")
    
    # Print statistics
    stats = engine.get_performance_stats()
    print(f"\nSuperlinkedRecommendationEngine Statistics:")
    print(f"Engine type: {stats['engine_type']}")
    print(f"Unified vector dimension: {stats['unified_vector_dimension']}")
    print(f"Total moves: {stats['total_moves']}")
    print(f"Embedding spaces: {stats['embedding_spaces']}")
    
    # Create mock music features for testing
    from .music_analyzer import MusicFeatures
    
    mock_music_features = MusicFeatures(
        tempo=120.0,
        beat_positions=np.array([0.5, 1.0, 1.5, 2.0]),
        duration=180.0,
        rms_energy=np.array([0.1, 0.2, 0.15, 0.18]),
        spectral_centroid=np.array([1500, 1600, 1550, 1580]),
        percussive_component=np.array([0.1, 0.12, 0.11, 0.13]),
        energy_profile=np.array([0.5, 0.7, 0.6, 0.65])
    )
    
    # Test unified vector recommendations
    print(f"\nTesting unified vector recommendations:")
    recommendations = engine.recommend_moves(
        music_features=mock_music_features,
        target_difficulty="intermediate",
        target_energy="medium",
        top_k=5
    )
    
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")
        print(f"   {rec.explanation}")
    
    # Test natural language queries
    print(f"\nTesting natural language queries:")
    nl_queries = [
        "energetic intermediate moves for 125 BPM song",
        "smooth beginner basic steps with low energy",
        "advanced turns and styling for fast bachata"
    ]
    
    for query in nl_queries:
        print(f"\nQuery: '{query}'")
        recommendations = engine.recommend_with_natural_language(query, mock_music_features, top_k=3)
        for rec in recommendations:
            print(f"- {rec.move_candidate.move_label} (score: {rec.similarity_score:.3f})")