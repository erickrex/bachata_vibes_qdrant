"""
Multi-factor scoring recommendation system for Bachata choreography generation.
Combines audio similarity, tempo matching, energy alignment, and difficulty compatibility.
Now supports Qdrant vector search for optimized similarity matching.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time

from .feature_fusion import FeatureFusion, MultiModalEmbedding, SimilarityScore
from .music_analyzer import MusicFeatures
from .move_analyzer import MoveAnalysisResult
from .qdrant_service import SuperlinkedQdrantService, QdrantConfig, SuperlinkedSearchResult, create_qdrant_service

logger = logging.getLogger(__name__)


@dataclass
class MoveCandidate:
    """Container for a move candidate with its analysis results and metadata."""
    move_id: str
    video_path: str
    move_label: str
    analysis_result: MoveAnalysisResult
    multimodal_embedding: MultiModalEmbedding
    
    # Metadata from annotations
    energy_level: str = "medium"  # low/medium/high
    difficulty: str = "intermediate"  # beginner/intermediate/advanced
    estimated_tempo: float = 120.0
    lead_follow_roles: str = "both"  # lead_focus/follow_focus/both


@dataclass
class RecommendationScore:
    """Container for detailed recommendation scoring results."""
    move_candidate: MoveCandidate
    overall_score: float
    
    # Component scores
    audio_similarity: float
    tempo_compatibility: float
    energy_alignment: float
    difficulty_compatibility: float
    
    # Detailed breakdown
    tempo_difference: float
    energy_match: bool
    difficulty_match: bool
    
    # Weights used
    weights: Dict[str, float]


@dataclass
class RecommendationRequest:
    """Container for recommendation request parameters."""
    music_features: MusicFeatures
    music_embedding: MultiModalEmbedding
    
    # User preferences
    target_difficulty: str = "intermediate"  # beginner/intermediate/advanced
    target_energy: Optional[str] = None  # None for auto-detect, or low/medium/high
    tempo_tolerance: float = 10.0  # BPM tolerance
    
    # Scoring weights
    weights: Optional[Dict[str, float]] = None


class RecommendationEngine:
    """
    Multi-factor scoring recommendation system that combines:
    - Audio similarity (40%) - now using Qdrant vector search
    - Tempo matching (30%) 
    - Energy alignment (20%)
    - Difficulty compatibility (10%)
    
    Features Qdrant integration for optimized similarity search with fallback to in-memory.
    """
    
    def __init__(self, use_qdrant: bool = True, qdrant_config: Optional[QdrantConfig] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            use_qdrant: Whether to use Qdrant for vector search
            qdrant_config: Optional Qdrant configuration
        """
        self.feature_fusion = FeatureFusion()
        
        # Default scoring weights
        self.default_weights = {
            'audio_similarity': 0.40,
            'tempo_matching': 0.30,
            'energy_alignment': 0.20,
            'difficulty_compatibility': 0.10
        }
        
        # Tempo compatibility parameters
        self.tempo_tolerance = 10.0  # ±10 BPM tolerance
        
        # Qdrant integration
        self.use_qdrant = use_qdrant
        self.qdrant_service = None
        self.qdrant_available = False
        
        if use_qdrant:
            try:
                self.qdrant_service = create_qdrant_service(qdrant_config)
                # Test if it's a real service (not mock)
                health = self.qdrant_service.health_check()
                self.qdrant_available = health.get("qdrant_available", False)
                
                if self.qdrant_available:
                    logger.info("RecommendationEngine initialized with Qdrant vector search")
                else:
                    logger.warning("Qdrant not available, falling back to in-memory similarity search")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant service: {e}, using in-memory search")
                self.qdrant_available = False
        
        # Performance tracking
        self.performance_stats = {
            'qdrant_searches': 0,
            'memory_searches': 0,
            'avg_qdrant_time_ms': 0.0,
            'avg_memory_time_ms': 0.0,
            'qdrant_fallbacks': 0
        }
        
        search_method = "Qdrant vector search" if self.qdrant_available else "in-memory cosine similarity"
        logger.info(f"RecommendationEngine initialized with {search_method}, weights: audio=40%, tempo=30%, energy=20%, difficulty=10%")
    
    def recommend_moves(self, 
                       request: RecommendationRequest,
                       move_candidates: Optional[List[MoveCandidate]] = None,
                       top_k: int = 10) -> List[RecommendationScore]:
        """
        Recommend top-k moves based on multi-factor scoring.
        Uses Qdrant vector search when available, falls back to in-memory search.
        
        Args:
            request: Recommendation request with music features and preferences
            move_candidates: List of available move candidates (optional if using Qdrant)
            top_k: Number of top recommendations to return
            
        Returns:
            List of RecommendationScore objects sorted by overall score (descending)
        """
        # Try Qdrant-based recommendation first
        if self.qdrant_available and self.qdrant_service:
            try:
                return self._recommend_moves_qdrant(request, top_k)
            except Exception as e:
                logger.warning(f"Qdrant recommendation failed: {e}, falling back to in-memory")
                self.performance_stats['qdrant_fallbacks'] += 1
                self.qdrant_available = False  # Disable for this session
        
        # Fallback to in-memory recommendation
        if move_candidates is None:
            raise ValueError("move_candidates required when Qdrant is not available")
            
        return self._recommend_moves_memory(request, move_candidates, top_k)
    
    def _recommend_moves_qdrant(self, 
                               request: RecommendationRequest,
                               top_k: int = 10) -> List[RecommendationScore]:
        """
        Recommend moves using Qdrant vector search.
        
        Args:
            request: Recommendation request with music features and preferences
            top_k: Number of top recommendations to return
            
        Returns:
            List of RecommendationScore objects sorted by overall score (descending)
        """
        start_time = time.time()
        
        logger.info(f"Generating Qdrant-based recommendations, top_k={top_k}")
        
        # Use provided weights or defaults
        weights = request.weights or self.default_weights
        
        # Auto-detect target energy if not specified
        target_energy = request.target_energy
        if target_energy is None:
            target_energy = self._detect_music_energy_level(request.music_features)
            logger.info(f"Auto-detected target energy level: {target_energy}")
        
        # Prepare query embedding (combine audio and pose components)
        music_audio = request.music_embedding.audio_embedding
        music_pose = request.music_embedding.pose_embedding
        query_embedding = np.concatenate([music_audio, music_pose])
        
        # Define tempo range for filtering
        tempo_tolerance = request.tempo_tolerance
        tempo_range = (
            request.music_features.tempo - tempo_tolerance,
            request.music_features.tempo + tempo_tolerance
        )
        
        # Search with metadata filtering
        # Request more results than needed to account for post-processing filtering
        search_limit = min(top_k * 3, 50)  # Get 3x results for better filtering
        
        search_results = self.qdrant_service.search_similar_moves(
            query_embedding=query_embedding,
            limit=search_limit,
            tempo_range=tempo_range,
            difficulty=request.target_difficulty if request.target_difficulty != "intermediate" else None,
            energy_level=target_energy if target_energy != "medium" else None,
            min_quality=0.7  # Minimum analysis quality
        )
        
        # Convert search results to RecommendationScore objects
        scores = []
        for search_result in search_results:
            # Create a minimal MoveCandidate from search result metadata
            candidate = self._create_candidate_from_search_result(search_result)
            
            # Calculate detailed scoring (tempo, energy, difficulty components)
            score = self._score_move_candidate_from_qdrant(
                request.music_features,
                request.music_embedding,
                candidate,
                search_result,
                request.target_difficulty,
                target_energy,
                tempo_tolerance,
                weights
            )
            scores.append(score)
        
        # Sort by overall score (descending) and take top_k
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        top_scores = scores[:top_k]
        
        # Update performance stats
        search_time = (time.time() - start_time) * 1000
        self.performance_stats['qdrant_searches'] += 1
        self.performance_stats['avg_qdrant_time_ms'] = (
            (self.performance_stats['avg_qdrant_time_ms'] * (self.performance_stats['qdrant_searches'] - 1) + search_time) /
            self.performance_stats['qdrant_searches']
        )
        
        logger.info(f"Qdrant recommendations completed in {search_time:.2f}ms: {[f'{s.move_candidate.move_label}={s.overall_score:.3f}' for s in top_scores[:3]]}")
        
        return top_scores
    
    def _recommend_moves_memory(self, 
                               request: RecommendationRequest,
                               move_candidates: List[MoveCandidate],
                               top_k: int = 10) -> List[RecommendationScore]:
        """
        Recommend moves using in-memory cosine similarity (fallback method).
        
        Args:
            request: Recommendation request with music features and preferences
            move_candidates: List of available move candidates
            top_k: Number of top recommendations to return
            
        Returns:
            List of RecommendationScore objects sorted by overall score (descending)
        """
        start_time = time.time()
        
        logger.info(f"Generating in-memory recommendations for {len(move_candidates)} candidates, top_k={top_k}")
        
        # Use provided weights or defaults
        weights = request.weights or self.default_weights
        
        # Auto-detect target energy if not specified
        target_energy = request.target_energy
        if target_energy is None:
            target_energy = self._detect_music_energy_level(request.music_features)
            logger.info(f"Auto-detected target energy level: {target_energy}")
        
        # Score all candidates
        scores = []
        for candidate in move_candidates:
            score = self._score_move_candidate(
                request.music_features,
                request.music_embedding,
                candidate,
                request.target_difficulty,
                target_energy,
                request.tempo_tolerance,
                weights
            )
            scores.append(score)
        
        # Sort by overall score (descending)
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Return top-k
        top_scores = scores[:top_k]
        
        # Update performance stats
        search_time = (time.time() - start_time) * 1000
        self.performance_stats['memory_searches'] += 1
        self.performance_stats['avg_memory_time_ms'] = (
            (self.performance_stats['avg_memory_time_ms'] * (self.performance_stats['memory_searches'] - 1) + search_time) /
            self.performance_stats['memory_searches']
        )
        
        logger.info(f"In-memory recommendations completed in {search_time:.2f}ms: {[f'{s.move_candidate.move_label}={s.overall_score:.3f}' for s in top_scores[:3]]}")
        
        return top_scores
    
    def _score_move_candidate(self,
                            music_features: MusicFeatures,
                            music_embedding: MultiModalEmbedding,
                            candidate: MoveCandidate,
                            target_difficulty: str,
                            target_energy: str,
                            tempo_tolerance: float,
                            weights: Dict[str, float]) -> RecommendationScore:
        """Score a single move candidate using multi-factor scoring."""
        
        # 1. Audio similarity (cosine similarity between embeddings)
        audio_similarity = self._calculate_audio_similarity(
            music_embedding, candidate.multimodal_embedding
        )
        
        # 2. Tempo compatibility (BPM range matching with tolerance)
        tempo_compatibility, tempo_difference = self._calculate_tempo_compatibility(
            music_features.tempo, candidate, tempo_tolerance
        )
        
        # 3. Energy alignment (low/medium/high matching)
        energy_alignment, energy_match = self._calculate_energy_alignment(
            target_energy, candidate.energy_level
        )
        
        # 4. Difficulty compatibility (beginner/intermediate/advanced matching)
        difficulty_compatibility, difficulty_match = self._calculate_difficulty_compatibility(
            target_difficulty, candidate.difficulty
        )
        
        # Calculate weighted overall score
        overall_score = (
            weights['audio_similarity'] * audio_similarity +
            weights['tempo_matching'] * tempo_compatibility +
            weights['energy_alignment'] * energy_alignment +
            weights['difficulty_compatibility'] * difficulty_compatibility
        )
        
        return RecommendationScore(
            move_candidate=candidate,
            overall_score=overall_score,
            audio_similarity=audio_similarity,
            tempo_compatibility=tempo_compatibility,
            energy_alignment=energy_alignment,
            difficulty_compatibility=difficulty_compatibility,
            tempo_difference=tempo_difference,
            energy_match=energy_match,
            difficulty_match=difficulty_match,
            weights=weights
        )
    
    def _calculate_audio_similarity(self,
                                  music_embedding: MultiModalEmbedding,
                                  move_embedding: MultiModalEmbedding) -> float:
        """Calculate cosine similarity between music and move audio embeddings."""
        # Use the audio components of the embeddings for similarity
        music_audio = music_embedding.audio_embedding
        move_audio = move_embedding.audio_embedding
        
        # Calculate cosine similarity
        dot_product = np.dot(music_audio, move_audio)
        norm_music = np.linalg.norm(music_audio)
        norm_move = np.linalg.norm(move_audio)
        
        if norm_music > 0 and norm_move > 0:
            similarity = dot_product / (norm_music * norm_move)
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            return (similarity + 1.0) / 2.0
        else:
            return 0.0
    
    def _calculate_tempo_compatibility(self,
                                     music_tempo: float,
                                     candidate: MoveCandidate,
                                     tolerance: float) -> Tuple[float, float]:
        """
        Calculate tempo compatibility with BPM range matching (±tolerance).
        
        Returns:
            Tuple of (compatibility_score, tempo_difference)
        """
        # Get move's tempo compatibility range
        move_tempo_range = candidate.analysis_result.tempo_compatibility_range
        min_bpm, max_bpm = move_tempo_range
        
        # Calculate tempo difference
        if min_bpm <= music_tempo <= max_bpm:
            # Music tempo is within move's compatibility range
            tempo_difference = 0.0
            compatibility = 1.0
        else:
            # Calculate distance to nearest edge of range
            if music_tempo < min_bpm:
                tempo_difference = min_bpm - music_tempo
            else:
                tempo_difference = music_tempo - max_bpm
            
            # Apply tolerance
            if tempo_difference <= tolerance:
                # Within tolerance - linear decay
                compatibility = 1.0 - (tempo_difference / tolerance)
            else:
                # Outside tolerance - exponential decay
                excess = tempo_difference - tolerance
                compatibility = np.exp(-excess / 20.0)  # Decay factor
        
        return max(0.0, min(1.0, compatibility)), tempo_difference
    
    def _calculate_energy_alignment(self,
                                  target_energy: str,
                                  move_energy: str) -> Tuple[float, bool]:
        """
        Calculate energy level alignment between target and move.
        
        Returns:
            Tuple of (alignment_score, exact_match)
        """
        # Energy level mapping to numeric values
        energy_levels = {'low': 1, 'medium': 2, 'high': 3}
        
        target_level = energy_levels.get(target_energy, 2)
        move_level = energy_levels.get(move_energy, 2)
        
        # Calculate alignment score
        difference = abs(target_level - move_level)
        
        if difference == 0:
            # Exact match
            return 1.0, True
        elif difference == 1:
            # Adjacent level (e.g., medium vs high)
            return 0.7, False
        else:
            # Opposite levels (low vs high)
            return 0.3, False
    
    def _calculate_difficulty_compatibility(self,
                                          target_difficulty: str,
                                          move_difficulty: str) -> Tuple[float, bool]:
        """
        Calculate difficulty compatibility between target and move.
        
        Returns:
            Tuple of (compatibility_score, exact_match)
        """
        # Difficulty level mapping to numeric values
        difficulty_levels = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        
        target_level = difficulty_levels.get(target_difficulty, 2)
        move_level = difficulty_levels.get(move_difficulty, 2)
        
        # Calculate compatibility score
        difference = abs(target_level - move_level)
        
        if difference == 0:
            # Exact match
            return 1.0, True
        elif difference == 1:
            # Adjacent level - still compatible
            return 0.8, False
        else:
            # Two levels apart - less compatible but not impossible
            return 0.4, False
    
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
    
    def create_move_candidate(self,
                            move_id: str,
                            video_path: str,
                            move_label: str,
                            analysis_result: MoveAnalysisResult,
                            music_features: MusicFeatures,
                            **metadata) -> MoveCandidate:
        """
        Create a MoveCandidate with multimodal embedding.
        
        Args:
            move_id: Unique identifier for the move
            video_path: Path to the move video file
            move_label: Human-readable move label
            analysis_result: Complete move analysis from MoveAnalyzer
            music_features: Music features for embedding creation
            **metadata: Additional metadata (energy_level, difficulty, etc.)
        """
        # Create multimodal embedding
        multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
            music_features, analysis_result
        )
        
        # Create candidate with metadata
        candidate = MoveCandidate(
            move_id=move_id,
            video_path=video_path,
            move_label=move_label,
            analysis_result=analysis_result,
            multimodal_embedding=multimodal_embedding,
            energy_level=metadata.get('energy_level', 'medium'),
            difficulty=metadata.get('difficulty', 'intermediate'),
            estimated_tempo=metadata.get('estimated_tempo', 120.0),
            lead_follow_roles=metadata.get('lead_follow_roles', 'both')
        )
        
        return candidate
    
    def batch_score_moves(self,
                         music_features: MusicFeatures,
                         music_embedding: MultiModalEmbedding,
                         move_candidates: List[MoveCandidate],
                         scoring_params: Dict[str, Any]) -> List[RecommendationScore]:
        """
        Batch score multiple moves with custom parameters.
        
        Args:
            music_features: Music analysis features
            music_embedding: Music multimodal embedding
            move_candidates: List of move candidates to score
            scoring_params: Dictionary with scoring parameters
                - target_difficulty: str
                - target_energy: str
                - tempo_tolerance: float
                - weights: Dict[str, float]
        
        Returns:
            List of RecommendationScore objects (unsorted)
        """
        target_difficulty = scoring_params.get('target_difficulty', 'intermediate')
        target_energy = scoring_params.get('target_energy', 'medium')
        tempo_tolerance = scoring_params.get('tempo_tolerance', 10.0)
        weights = scoring_params.get('weights', self.default_weights)
        
        scores = []
        for candidate in move_candidates:
            score = self._score_move_candidate(
                music_features,
                music_embedding,
                candidate,
                target_difficulty,
                target_energy,
                tempo_tolerance,
                weights
            )
            scores.append(score)
        
        return scores
    
    def get_scoring_explanation(self, score: RecommendationScore) -> Dict[str, str]:
        """
        Get human-readable explanation of scoring components.
        
        Args:
            score: RecommendationScore to explain
            
        Returns:
            Dictionary with component explanations
        """
        explanations = {}
        
        # Audio similarity explanation
        if score.audio_similarity > 0.8:
            explanations['audio'] = "Excellent musical match"
        elif score.audio_similarity > 0.6:
            explanations['audio'] = "Good musical compatibility"
        elif score.audio_similarity > 0.4:
            explanations['audio'] = "Moderate musical fit"
        else:
            explanations['audio'] = "Limited musical compatibility"
        
        # Tempo explanation
        if score.tempo_difference == 0:
            explanations['tempo'] = "Perfect tempo match"
        elif score.tempo_difference <= 5:
            explanations['tempo'] = f"Very close tempo (±{score.tempo_difference:.1f} BPM)"
        elif score.tempo_difference <= 10:
            explanations['tempo'] = f"Good tempo fit (±{score.tempo_difference:.1f} BPM)"
        else:
            explanations['tempo'] = f"Tempo stretch needed (±{score.tempo_difference:.1f} BPM)"
        
        # Energy explanation
        if score.energy_match:
            explanations['energy'] = "Perfect energy level match"
        elif score.energy_alignment > 0.6:
            explanations['energy'] = "Compatible energy level"
        else:
            explanations['energy'] = "Different energy level"
        
        # Difficulty explanation
        if score.difficulty_match:
            explanations['difficulty'] = "Perfect difficulty match"
        elif score.difficulty_compatibility > 0.7:
            explanations['difficulty'] = "Compatible difficulty level"
        else:
            explanations['difficulty'] = "Different difficulty level"
        
        return explanations
    
    def _create_candidate_from_search_result(self, search_result: SuperlinkedSearchResult) -> MoveCandidate:
        """
        Create a MoveCandidate from Qdrant search result metadata.
        
        Args:
            search_result: Search result from Qdrant
            
        Returns:
            MoveCandidate object
        """
        metadata = search_result.metadata
        
        # Create minimal MoveAnalysisResult from metadata
        from .move_analyzer import MovementDynamics, PoseFeatures, HandFeatures
        
        # Create minimal movement dynamics
        movement_dynamics = MovementDynamics(
            velocity_profile=np.array([]),
            acceleration_profile=np.array([]),
            spatial_coverage=0.0,
            rhythm_score=0.8,
            complexity_score=metadata["movement_complexity"],
            dominant_movement_direction="forward",
            energy_level=metadata["energy_level"],
            footwork_area_coverage=0.0,
            upper_body_movement_range=0.0,
            rhythm_compatibility_score=0.8,
            movement_periodicity=0.8,
            transition_points=[],
            movement_intensity_profile=np.array([]),
            spatial_distribution={}
        )
        
        analysis_result = MoveAnalysisResult(
            video_path=metadata["video_path"],
            duration=metadata["duration"],
            frame_count=int(metadata["duration"] * 30),  # Estimate frames
            fps=30.0,
            pose_features=[],  # Empty for Qdrant-based results
            hand_features=[],  # Empty for Qdrant-based results
            movement_dynamics=movement_dynamics,
            pose_embedding=np.zeros(384),  # Placeholder
            movement_embedding=np.zeros(128),  # Placeholder
            movement_complexity_score=metadata["movement_complexity"],
            tempo_compatibility_range=(metadata["tempo_min"], metadata["tempo_max"]),
            difficulty_score=metadata["difficulty_score"],
            analysis_quality=metadata["analysis_quality"],
            pose_detection_rate=metadata["pose_detection_rate"]
        )
        
        # Create minimal MultiModalEmbedding (will use Qdrant similarity score)
        multimodal_embedding = MultiModalEmbedding(
            audio_embedding=np.zeros(128),  # Placeholder
            pose_embedding=np.zeros(384),   # Placeholder
            combined_embedding=np.zeros(512)  # Placeholder
        )
        
        # Create MoveCandidate
        candidate = MoveCandidate(
            move_id=metadata["move_id"],
            video_path=metadata["video_path"],
            move_label=metadata["move_label"],
            analysis_result=analysis_result,
            multimodal_embedding=multimodal_embedding,
            energy_level=metadata["energy_level"],
            difficulty=metadata["difficulty"],
            estimated_tempo=metadata["estimated_tempo"],
            lead_follow_roles=metadata["lead_follow_roles"]
        )
        
        return candidate
    
    def _score_move_candidate_from_qdrant(self,
                                        music_features: MusicFeatures,
                                        music_embedding: MultiModalEmbedding,
                                        candidate: MoveCandidate,
                                        search_result: SuperlinkedSearchResult,
                                        target_difficulty: str,
                                        target_energy: str,
                                        tempo_tolerance: float,
                                        weights: Dict[str, float]) -> RecommendationScore:
        """
        Score a move candidate using Qdrant search result and additional factors.
        
        Args:
            music_features: Music analysis features
            music_embedding: Music multimodal embedding
            candidate: Move candidate (created from search result)
            search_result: Original Qdrant search result
            target_difficulty: Target difficulty level
            target_energy: Target energy level
            tempo_tolerance: Tempo tolerance in BPM
            weights: Scoring weights
            
        Returns:
            RecommendationScore object
        """
        # 1. Audio similarity from Qdrant search score
        # Qdrant returns cosine similarity (0-1), normalize if needed
        audio_similarity = search_result.score
        
        # 2. Tempo compatibility (BPM range matching with tolerance)
        tempo_compatibility, tempo_difference = self._calculate_tempo_compatibility(
            music_features.tempo, candidate, tempo_tolerance
        )
        
        # 3. Energy alignment (low/medium/high matching)
        energy_alignment, energy_match = self._calculate_energy_alignment(
            target_energy, candidate.energy_level
        )
        
        # 4. Difficulty compatibility (beginner/intermediate/advanced matching)
        difficulty_compatibility, difficulty_match = self._calculate_difficulty_compatibility(
            target_difficulty, candidate.difficulty
        )
        
        # Calculate weighted overall score
        overall_score = (
            weights['audio_similarity'] * audio_similarity +
            weights['tempo_matching'] * tempo_compatibility +
            weights['energy_alignment'] * energy_alignment +
            weights['difficulty_compatibility'] * difficulty_compatibility
        )
        
        return RecommendationScore(
            move_candidate=candidate,
            overall_score=overall_score,
            audio_similarity=audio_similarity,
            tempo_compatibility=tempo_compatibility,
            energy_alignment=energy_alignment,
            difficulty_compatibility=difficulty_compatibility,
            tempo_difference=tempo_difference,
            energy_match=energy_match,
            difficulty_match=difficulty_match,
            weights=weights
        )
    
    def populate_qdrant_from_candidates(self, move_candidates: List[MoveCandidate]) -> Dict[str, Any]:
        """
        Populate Qdrant with move embeddings from existing candidates.
        
        Args:
            move_candidates: List of move candidates to store
            
        Returns:
            Population summary
        """
        if not self.qdrant_available or not self.qdrant_service:
            return {"error": "Qdrant not available"}
        
        try:
            summary = self.qdrant_service.migrate_from_memory_cache(move_candidates)
            logger.info(f"Populated Qdrant with {summary['successful_migrations']} move embeddings")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to populate Qdrant: {e}")
            return {"error": str(e)}
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """
        Get performance comparison between Qdrant and in-memory search.
        
        Returns:
            Performance statistics
        """
        stats = self.performance_stats.copy()
        
        # Add derived metrics
        total_searches = stats['qdrant_searches'] + stats['memory_searches']
        if total_searches > 0:
            stats['qdrant_usage_percent'] = (stats['qdrant_searches'] / total_searches) * 100
            stats['memory_usage_percent'] = (stats['memory_searches'] / total_searches) * 100
        else:
            stats['qdrant_usage_percent'] = 0
            stats['memory_usage_percent'] = 0
        
        # Speed comparison
        if stats['avg_qdrant_time_ms'] > 0 and stats['avg_memory_time_ms'] > 0:
            if stats['avg_qdrant_time_ms'] < stats['avg_memory_time_ms']:
                stats['qdrant_speedup'] = stats['avg_memory_time_ms'] / stats['avg_qdrant_time_ms']
                stats['faster_method'] = "qdrant"
            else:
                stats['qdrant_speedup'] = stats['avg_qdrant_time_ms'] / stats['avg_memory_time_ms']
                stats['faster_method'] = "memory"
        
        # Qdrant service stats if available
        if self.qdrant_service:
            try:
                qdrant_stats = self.qdrant_service.get_statistics()
                stats['qdrant_collection_size'] = qdrant_stats.total_points
                stats['qdrant_collection_size_mb'] = qdrant_stats.collection_size_mb
            except Exception as e:
                logger.warning(f"Failed to get Qdrant stats: {e}")
        
        return stats
    
    def is_qdrant_available(self) -> bool:
        """Check if Qdrant is available and healthy."""
        if not self.qdrant_available or not self.qdrant_service:
            return False
        
        try:
            health = self.qdrant_service.health_check()
            return health.get("can_search", False) and health.get("can_store", False)
        except Exception:
            return False
    
    def force_qdrant_refresh(self, qdrant_config: Optional[QdrantConfig] = None) -> bool:
        """
        Force refresh of Qdrant connection.
        
        Args:
            qdrant_config: Optional new configuration
            
        Returns:
            True if refresh successful
        """
        try:
            self.qdrant_service = create_qdrant_service(qdrant_config)
            health = self.qdrant_service.health_check()
            self.qdrant_available = health.get("qdrant_available", False)
            
            if self.qdrant_available:
                logger.info("Qdrant connection refreshed successfully")
                return True
            else:
                logger.warning("Qdrant refresh failed - service not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to refresh Qdrant connection: {e}")
            self.qdrant_available = False
            return False