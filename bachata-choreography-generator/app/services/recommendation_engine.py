"""
Multi-factor scoring recommendation system for Bachata choreography generation.
Combines audio similarity, tempo matching, energy alignment, and difficulty compatibility.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from .feature_fusion import FeatureFusion, MultiModalEmbedding, SimilarityScore
from .music_analyzer import MusicFeatures
from .move_analyzer import MoveAnalysisResult

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
    - Audio similarity (40%)
    - Tempo matching (30%) 
    - Energy alignment (20%)
    - Difficulty compatibility (10%)
    """
    
    def __init__(self):
        """Initialize the recommendation engine."""
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
        
        logger.info("RecommendationEngine initialized with default weights: audio=40%, tempo=30%, energy=20%, difficulty=10%")
    
    def recommend_moves(self, 
                       request: RecommendationRequest,
                       move_candidates: List[MoveCandidate],
                       top_k: int = 10) -> List[RecommendationScore]:
        """
        Recommend top-k moves based on multi-factor scoring.
        
        Args:
            request: Recommendation request with music features and preferences
            move_candidates: List of available move candidates
            top_k: Number of top recommendations to return
            
        Returns:
            List of RecommendationScore objects sorted by overall score (descending)
        """
        logger.info(f"Generating recommendations for {len(move_candidates)} candidates, top_k={top_k}")
        
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
        
        logger.info(f"Top recommendation scores: {[f'{s.move_candidate.move_label}={s.overall_score:.3f}' for s in top_scores[:3]]}")
        
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