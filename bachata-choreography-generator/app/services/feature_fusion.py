"""
Feature fusion system for multi-modal embeddings.
Combines audio and pose features into unified embeddings for similarity matching.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .music_analyzer import MusicFeatures
from .move_analyzer import MoveAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class MultiModalEmbedding:
    """Container for multi-modal embedding combining audio and pose features."""
    combined_embedding: np.ndarray  # 512-dimensional combined feature vector
    audio_embedding: np.ndarray     # 128-dimensional audio features
    pose_embedding: np.ndarray      # 384-dimensional pose features
    
    # Metadata
    audio_source: Optional[str] = None
    move_source: Optional[str] = None
    embedding_version: str = "1.0"


@dataclass
class SimilarityScore:
    """Container for similarity scoring results."""
    overall_score: float
    audio_similarity: float
    pose_similarity: float
    tempo_compatibility: float
    energy_alignment: float
    
    # Component weights used
    weights: Dict[str, float]


class FeatureFusion:
    """
    Feature fusion system that combines audio and pose features into unified embeddings
    for multi-modal similarity matching in Bachata choreography generation.
    """
    
    def __init__(self):
        """Initialize the feature fusion system."""
        self.audio_dim = 128
        self.pose_dim = 384
        self.combined_dim = 512
        
        logger.info(f"FeatureFusion initialized: {self.audio_dim}D audio + {self.pose_dim}D pose → {self.combined_dim}D combined")
    
    def create_audio_embedding(self, music_features: MusicFeatures) -> np.ndarray:
        """
        Create 128-dimensional audio embedding from music features using MFCC, Chroma, and Tonnetz.
        
        Args:
            music_features: Extracted music features from MusicAnalyzer
            
        Returns:
            128-dimensional audio feature vector
        """
        embedding = []
        
        # 1. MFCC features (40 dimensions)
        # Use mean and std of first 13 MFCCs + delta features
        mfcc_mean = np.mean(music_features.mfcc_features, axis=1)[:13]  # 13 dims
        mfcc_std = np.std(music_features.mfcc_features, axis=1)[:13]    # 13 dims
        
        # Delta MFCC (first derivative) - 14 dims
        mfcc_delta = np.diff(music_features.mfcc_features, axis=1)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)[:7]  # 7 dims
        mfcc_delta_std = np.std(mfcc_delta, axis=1)[:7]    # 7 dims
        
        embedding.extend(mfcc_mean.tolist())
        embedding.extend(mfcc_std.tolist())
        embedding.extend(mfcc_delta_mean.tolist())
        embedding.extend(mfcc_delta_std.tolist())
        
        # 2. Chroma features (24 dimensions)
        chroma_mean = np.mean(music_features.chroma_features, axis=1)  # 12 dims
        chroma_std = np.std(music_features.chroma_features, axis=1)    # 12 dims
        
        embedding.extend(chroma_mean.tolist())
        embedding.extend(chroma_std.tolist())
        
        # 3. Tonnetz features (12 dimensions)
        # Tonnetz represents harmonic relationships
        try:
            import librosa
            # Load audio again to compute tonnetz (this could be optimized)
            if hasattr(music_features, 'audio_path'):
                y, sr = librosa.load(music_features.audio_path, sr=22050)
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                tonnetz_mean = np.mean(tonnetz, axis=1)  # 6 dims
                tonnetz_std = np.std(tonnetz, axis=1)    # 6 dims
                
                embedding.extend(tonnetz_mean.tolist())
                embedding.extend(tonnetz_std.tolist())
            else:
                # Fallback: use chroma-derived harmonic features
                harmonic_features = self._derive_harmonic_features_from_chroma(music_features.chroma_features)
                embedding.extend(harmonic_features)
        except:
            # Fallback: use chroma-derived harmonic features
            harmonic_features = self._derive_harmonic_features_from_chroma(music_features.chroma_features)
            embedding.extend(harmonic_features)
        
        # 4. Spectral features (16 dimensions)
        spectral_centroid_mean = np.mean(music_features.spectral_centroid)
        spectral_centroid_std = np.std(music_features.spectral_centroid)
        spectral_centroid_range = np.max(music_features.spectral_centroid) - np.min(music_features.spectral_centroid)
        spectral_centroid_skew = self._calculate_skewness(music_features.spectral_centroid.flatten())
        
        zcr_mean = np.mean(music_features.zero_crossing_rate)
        zcr_std = np.std(music_features.zero_crossing_rate)
        
        rms_mean = np.mean(music_features.rms_energy)
        rms_std = np.std(music_features.rms_energy)
        rms_range = np.max(music_features.rms_energy) - np.min(music_features.rms_energy)
        
        # Harmonic-percussive separation features
        harmonic_ratio = np.mean(np.abs(music_features.harmonic_component)) / (
            np.mean(np.abs(music_features.harmonic_component)) + 
            np.mean(np.abs(music_features.percussive_component)) + 1e-6
        )
        
        # Energy dynamics
        energy_variance = np.var(music_features.energy_profile)
        energy_trend = self._calculate_trend(music_features.energy_profile)
        
        # Rhythm features
        rhythm_strength = music_features.rhythm_pattern_strength
        syncopation_level = music_features.syncopation_level
        
        # Tempo stability
        tempo_confidence = getattr(music_features, 'tempo_confidence', 0.8)
        tempo_normalized = music_features.tempo / 150.0  # Normalize to typical Bachata range
        
        spectral_features = [
            spectral_centroid_mean, spectral_centroid_std, spectral_centroid_range, spectral_centroid_skew,
            zcr_mean, zcr_std, rms_mean, rms_std, rms_range,
            harmonic_ratio, energy_variance, energy_trend,
            rhythm_strength, syncopation_level, tempo_confidence, tempo_normalized
        ]
        
        embedding.extend(spectral_features)
        
        # 5. Musical structure features (36 dimensions)
        structure_features = self._extract_structure_features(music_features)
        embedding.extend(structure_features)
        
        # Ensure exactly 128 dimensions
        embedding = np.array(embedding)
        if len(embedding) > self.audio_dim:
            embedding = embedding[:self.audio_dim]
        elif len(embedding) < self.audio_dim:
            # Pad with zeros if needed
            padding = np.zeros(self.audio_dim - len(embedding))
            embedding = np.concatenate([embedding, padding])
        
        # Normalize the embedding
        embedding = self._normalize_embedding(embedding)
        
        logger.debug(f"Created {len(embedding)}D audio embedding")
        return embedding
    
    def create_pose_embedding_aggregation(self, move_result: MoveAnalysisResult) -> np.ndarray:
        """
        Create pose feature aggregation from MediaPipe landmarks across video frames.
        
        Args:
            move_result: Complete move analysis result from MoveAnalyzer
            
        Returns:
            384-dimensional pose feature vector
        """
        # The MoveAnalyzer already creates a 384-dimensional pose embedding
        # We'll use that as the base and potentially enhance it
        base_embedding = move_result.pose_embedding
        
        if len(base_embedding) != self.pose_dim:
            logger.warning(f"Expected {self.pose_dim}D pose embedding, got {len(base_embedding)}D")
            # Resize if needed
            if len(base_embedding) > self.pose_dim:
                base_embedding = base_embedding[:self.pose_dim]
            else:
                padding = np.zeros(self.pose_dim - len(base_embedding))
                base_embedding = np.concatenate([base_embedding, padding])
        
        # Normalize the embedding
        pose_embedding = self._normalize_embedding(base_embedding)
        
        logger.debug(f"Created {len(pose_embedding)}D pose embedding")
        return pose_embedding
    
    def create_multimodal_embedding(self, 
                                  music_features: MusicFeatures, 
                                  move_result: MoveAnalysisResult) -> MultiModalEmbedding:
        """
        Create 512-dimensional combined feature vector from audio and pose features.
        
        Args:
            music_features: Extracted music features
            move_result: Complete move analysis result
            
        Returns:
            MultiModalEmbedding with combined 512D vector
        """
        # Create individual embeddings
        audio_embedding = self.create_audio_embedding(music_features)
        pose_embedding = self.create_pose_embedding_aggregation(move_result)
        
        # Combine embeddings using concatenation
        combined_embedding = np.concatenate([audio_embedding, pose_embedding])
        
        # Ensure exactly 512 dimensions
        if len(combined_embedding) != self.combined_dim:
            logger.warning(f"Combined embedding has {len(combined_embedding)} dims, expected {self.combined_dim}")
            if len(combined_embedding) > self.combined_dim:
                combined_embedding = combined_embedding[:self.combined_dim]
            else:
                padding = np.zeros(self.combined_dim - len(combined_embedding))
                combined_embedding = np.concatenate([combined_embedding, padding])
        
        # Final normalization
        combined_embedding = self._normalize_embedding(combined_embedding)
        
        logger.info(f"Created multimodal embedding: {len(audio_embedding)}D audio + {len(pose_embedding)}D pose = {len(combined_embedding)}D combined")
        
        return MultiModalEmbedding(
            combined_embedding=combined_embedding,
            audio_embedding=audio_embedding,
            pose_embedding=pose_embedding,
            audio_source=getattr(music_features, 'audio_path', None),
            move_source=move_result.video_path
        )
    
    def calculate_similarity(self, 
                           embedding1: MultiModalEmbedding, 
                           embedding2: MultiModalEmbedding,
                           weights: Optional[Dict[str, float]] = None) -> SimilarityScore:
        """
        Calculate similarity between two multimodal embeddings.
        
        Args:
            embedding1: First multimodal embedding
            embedding2: Second multimodal embedding
            weights: Optional custom weights for similarity components
            
        Returns:
            SimilarityScore with detailed similarity breakdown
        """
        if weights is None:
            weights = {
                'audio': 0.4,
                'pose': 0.4,
                'combined': 0.2
            }
        
        # Calculate cosine similarities
        audio_sim = self._cosine_similarity(embedding1.audio_embedding, embedding2.audio_embedding)
        pose_sim = self._cosine_similarity(embedding1.pose_embedding, embedding2.pose_embedding)
        combined_sim = self._cosine_similarity(embedding1.combined_embedding, embedding2.combined_embedding)
        
        # Calculate overall similarity
        overall_score = (
            weights['audio'] * audio_sim +
            weights['pose'] * pose_sim +
            weights['combined'] * combined_sim
        )
        
        return SimilarityScore(
            overall_score=overall_score,
            audio_similarity=audio_sim,
            pose_similarity=pose_sim,
            tempo_compatibility=0.0,  # Will be calculated in recommendation engine
            energy_alignment=0.0,     # Will be calculated in recommendation engine
            weights=weights
        )
    
    def test_embedding_quality(self, 
                             embeddings: List[MultiModalEmbedding],
                             labels: List[str]) -> Dict[str, float]:
        """
        Test embedding quality using similarity metrics between known compatible moves and music.
        
        Args:
            embeddings: List of multimodal embeddings to test
            labels: Corresponding labels for the embeddings
            
        Returns:
            Dictionary of quality metrics
        """
        if len(embeddings) < 2:
            return {"error": "Need at least 2 embeddings for quality testing"}
        
        metrics = {}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim_score = self.calculate_similarity(embeddings[i], embeddings[j])
                similarities.append(sim_score.overall_score)
        
        # Basic statistics
        metrics['mean_similarity'] = np.mean(similarities)
        metrics['std_similarity'] = np.std(similarities)
        metrics['min_similarity'] = np.min(similarities)
        metrics['max_similarity'] = np.max(similarities)
        
        # Embedding quality metrics
        # 1. Dimensionality utilization
        all_combined = np.vstack([emb.combined_embedding for emb in embeddings])
        non_zero_dims = np.count_nonzero(np.std(all_combined, axis=0) > 1e-6)
        metrics['dimensionality_utilization'] = non_zero_dims / self.combined_dim
        
        # 2. Embedding variance (higher is better for discrimination)
        metrics['embedding_variance'] = np.mean(np.var(all_combined, axis=0))
        
        # 3. Audio vs Pose embedding correlation
        audio_embeddings = np.vstack([emb.audio_embedding for emb in embeddings])
        pose_embeddings = np.vstack([emb.pose_embedding for emb in embeddings])
        
        audio_mean = np.mean(audio_embeddings, axis=1)
        pose_mean = np.mean(pose_embeddings, axis=1)
        
        if len(audio_mean) > 1 and len(pose_mean) > 1:
            correlation = np.corrcoef(audio_mean, pose_mean)[0, 1]
            metrics['audio_pose_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['audio_pose_correlation'] = 0.0
        
        logger.info(f"Embedding quality metrics: {metrics}")
        return metrics
    
    def _derive_harmonic_features_from_chroma(self, chroma_features: np.ndarray) -> List[float]:
        """Derive harmonic features from chroma when Tonnetz is not available."""
        # Calculate harmonic relationships from chroma
        harmonic_features = []
        
        # Circle of fifths progression
        fifths_circle = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C, G, D, A, E, B, F#, C#, G#, D#, A#, F
        
        chroma_mean = np.mean(chroma_features, axis=1)
        
        # Calculate progression strength along circle of fifths
        for i in range(6):  # 6 pairs for 12 dimensions
            idx1 = fifths_circle[i]
            idx2 = fifths_circle[i + 6]
            harmonic_features.append(chroma_mean[idx1] - chroma_mean[idx2])
        
        # Major/minor tendencies
        major_indices = [0, 4, 7]  # C, E, G
        minor_indices = [0, 3, 7]  # C, Eb, G
        
        major_strength = np.mean([chroma_mean[i] for i in major_indices])
        minor_strength = np.mean([chroma_mean[i] for i in minor_indices])
        
        harmonic_features.extend([major_strength, minor_strength])
        
        # Tritone relationships
        for i in range(4):
            tritone_strength = chroma_mean[i] * chroma_mean[(i + 6) % 12]
            harmonic_features.append(tritone_strength)
        
        return harmonic_features[:12]  # Ensure exactly 12 dimensions
    
    def _extract_structure_features(self, music_features: MusicFeatures) -> List[float]:
        """Extract musical structure features."""
        features = []
        
        # Section-based features
        sections = music_features.sections
        if sections:
            # Number of sections
            features.append(len(sections) / 10.0)  # Normalize
            
            # Section duration statistics
            durations = [s.end_time - s.start_time for s in sections]
            features.extend([
                np.mean(durations) / 60.0,  # Normalize to minutes
                np.std(durations) / 60.0,
                np.min(durations) / 60.0,
                np.max(durations) / 60.0
            ])
            
            # Energy progression
            energies = [s.energy_level for s in sections]
            features.extend([
                np.mean(energies),
                np.std(energies),
                np.max(energies) - np.min(energies)  # Energy range
            ])
            
            # Section type distribution
            section_types = [s.section_type for s in sections]
            type_counts = {
                'intro': section_types.count('intro'),
                'verse': section_types.count('verse'),
                'chorus': section_types.count('chorus'),
                'bridge': section_types.count('bridge'),
                'outro': section_types.count('outro')
            }
            
            total_sections = len(sections)
            for section_type in ['intro', 'verse', 'chorus', 'bridge', 'outro']:
                features.append(type_counts[section_type] / total_sections)
            
            # Tempo stability across sections
            if hasattr(sections[0], 'tempo_stability'):
                tempo_stabilities = [s.tempo_stability for s in sections]
                features.extend([
                    np.mean(tempo_stabilities),
                    np.std(tempo_stabilities)
                ])
            else:
                features.extend([0.8, 0.1])  # Default values
        else:
            # Default values when no sections detected
            features.extend([0.0] * 20)
        
        # Energy profile features
        energy_profile = music_features.energy_profile
        if energy_profile and len(energy_profile) > 1:
            features.extend([
                np.mean(energy_profile),
                np.std(energy_profile),
                np.max(energy_profile) - np.min(energy_profile),
                self._calculate_trend(energy_profile),
                self._calculate_energy_peaks(energy_profile)
            ])
        else:
            features.extend([0.0] * 5)
        
        # Beat and rhythm features
        beat_positions = music_features.beat_positions
        if beat_positions and len(beat_positions) > 1:
            beat_intervals = np.diff(beat_positions)
            features.extend([
                np.mean(beat_intervals),
                np.std(beat_intervals),
                np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)  # Coefficient of variation
            ])
        else:
            features.extend([0.5, 0.1, 0.2])  # Default values
        
        # Bachata-specific rhythm features
        features.extend([
            music_features.rhythm_pattern_strength,
            music_features.syncopation_level
        ])
        
        # Duration and tempo features
        features.extend([
            music_features.duration / 300.0,  # Normalize to 5 minutes
            music_features.tempo / 150.0      # Normalize to typical Bachata tempo
        ])
        
        # Pad or truncate to exactly 36 dimensions
        features = features[:36]
        while len(features) < 36:
            features.append(0.0)
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate linear trend in data."""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Simple linear regression slope
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope
    
    def _calculate_energy_peaks(self, energy_profile: List[float]) -> float:
        """Calculate number of energy peaks normalized by length."""
        if len(energy_profile) < 3:
            return 0.0
        
        energy = np.array(energy_profile)
        peaks = 0
        
        for i in range(1, len(energy) - 1):
            if energy[i] > energy[i-1] and energy[i] > energy[i+1]:
                peaks += 1
        
        return peaks / len(energy_profile)
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        else:
            return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        else:
            return 0.0