"""
Music analysis service using Librosa for audio feature extraction.
"""

import librosa
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

logger = logging.getLogger(__name__)


@dataclass
class MusicSection:
    """Represents a musical section with timing and characteristics."""
    start_time: float
    end_time: float
    section_type: str  # intro, verse, chorus, bridge, outro
    energy_level: float
    tempo_stability: float
    recommended_move_types: List[str]


@dataclass
class MusicFeatures:
    """Container for extracted music features."""
    tempo: float
    beat_positions: List[float]
    duration: float
    
    # Spectral features
    mfcc_features: np.ndarray
    chroma_features: np.ndarray
    spectral_centroid: np.ndarray
    zero_crossing_rate: np.ndarray
    
    # Energy and dynamics
    rms_energy: np.ndarray
    harmonic_component: np.ndarray
    percussive_component: np.ndarray
    
    # Derived features
    energy_profile: List[float]
    tempo_confidence: float
    
    # Musical structure
    sections: List[MusicSection]
    
    # Bachata-specific rhythm features
    rhythm_pattern_strength: float
    syncopation_level: float
    
    # Embedding for similarity matching
    audio_embedding: List[float]


class MusicAnalyzer:
    """Analyzes audio files to extract musical features for choreography generation."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the music analyzer.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_mfcc = 13
        
    def analyze_audio(self, audio_path: str) -> MusicFeatures:
        """
        Analyze an audio file and extract comprehensive musical features.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            MusicFeatures object containing all extracted features
        """
        logger.info(f"Starting audio analysis for: {audio_path}")
        
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        
        logger.info(f"Loaded audio: duration={duration:.2f}s, sample_rate={sr}")
        
        # Extract tempo and beat positions
        tempo, beats = self._extract_tempo_and_beats(y, sr)
        
        # Extract spectral features
        mfcc_features = self._extract_mfcc(y, sr)
        chroma_features = self._extract_chroma(y, sr)
        spectral_centroid = self._extract_spectral_centroid(y, sr)
        zero_crossing_rate = self._extract_zero_crossing_rate(y)
        
        # Extract energy and dynamics
        rms_energy = self._extract_rms_energy(y, sr)
        harmonic_component, percussive_component = self._separate_harmonic_percussive(y)
        
        # Calculate energy profile
        energy_profile = self._calculate_energy_profile(rms_energy)
        
        # Perform musical structure segmentation
        sections = self._segment_musical_structure(y, sr, beats, energy_profile)
        
        # Extract Bachata-specific rhythm features
        rhythm_strength, syncopation = self._extract_bachata_rhythm_features(
            y, sr, beats, percussive_component
        )
        
        # Generate enhanced audio embedding with rhythm characteristics
        audio_embedding = self._generate_audio_embedding(
            mfcc_features, chroma_features, spectral_centroid, tempo,
            rhythm_strength, syncopation, energy_profile
        )
        
        logger.info(f"Analysis complete: tempo={tempo:.1f} BPM, {len(beats)} beats detected")
        
        return MusicFeatures(
            tempo=tempo,
            beat_positions=beats.tolist(),
            duration=duration,
            mfcc_features=mfcc_features,
            chroma_features=chroma_features,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
            rms_energy=rms_energy,
            harmonic_component=harmonic_component,
            percussive_component=percussive_component,
            energy_profile=energy_profile,
            tempo_confidence=0.0,  # Will be calculated in future iterations
            sections=sections,
            rhythm_pattern_strength=rhythm_strength,
            syncopation_level=syncopation,
            audio_embedding=audio_embedding
        )
    
    def _extract_tempo_and_beats(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        """Extract tempo and beat positions from audio."""
        # Use onset strength for better beat tracking in Bachata
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Extract tempo with multiple candidates for better accuracy
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=self.hop_length,
            units='time'
        )
        
        # Validate tempo range for Bachata (typically 90-150 BPM)
        if tempo < 80 or tempo > 160:
            logger.warning(f"Detected tempo {tempo:.1f} BPM is outside typical Bachata range (90-150 BPM)")
        
        return float(tempo), beats
    
    def _extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract Mel-frequency cepstral coefficients."""
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        return mfcc
    
    def _extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract chroma features for harmonic content analysis."""
        chroma = librosa.feature.chroma_stft(
            y=y, 
            sr=sr,
            hop_length=self.hop_length
        )
        return chroma
    
    def _extract_spectral_centroid(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral centroid for brightness analysis."""
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, 
            sr=sr,
            hop_length=self.hop_length
        )
        return spectral_centroid
    
    def _extract_zero_crossing_rate(self, y: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate for texture analysis."""
        zcr = librosa.feature.zero_crossing_rate(
            y, 
            hop_length=self.hop_length
        )
        return zcr
    
    def _extract_rms_energy(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract RMS energy for dynamics analysis."""
        rms = librosa.feature.rms(
            y=y,
            hop_length=self.hop_length
        )
        return rms
    
    def _separate_harmonic_percussive(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separate harmonic and percussive components."""
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        return y_harmonic, y_percussive
    
    def _calculate_energy_profile(self, rms_energy: np.ndarray) -> List[float]:
        """Calculate energy profile over time for section analysis."""
        # Smooth the energy profile using a moving average
        window_size = 10
        smoothed_energy = np.convolve(
            rms_energy.flatten(), 
            np.ones(window_size) / window_size, 
            mode='same'
        )
        return smoothed_energy.tolist()
    
    def _segment_musical_structure(
        self, 
        y: np.ndarray, 
        sr: int, 
        beats: np.ndarray,
        energy_profile: List[float]
    ) -> List[MusicSection]:
        """
        Enhanced musical structure segmentation optimized for Bachata songs.
        Uses multiple features including chroma, tempo stability, and energy patterns.
        """
        try:
            duration = len(y) / sr
            
            # Use multiple features for better segmentation
            # 1. Chroma features for harmonic analysis
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            
            # 2. MFCC for timbral analysis
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # 3. Spectral centroid for brightness changes
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            
            # Combine features for segmentation
            combined_features = np.vstack([
                chroma,
                mfcc[:5],  # Use first 5 MFCCs
                spectral_centroid
            ])
            
            # Compute recurrence matrix with combined features
            R = librosa.segment.recurrence_matrix(
                combined_features, 
                width=43,  # ~2 seconds at 22050 Hz
                mode='affinity',
                sym=True
            )
            
            # Apply Gaussian filter to smooth the recurrence matrix if scipy is available
            if ndimage is not None:
                R_filtered = ndimage.gaussian_filter(R, sigma=1.0)
            else:
                R_filtered = R  # Use unfiltered matrix if scipy not available
            
            # Detect segment boundaries with adaptive number of segments
            # Bachata songs typically have 4-8 distinct sections
            target_sections = max(4, min(8, int(duration / 30)))  # ~30 seconds per section
            boundaries = librosa.segment.agglomerative(R_filtered, k=target_sections)
            boundary_times = librosa.frames_to_time(boundaries, sr=sr, hop_length=self.hop_length)
            
            # Ensure we have start and end boundaries
            if boundary_times[0] > 0:
                boundary_times = np.insert(boundary_times, 0, 0.0)
            if boundary_times[-1] < duration:
                boundary_times = np.append(boundary_times, duration)
            
            # Create sections with enhanced analysis
            sections = []
            for i in range(len(boundary_times) - 1):
                start_time = boundary_times[i]
                end_time = boundary_times[i + 1]
                section_duration = end_time - start_time
                
                # Calculate section characteristics
                section_energy, tempo_stability = self._analyze_section_characteristics(
                    y, sr, start_time, end_time, beats, energy_profile
                )
                
                # Enhanced section classification
                section_type = self._classify_section_type_enhanced(
                    i, len(boundary_times) - 1, section_energy, start_time, 
                    section_duration, duration, tempo_stability
                )
                
                # Recommend move types based on enhanced analysis
                move_types = self._recommend_move_types_for_section_enhanced(
                    section_type, section_energy, tempo_stability, section_duration
                )
                
                sections.append(MusicSection(
                    start_time=start_time,
                    end_time=end_time,
                    section_type=section_type,
                    energy_level=section_energy,
                    tempo_stability=tempo_stability,
                    recommended_move_types=move_types
                ))
            
            logger.info(f"Detected {len(sections)} musical sections using enhanced segmentation")
            return sections
            
        except Exception as e:
            logger.warning(f"Enhanced structure segmentation failed: {e}. Using fallback method.")
            return self._create_simple_sections(len(y) / sr, energy_profile)
    
    def _analyze_section_characteristics(
        self, 
        y: np.ndarray, 
        sr: int, 
        start_time: float, 
        end_time: float,
        beats: np.ndarray,
        energy_profile: List[float]
    ) -> Tuple[float, float]:
        """
        Analyze characteristics of a musical section.
        
        Returns:
            Tuple of (section_energy, tempo_stability)
        """
        # Calculate energy for this section
        start_idx = int(start_time * len(energy_profile) / (len(y) / sr))
        end_idx = int(end_time * len(energy_profile) / (len(y) / sr))
        start_idx = max(0, min(start_idx, len(energy_profile) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(energy_profile)))
        
        section_energy = np.mean(energy_profile[start_idx:end_idx])
        
        # Calculate tempo stability within this section
        section_beats = beats[(beats >= start_time) & (beats <= end_time)]
        if len(section_beats) > 3:
            beat_intervals = np.diff(section_beats)
            tempo_stability = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
            tempo_stability = max(0.0, min(1.0, tempo_stability))
        else:
            tempo_stability = 0.5  # Default for short sections
        
        return section_energy, tempo_stability
    
    def _classify_section_type_enhanced(
        self, 
        section_idx: int, 
        total_sections: int, 
        energy: float, 
        start_time: float,
        duration: float,
        total_duration: float,
        tempo_stability: float
    ) -> str:
        """Enhanced section classification using multiple characteristics."""
        # Position-based classification
        relative_position = start_time / total_duration
        
        # First section is usually intro
        if section_idx == 0:
            return "intro"
        
        # Last section is usually outro
        elif section_idx == total_sections - 1:
            return "outro"
        
        # Very short sections are often bridges or transitions
        elif duration < 15:
            return "bridge"
        
        # High energy sections in the middle are likely chorus
        elif energy > 0.25 and 0.2 < relative_position < 0.8:
            return "chorus"
        
        # Sections with high tempo stability and moderate energy are verses
        elif tempo_stability > 0.7 and 0.15 < energy < 0.3:
            return "verse"
        
        # Sections in the middle with lower energy might be pre-chorus or bridge
        elif energy < 0.2 and 0.3 < relative_position < 0.7:
            return "bridge"
        
        # Default classification based on energy
        elif energy > 0.25:
            return "chorus"
        else:
            return "verse"
    
    def _classify_section_type(
        self, 
        section_idx: int, 
        total_sections: int, 
        energy: float, 
        start_time: float,
        duration: float
    ) -> str:
        """Legacy method for backward compatibility."""
        return self._classify_section_type_enhanced(
            section_idx, total_sections, energy, start_time, 
            duration, 300.0, 0.8  # Default values
        )
    
    def _recommend_move_types_for_section_enhanced(
        self, 
        section_type: str, 
        energy: float, 
        tempo_stability: float,
        duration: float
    ) -> List[str]:
        """Enhanced move recommendation based on multiple section characteristics."""
        moves = []
        
        if section_type == "intro":
            # Intro: Start with basic moves to establish connection
            moves = ["basic_step", "side_step"]
            if duration > 20:  # Longer intros can include more variety
                moves.append("forward_backward")
        
        elif section_type == "verse":
            # Verse: Moderate complexity, focus on connection and flow
            moves = ["basic_step", "forward_backward", "side_step"]
            if energy > 0.2:
                moves.extend(["cross_body_lead", "hammerlock"])
            if tempo_stability > 0.8:  # Stable tempo allows for more complex moves
                moves.append("lady_left_turn")
        
        elif section_type == "chorus":
            # Chorus: High energy, showcase moves
            moves = ["cross_body_lead", "lady_left_turn", "lady_right_turn"]
            if energy > 0.3:
                moves.extend(["copa", "turns", "styling"])
            if energy > 0.4:  # Very high energy
                moves.extend(["advanced_turns", "body_roll"])
        
        elif section_type == "bridge":
            # Bridge: Intimate, close connection moves
            moves = ["basic_step", "arm_styling"]
            if duration > 15:
                moves.extend(["body_roll", "dips"])
            if energy < 0.15:  # Very low energy bridges
                moves.append("body_roll")
        
        elif section_type == "outro":
            # Outro: Graceful ending moves
            moves = ["basic_step", "dips"]
            if duration > 10:
                moves.extend(["arm_styling", "body_roll"])
        
        else:
            # Default fallback
            moves = ["basic_step", "side_step", "forward_backward"]
        
        # Add energy-based adjustments
        if energy > 0.35:  # Very high energy
            if "styling" not in moves:
                moves.append("styling")
        elif energy < 0.15:  # Very low energy
            moves = [move for move in moves if move in ["basic_step", "body_roll", "arm_styling"]]
        
        return moves
    
    def _recommend_move_types_for_section(self, section_type: str, energy: float) -> List[str]:
        """Legacy method for backward compatibility."""
        return self._recommend_move_types_for_section_enhanced(
            section_type, energy, 0.8, 30.0  # Default values
        )
    
    def _create_simple_sections(self, duration: float, energy_profile: List[float]) -> List[MusicSection]:
        """Create simple time-based sections as fallback."""
        sections = []
        section_duration = duration / 4  # Divide into 4 equal sections
        
        for i in range(4):
            start_time = i * section_duration
            end_time = (i + 1) * section_duration
            
            # Calculate energy for this time segment
            start_idx = int(i * len(energy_profile) / 4)
            end_idx = int((i + 1) * len(energy_profile) / 4)
            section_energy = np.mean(energy_profile[start_idx:end_idx])
            
            section_type = ["intro", "verse", "chorus", "outro"][i]
            move_types = self._recommend_move_types_for_section(section_type, section_energy)
            
            sections.append(MusicSection(
                start_time=start_time,
                end_time=end_time,
                section_type=section_type,
                energy_level=section_energy,
                tempo_stability=0.8,
                recommended_move_types=move_types
            ))
        
        return sections
    
    def _extract_bachata_rhythm_features(
        self, 
        y: np.ndarray, 
        sr: int, 
        beats: np.ndarray,
        percussive: np.ndarray
    ) -> Tuple[float, float]:
        """
        Enhanced Bachata-specific rhythm analysis.
        
        Analyzes multiple rhythm characteristics:
        - Beat regularity and 4/4 pattern strength
        - Syncopation patterns typical in Bachata
        - Percussion emphasis on beats 1 and 3
        - Guitar pattern recognition
        
        Returns:
            Tuple of (rhythm_pattern_strength, syncopation_level)
        """
        if len(beats) < 8:  # Need sufficient beats for analysis
            return 0.5, 0.5
        
        # 1. Beat regularity analysis
        beat_intervals = np.diff(beats)
        mean_interval = np.mean(beat_intervals)
        interval_std = np.std(beat_intervals)
        
        # Bachata should have very regular beat intervals
        interval_consistency = 1.0 - min(1.0, interval_std / mean_interval)
        
        # 2. 4/4 pattern strength analysis
        # Bachata typically emphasizes beats 1 and 3
        four_four_strength = self._analyze_four_four_pattern(y, sr, beats, percussive)
        
        # 3. Bachata-specific rhythm pattern detection
        bachata_pattern_strength = self._detect_bachata_pattern(y, sr, beats)
        
        # Combine for overall rhythm strength
        rhythm_strength = (interval_consistency * 0.4 + 
                          four_four_strength * 0.3 + 
                          bachata_pattern_strength * 0.3)
        rhythm_strength = max(0.0, min(1.0, rhythm_strength))
        
        # 4. Enhanced syncopation analysis
        syncopation = self._analyze_bachata_syncopation(y, sr, beats, percussive)
        
        return rhythm_strength, syncopation
    
    def _analyze_four_four_pattern(
        self, 
        y: np.ndarray, 
        sr: int, 
        beats: np.ndarray,
        percussive: np.ndarray
    ) -> float:
        """Analyze 4/4 pattern strength typical in Bachata."""
        try:
            # Calculate RMS energy around each beat
            beat_energies = []
            window_size = int(0.1 * sr)  # 100ms window around each beat
            
            for beat_time in beats:
                beat_sample = int(beat_time * sr)
                start_sample = max(0, beat_sample - window_size // 2)
                end_sample = min(len(percussive), beat_sample + window_size // 2)
                
                if end_sample > start_sample:
                    beat_energy = np.sqrt(np.mean(percussive[start_sample:end_sample] ** 2))
                    beat_energies.append(beat_energy)
            
            if len(beat_energies) < 8:
                return 0.5
            
            # Group beats into measures of 4
            measures = []
            for i in range(0, len(beat_energies) - 3, 4):
                measure = beat_energies[i:i+4]
                if len(measure) == 4:
                    measures.append(measure)
            
            if len(measures) < 2:
                return 0.5
            
            # Analyze pattern: beats 1 and 3 should be stronger
            pattern_scores = []
            for measure in measures:
                # Check if beats 1 and 3 are stronger than 2 and 4
                strong_beats = (measure[0] + measure[2]) / 2  # beats 1 and 3
                weak_beats = (measure[1] + measure[3]) / 2    # beats 2 and 4
                
                if strong_beats > 0:
                    pattern_score = min(1.0, strong_beats / (weak_beats + 1e-6))
                    pattern_scores.append(pattern_score)
            
            return np.mean(pattern_scores) if pattern_scores else 0.5
            
        except Exception as e:
            logger.warning(f"4/4 pattern analysis failed: {e}")
            return 0.5
    
    def _detect_bachata_pattern(self, y: np.ndarray, sr: int, beats: np.ndarray) -> float:
        """Detect Bachata-specific rhythm patterns."""
        try:
            # Bachata often has a distinctive guitar pattern
            # Analyze spectral characteristics around guitar frequency range (80-1000 Hz)
            
            # Use constant-Q transform for better frequency resolution
            C = librosa.cqt(y, sr=sr, hop_length=self.hop_length, 
                           fmin=librosa.note_to_hz('E2'), n_bins=60)
            
            # Focus on guitar frequency range
            guitar_range = C[10:40]  # Approximate guitar range in CQT
            guitar_energy = np.mean(np.abs(guitar_range), axis=0)
            
            # Analyze periodicity in guitar energy
            # Bachata guitar often has a 2-beat pattern
            beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=self.hop_length)
            
            if len(beat_frames) < 4:
                return 0.5
            
            # Sample guitar energy at beat positions
            beat_guitar_energies = []
            for frame in beat_frames:
                if frame < len(guitar_energy):
                    beat_guitar_energies.append(guitar_energy[frame])
            
            if len(beat_guitar_energies) < 4:
                return 0.5
            
            # Look for alternating pattern (typical in Bachata)
            alternation_score = 0.0
            for i in range(0, len(beat_guitar_energies) - 1, 2):
                if i + 1 < len(beat_guitar_energies):
                    # Check for alternating high-low pattern
                    diff = abs(beat_guitar_energies[i] - beat_guitar_energies[i + 1])
                    alternation_score += diff
            
            # Normalize by number of pairs
            if len(beat_guitar_energies) > 1:
                alternation_score /= (len(beat_guitar_energies) // 2)
                return min(1.0, alternation_score * 2)  # Scale to 0-1 range
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Bachata pattern detection failed: {e}")
            return 0.5
    
    def _analyze_bachata_syncopation(
        self, 
        y: np.ndarray, 
        sr: int, 
        beats: np.ndarray,
        percussive: np.ndarray
    ) -> float:
        """Enhanced syncopation analysis for Bachata characteristics."""
        try:
            # Detect onsets with higher sensitivity for Bachata
            onsets = librosa.onset.onset_detect(
                y=percussive, 
                sr=sr, 
                units='time',
                backtrack=True,
                delta=0.02,  # Lower threshold for subtle onsets
                wait=0.05    # Minimum time between onsets
            )
            
            if len(onsets) == 0:
                return 0.3  # Default moderate syncopation
            
            # Analyze onset patterns relative to beats
            syncopation_features = []
            
            # 1. Off-beat onset ratio
            off_beat_onsets = 0
            for onset in onsets:
                closest_beat_distance = min(abs(onset - beat) for beat in beats)
                if closest_beat_distance > 0.08:  # More sensitive threshold
                    off_beat_onsets += 1
            
            off_beat_ratio = off_beat_onsets / len(onsets)
            syncopation_features.append(off_beat_ratio)
            
            # 2. Anticipation patterns (onsets slightly before beats)
            anticipation_count = 0
            for beat in beats:
                # Look for onsets 0.05-0.15 seconds before each beat
                anticipating_onsets = [o for o in onsets 
                                     if beat - 0.15 < o < beat - 0.05]
                if anticipating_onsets:
                    anticipation_count += 1
            
            anticipation_ratio = anticipation_count / len(beats) if beats.size > 0 else 0
            syncopation_features.append(anticipation_ratio)
            
            # 3. Weak beat emphasis (onsets on beats 2 and 4)
            weak_beat_emphasis = 0
            for i, beat in enumerate(beats):
                beat_position = i % 4  # Position within 4-beat measure
                if beat_position in [1, 3]:  # Beats 2 and 4 (0-indexed)
                    nearby_onsets = [o for o in onsets if abs(o - beat) < 0.05]
                    if nearby_onsets:
                        weak_beat_emphasis += 1
            
            weak_beat_ratio = weak_beat_emphasis / (len(beats) // 2) if len(beats) > 1 else 0
            syncopation_features.append(weak_beat_ratio)
            
            # Combine syncopation features
            syncopation_level = np.mean(syncopation_features)
            return max(0.0, min(1.0, syncopation_level))
            
        except Exception as e:
            logger.warning(f"Enhanced syncopation analysis failed: {e}")
            return 0.3
    
    def _generate_audio_embedding(
        self, 
        mfcc: np.ndarray, 
        chroma: np.ndarray, 
        spectral_centroid: np.ndarray,
        tempo: float,
        rhythm_strength: float,
        syncopation: float,
        energy_profile: List[float]
    ) -> List[float]:
        """
        Generate an enhanced 128-dimensional audio embedding optimized for Bachata similarity matching.
        
        Feature distribution:
        - MFCC statistics: 26 dimensions (timbral characteristics)
        - Chroma statistics: 24 dimensions (harmonic content)
        - Spectral features: 8 dimensions (brightness and texture)
        - Rhythm features: 20 dimensions (Bachata-specific patterns)
        - Energy dynamics: 15 dimensions (song structure and dynamics)
        - Tempo characteristics: 15 dimensions (tempo-related features)
        - Musical structure: 20 dimensions (section-based features)
        """
        embedding = []
        
        # 1. MFCC statistics (26 dimensions) - Timbral characteristics
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        embedding.extend(mfcc_mean.tolist())
        embedding.extend(mfcc_std.tolist())
        
        # 2. Chroma statistics (24 dimensions) - Harmonic content
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        embedding.extend(chroma_mean.tolist())
        embedding.extend(chroma_std.tolist())
        
        # 3. Enhanced spectral features (8 dimensions)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        sc_range = np.max(spectral_centroid) - np.min(spectral_centroid)
        sc_skewness = self._calculate_skewness(spectral_centroid.flatten())
        
        # Spectral rolloff and bandwidth for additional texture info
        spectral_rolloff = np.mean(spectral_centroid) / 11025.0  # Normalized
        spectral_spread = sc_std / (sc_mean + 1e-6)  # Relative spread
        
        embedding.extend([
            sc_mean / 11025.0,  # Normalized spectral centroid mean
            sc_std / 11025.0,   # Normalized spectral centroid std
            sc_range / 11025.0, # Normalized spectral centroid range
            sc_skewness,        # Spectral centroid skewness
            spectral_rolloff,   # Spectral rolloff
            spectral_spread,    # Spectral spread
            min(1.0, sc_mean / 2000.0),  # Brightness indicator
            1.0 if sc_mean > 1500 else 0.0  # High brightness flag
        ])
        
        # 4. Enhanced rhythm features (20 dimensions) - Bachata-specific
        # Basic rhythm features
        normalized_tempo = max(0.0, min(1.0, (tempo - 60) / 120))
        tempo_bachata_score = self._calculate_bachata_tempo_score(tempo)
        
        # Advanced rhythm characteristics
        rhythm_complexity = rhythm_strength * syncopation
        rhythm_stability = rhythm_strength * (1.0 - syncopation)
        
        # Tempo-based rhythm features
        slow_bachata_score = max(0.0, 1.0 - abs(tempo - 100) / 20.0)
        medium_bachata_score = max(0.0, 1.0 - abs(tempo - 120) / 15.0)
        fast_bachata_score = max(0.0, 1.0 - abs(tempo - 140) / 20.0)
        
        embedding.extend([
            normalized_tempo,
            rhythm_strength,
            syncopation,
            rhythm_complexity,
            rhythm_stability,
            tempo_bachata_score,
            slow_bachata_score,
            medium_bachata_score,
            fast_bachata_score,
            1.0 if 90 <= tempo <= 150 else 0.0,  # Bachata range flag
            (tempo % 4) / 4.0,  # Tempo modulo for micro-rhythm
            rhythm_strength * normalized_tempo,  # Combined rhythm-tempo
            syncopation * (1.0 - abs(tempo - 120) / 60.0),  # Adjusted syncopation
            1.0 if rhythm_strength > 0.8 else 0.0,  # High rhythm strength flag
            1.0 if syncopation > 0.5 else 0.0,  # High syncopation flag
            abs(tempo - 120) / 60.0,  # Distance from ideal Bachata tempo
            min(1.0, tempo / 100.0) if tempo < 100 else min(1.0, 200.0 / tempo),  # Tempo extremeness
            rhythm_strength * (1.0 if 110 <= tempo <= 130 else 0.5),  # Optimal range bonus
            syncopation * tempo_bachata_score,  # Syncopation in Bachata context
            (rhythm_strength + (1.0 - syncopation)) / 2.0  # Overall rhythm quality
        ])
        
        # 5. Energy dynamics (15 dimensions)
        if energy_profile:
            energy_array = np.array(energy_profile)
            energy_mean = np.mean(energy_array)
            energy_std = np.std(energy_array)
            
            # Energy distribution features
            energy_percentiles = [
                np.percentile(energy_array, 10),
                np.percentile(energy_array, 25),
                np.percentile(energy_array, 50),
                np.percentile(energy_array, 75),
                np.percentile(energy_array, 90)
            ]
            
            # Energy dynamics
            energy_range = np.max(energy_array) - np.min(energy_array)
            energy_variance = np.var(energy_array)
            high_energy_ratio = np.sum(energy_array > energy_mean) / len(energy_array)
            energy_skewness = self._calculate_skewness(energy_array)
            
            # Energy build patterns (typical in Bachata)
            energy_trend = self._calculate_energy_trend(energy_array)
            
            embedding.extend([
                energy_mean,
                energy_std,
                energy_range,
                energy_variance,
                high_energy_ratio,
                energy_skewness,
                energy_trend,
                *energy_percentiles,  # 5 dimensions
                1.0 if energy_mean > 0.3 else 0.0,  # High energy flag
                1.0 if energy_std > 0.1 else 0.0,   # Dynamic energy flag
                energy_mean * rhythm_strength  # Energy-rhythm interaction
            ])
        else:
            embedding.extend([0.0] * 15)
        
        # 6. Tempo characteristics (15 dimensions)
        tempo_features = self._generate_tempo_features(tempo, rhythm_strength, syncopation)
        embedding.extend(tempo_features)
        
        # 7. Musical structure features (20 dimensions)
        structure_features = self._generate_structure_features(
            energy_profile, tempo, rhythm_strength
        )
        embedding.extend(structure_features)
        
        # Ensure exactly 128 dimensions
        current_length = len(embedding)
        if current_length < 128:
            embedding.extend([0.0] * (128 - current_length))
        elif current_length > 128:
            embedding = embedding[:128]
        
        # Normalize embedding to prevent any single feature from dominating
        embedding = self._normalize_embedding(embedding)
        
        return embedding
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of a data array."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_bachata_tempo_score(self, tempo: float) -> float:
        """Calculate how well the tempo fits Bachata characteristics."""
        # Bachata sweet spots: 100 BPM (romantic), 120 BPM (classic), 140 BPM (modern)
        sweet_spots = [100, 120, 140]
        scores = [max(0.0, 1.0 - abs(tempo - spot) / 25.0) for spot in sweet_spots]
        return max(scores)
    
    def _calculate_energy_trend(self, energy_array: np.ndarray) -> float:
        """Calculate overall energy trend (building vs. declining)."""
        if len(energy_array) < 3:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(energy_array))
        slope = np.polyfit(x, energy_array, 1)[0]
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, slope * len(energy_array)))
    
    def _generate_tempo_features(
        self, 
        tempo: float, 
        rhythm_strength: float, 
        syncopation: float
    ) -> List[float]:
        """Generate tempo-related features."""
        features = []
        
        # Basic tempo features
        normalized_tempo = (tempo - 60) / 120
        tempo_log = np.log(tempo / 60.0)  # Log-scaled tempo
        
        # Tempo category features
        features.extend([
            1.0 if tempo < 100 else 0.0,    # Slow
            1.0 if 100 <= tempo < 120 else 0.0,  # Medium-slow
            1.0 if 120 <= tempo < 140 else 0.0,  # Medium-fast
            1.0 if tempo >= 140 else 0.0,   # Fast
            normalized_tempo,
            tempo_log,
            tempo / 60.0,  # Tempo in Hz
            1.0 / (tempo / 60.0),  # Period
            tempo * rhythm_strength,  # Effective tempo
            tempo * (1.0 - syncopation),  # Stable tempo component
            abs(tempo - 120),  # Distance from standard
            (tempo - 90) / 60.0,  # Bachata range position
            min(1.0, tempo / 90.0) if tempo < 90 else min(1.0, 180.0 / tempo),  # Extremeness
            1.0 if 110 <= tempo <= 130 else 0.0,  # Optimal range
            self._calculate_bachata_tempo_score(tempo)  # Bachata fit score
        ])
        
        return features
    
    def _generate_structure_features(
        self, 
        energy_profile: List[float], 
        tempo: float,
        rhythm_strength: float
    ) -> List[float]:
        """Generate musical structure-related features."""
        features = []
        
        if energy_profile and len(energy_profile) > 10:
            energy_array = np.array(energy_profile)
            
            # Structural features
            # Divide into sections and analyze
            n_sections = 4
            section_size = len(energy_array) // n_sections
            section_energies = []
            
            for i in range(n_sections):
                start_idx = i * section_size
                end_idx = (i + 1) * section_size if i < n_sections - 1 else len(energy_array)
                section_energy = np.mean(energy_array[start_idx:end_idx])
                section_energies.append(section_energy)
            
            # Section-based features
            features.extend(section_energies)  # 4 dimensions
            
            # Section relationships
            intro_outro_ratio = section_energies[0] / (section_energies[-1] + 1e-6)
            middle_energy = np.mean(section_energies[1:-1])
            edge_energy = (section_energies[0] + section_energies[-1]) / 2
            middle_edge_ratio = middle_energy / (edge_energy + 1e-6)
            
            features.extend([
                intro_outro_ratio,
                middle_edge_ratio,
                max(section_energies) / (min(section_energies) + 1e-6),  # Dynamic range
                np.std(section_energies),  # Section variability
                1.0 if section_energies[1] > section_energies[0] else 0.0,  # Build from intro
                1.0 if section_energies[-1] < section_energies[-2] else 0.0,  # Fade to outro
            ])
            
            # Energy pattern features
            # Look for typical song patterns
            has_buildup = 1.0 if any(section_energies[i+1] > section_energies[i] * 1.2 
                                   for i in range(len(section_energies)-1)) else 0.0
            has_breakdown = 1.0 if any(section_energies[i+1] < section_energies[i] * 0.8 
                                     for i in range(len(section_energies)-1)) else 0.0
            
            features.extend([
                has_buildup,
                has_breakdown,
                tempo * middle_energy,  # Tempo-energy interaction
                rhythm_strength * np.std(section_energies),  # Rhythm-structure interaction
                len(energy_profile) / 1000.0,  # Duration proxy
                np.mean(energy_array) * tempo / 120.0,  # Normalized energy-tempo
                1.0 if max(section_energies) > 0.4 else 0.0,  # Has high energy section
                1.0 if min(section_energies) < 0.1 else 0.0,  # Has low energy section
            ])
        else:
            # Default values when energy profile is not available
            features.extend([0.0] * 20)
        
        return features
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to prevent feature dominance."""
        embedding_array = np.array(embedding)
        
        # Replace any infinite or NaN values
        embedding_array = np.nan_to_num(embedding_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply L2 normalization to the entire embedding
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        
        return embedding_array.tolist()
    
    def validate_tempo_accuracy(self, detected_tempo: float, expected_tempo: float = None) -> bool:
        """
        Validate tempo detection accuracy for Bachata music.
        
        Args:
            detected_tempo: The tempo detected by the analyzer
            expected_tempo: Optional expected tempo for validation
            
        Returns:
            True if tempo is within acceptable range for Bachata
        """
        # Bachata typically ranges from 90-150 BPM
        if not (90 <= detected_tempo <= 150):
            logger.warning(f"Detected tempo {detected_tempo:.1f} BPM outside Bachata range")
            return False
            
        if expected_tempo:
            # Allow 5% tolerance for tempo matching
            tolerance = expected_tempo * 0.05
            if abs(detected_tempo - expected_tempo) > tolerance:
                logger.warning(
                    f"Tempo mismatch: detected {detected_tempo:.1f}, "
                    f"expected {expected_tempo:.1f} (tolerance: ±{tolerance:.1f})"
                )
                return False
                
        return True