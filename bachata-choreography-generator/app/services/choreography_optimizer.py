"""
Diversity selection and choreography flow optimization system.
Implements diversity selection algorithm, transition compatibility matrix,
sequence optimization using dynamic programming, and musical structure awareness.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
import logging

from .recommendation_engine import MoveCandidate, RecommendationScore
from .music_analyzer import MusicFeatures, MusicSection

logger = logging.getLogger(__name__)


@dataclass
class TransitionScore:
    """Container for transition compatibility scoring between two moves."""
    from_move: str
    to_move: str
    compatibility_score: float
    
    # Detailed scoring components
    pose_similarity: float
    movement_flow: float
    energy_continuity: float
    difficulty_progression: float
    
    # Transition characteristics
    is_smooth: bool
    requires_pause: bool
    transition_time: float  # seconds


@dataclass
class ChoreographySequence:
    """Container for optimized choreography sequence."""
    moves: List[MoveCandidate]
    total_duration: float
    total_score: float
    
    # Sequence characteristics
    diversity_score: float
    flow_score: float
    musical_alignment_score: float
    
    # Detailed breakdown
    transition_scores: List[TransitionScore]
    section_alignment: List[Tuple[MusicSection, List[MoveCandidate]]]
    
    # Optimization metadata
    optimization_method: str
    iterations: int


@dataclass
class OptimizationRequest:
    """Container for choreography optimization request parameters."""
    music_features: MusicFeatures
    candidate_moves: List[RecommendationScore]
    target_duration: float
    
    # Optimization parameters
    diversity_weight: float = 0.3
    flow_weight: float = 0.4
    musical_alignment_weight: float = 0.3
    
    # Constraints
    max_repetitions: int = 2  # Maximum times a move can be repeated
    min_diversity_threshold: float = 0.6  # Minimum diversity score
    transition_time_budget: float = 2.0  # Total time budget for transitions


class ChoreographyOptimizer:
    """
    Choreography flow optimization system that implements:
    - Diversity selection algorithm to avoid repetitive move sequences
    - Transition compatibility matrix between all move pairs
    - Sequence optimization using dynamic programming
    - Musical structure awareness for move complexity alignment
    """
    
    def __init__(self):
        """Initialize the choreography optimizer."""
        self.transition_cache = {}  # Cache for transition compatibility scores
        
        # Transition scoring weights
        self.transition_weights = {
            'pose_similarity': 0.25,
            'movement_flow': 0.35,
            'energy_continuity': 0.25,
            'difficulty_progression': 0.15
        }
        
        logger.info("ChoreographyOptimizer initialized with transition weights: pose=25%, flow=35%, energy=25%, difficulty=15%")
    
    def optimize_choreography(self, request: OptimizationRequest) -> ChoreographySequence:
        """
        Optimize choreography sequence using diversity selection and flow optimization.
        
        Args:
            request: Optimization request with music features and candidate moves
            
        Returns:
            ChoreographySequence with optimized move sequence
        """
        logger.info(f"Optimizing choreography for {len(request.candidate_moves)} candidates, target duration: {request.target_duration:.1f}s")
        
        # Step 1: Build transition compatibility matrix
        transition_matrix = self._build_transition_matrix(request.candidate_moves)
        
        # Step 2: Apply diversity selection
        diverse_candidates = self._apply_diversity_selection(
            request.candidate_moves, 
            request.max_repetitions,
            request.min_diversity_threshold
        )
        
        # Step 3: Align with musical structure
        section_assignments = self._align_with_musical_structure(
            request.music_features.sections,
            diverse_candidates
        )
        
        # Step 4: Optimize sequence using dynamic programming
        optimized_sequence = self._optimize_sequence_dynamic_programming(
            section_assignments,
            transition_matrix,
            request
        )
        
        logger.info(f"Optimization complete: {len(optimized_sequence.moves)} moves, "
                   f"diversity={optimized_sequence.diversity_score:.3f}, "
                   f"flow={optimized_sequence.flow_score:.3f}")
        
        return optimized_sequence
    
    def _build_transition_matrix(self, candidates: List[RecommendationScore]) -> Dict[Tuple[str, str], TransitionScore]:
        """
        Build transition compatibility matrix between all move pairs based on pose analysis.
        
        Args:
            candidates: List of move candidates with analysis results
            
        Returns:
            Dictionary mapping (from_move, to_move) pairs to TransitionScore objects
        """
        logger.info(f"Building transition matrix for {len(candidates)} moves")
        
        transition_matrix = {}
        
        for i, from_candidate in enumerate(candidates):
            for j, to_candidate in enumerate(candidates):
                if i == j:
                    continue  # Skip self-transitions
                
                from_move = from_candidate.move_candidate.move_label
                to_move = to_candidate.move_candidate.move_label
                from_id = from_candidate.move_candidate.move_id
                to_id = to_candidate.move_candidate.move_id
                
                # Use unique candidate IDs for matrix key, but move labels for cache
                matrix_key = (from_id, to_id)
                cache_key = (from_move, to_move)
                
                # Check cache first
                if cache_key in self.transition_cache:
                    transition_matrix[matrix_key] = self.transition_cache[cache_key]
                    continue
                
                # Calculate transition score
                transition_score = self._calculate_transition_compatibility(
                    from_candidate.move_candidate,
                    to_candidate.move_candidate
                )
                
                # Cache and store
                self.transition_cache[cache_key] = transition_score
                transition_matrix[matrix_key] = transition_score
        
        logger.info(f"Built transition matrix with {len(transition_matrix)} transitions")
        return transition_matrix
    
    def _get_transition_score(self, 
                            transition_matrix: Dict[Tuple[str, str], TransitionScore],
                            from_move: MoveCandidate,
                            to_move: MoveCandidate) -> Optional[TransitionScore]:
        """Get transition score from matrix using move IDs."""
        key = (from_move.move_id, to_move.move_id)
        return transition_matrix.get(key)
    
    def _calculate_transition_compatibility(self, 
                                          from_move: MoveCandidate, 
                                          to_move: MoveCandidate) -> TransitionScore:
        """Calculate transition compatibility between two moves."""
        
        # 1. Pose similarity (ending pose of from_move vs starting pose of to_move)
        pose_similarity = self._calculate_pose_transition_similarity(from_move, to_move)
        
        # 2. Movement flow (velocity and direction continuity)
        movement_flow = self._calculate_movement_flow_compatibility(from_move, to_move)
        
        # 3. Energy continuity (smooth energy transitions)
        energy_continuity = self._calculate_energy_continuity(from_move, to_move)
        
        # 4. Difficulty progression (logical skill progression)
        difficulty_progression = self._calculate_difficulty_progression(from_move, to_move)
        
        # Calculate overall compatibility score
        compatibility_score = (
            self.transition_weights['pose_similarity'] * pose_similarity +
            self.transition_weights['movement_flow'] * movement_flow +
            self.transition_weights['energy_continuity'] * energy_continuity +
            self.transition_weights['difficulty_progression'] * difficulty_progression
        )
        
        # Determine transition characteristics
        is_smooth = compatibility_score > 0.7
        requires_pause = compatibility_score < 0.4
        transition_time = 0.5 if is_smooth else (2.0 if requires_pause else 1.0)
        
        return TransitionScore(
            from_move=from_move.move_label,
            to_move=to_move.move_label,
            compatibility_score=compatibility_score,
            pose_similarity=pose_similarity,
            movement_flow=movement_flow,
            energy_continuity=energy_continuity,
            difficulty_progression=difficulty_progression,
            is_smooth=is_smooth,
            requires_pause=requires_pause,
            transition_time=transition_time
        )
    
    def _calculate_pose_transition_similarity(self, from_move: MoveCandidate, to_move: MoveCandidate) -> float:
        """Calculate pose similarity for smooth transitions."""
        # Get ending pose of from_move and starting pose of to_move
        from_analysis = from_move.analysis_result
        to_analysis = to_move.analysis_result
        
        if not from_analysis.pose_features or not to_analysis.pose_features:
            return 0.5  # Default moderate similarity
        
        # Use last pose of from_move and first pose of to_move
        from_ending_pose = from_analysis.pose_features[-1]
        to_starting_pose = to_analysis.pose_features[0]
        
        # Calculate similarity based on key joint positions
        key_joints = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
        similarities = []
        
        for joint in key_joints:
            from_angle = from_ending_pose.joint_angles.get(joint, 180.0)
            to_angle = to_starting_pose.joint_angles.get(joint, 180.0)
            
            # Calculate angle similarity (closer angles = higher similarity)
            angle_diff = abs(from_angle - to_angle)
            similarity = max(0.0, 1.0 - (angle_diff / 180.0))
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _calculate_movement_flow_compatibility(self, from_move: MoveCandidate, to_move: MoveCandidate) -> float:
        """Calculate movement flow compatibility between moves."""
        from_dynamics = from_move.analysis_result.movement_dynamics
        to_dynamics = to_move.analysis_result.movement_dynamics
        
        # 1. Velocity compatibility (similar ending and starting velocities)
        from_ending_velocity = from_dynamics.velocity_profile[-1] if len(from_dynamics.velocity_profile) > 0 else 0.0
        to_starting_velocity = to_dynamics.velocity_profile[0] if len(to_dynamics.velocity_profile) > 0 else 0.0
        
        velocity_diff = abs(from_ending_velocity - to_starting_velocity)
        velocity_compatibility = max(0.0, 1.0 - velocity_diff * 10)  # Scale factor
        
        # 2. Movement direction compatibility
        from_direction = from_dynamics.dominant_movement_direction
        to_direction = to_dynamics.dominant_movement_direction
        
        direction_compatibility = 1.0 if from_direction == to_direction else 0.7
        
        # 3. Spatial coverage compatibility (avoid jarring changes)
        spatial_diff = abs(from_dynamics.spatial_coverage - to_dynamics.spatial_coverage)
        spatial_compatibility = max(0.0, 1.0 - spatial_diff * 5)  # Scale factor
        
        return (velocity_compatibility * 0.4 + 
                direction_compatibility * 0.3 + 
                spatial_compatibility * 0.3)
    
    def _calculate_energy_continuity(self, from_move: MoveCandidate, to_move: MoveCandidate) -> float:
        """Calculate energy level continuity between moves."""
        # Energy level mapping
        energy_levels = {'low': 1, 'medium': 2, 'high': 3}
        
        from_energy = energy_levels.get(from_move.energy_level, 2)
        to_energy = energy_levels.get(to_move.energy_level, 2)
        
        energy_diff = abs(from_energy - to_energy)
        
        if energy_diff == 0:
            return 1.0  # Perfect continuity
        elif energy_diff == 1:
            return 0.8  # Good continuity
        else:
            return 0.4  # Poor continuity (dramatic change)
    
    def _calculate_difficulty_progression(self, from_move: MoveCandidate, to_move: MoveCandidate) -> float:
        """Calculate logical difficulty progression between moves."""
        # Difficulty level mapping
        difficulty_levels = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        
        from_difficulty = difficulty_levels.get(from_move.difficulty, 2)
        to_difficulty = difficulty_levels.get(to_move.difficulty, 2)
        
        difficulty_diff = to_difficulty - from_difficulty
        
        if difficulty_diff == 0:
            return 1.0  # Same level - perfect
        elif difficulty_diff == 1:
            return 0.9  # Logical progression up
        elif difficulty_diff == -1:
            return 0.7  # Step down - acceptable
        elif difficulty_diff == 2:
            return 0.5  # Big jump up - challenging
        else:
            return 0.3  # Big jump down or extreme change
    
    def _apply_diversity_selection(self, 
                                 candidates: List[RecommendationScore],
                                 max_repetitions: int,
                                 min_diversity_threshold: float) -> List[RecommendationScore]:
        """
        Apply diversity selection algorithm to avoid repetitive move sequences.
        
        Args:
            candidates: List of candidate moves with scores
            max_repetitions: Maximum times a move can be repeated
            min_diversity_threshold: Minimum diversity score required
            
        Returns:
            List of diverse move candidates
        """
        logger.info(f"Applying diversity selection: max_repetitions={max_repetitions}, min_threshold={min_diversity_threshold}")
        
        # Group candidates by move type
        move_groups = {}
        for candidate in candidates:
            move_type = self._get_move_type(candidate.move_candidate.move_label)
            if move_type not in move_groups:
                move_groups[move_type] = []
            move_groups[move_type].append(candidate)
        
        # Sort each group by score (descending)
        for move_type in move_groups:
            move_groups[move_type].sort(key=lambda x: x.overall_score, reverse=True)
        
        # Select diverse moves
        selected_moves = []
        move_type_counts = {}
        
        # First pass: select top moves from each type
        for move_type, group in move_groups.items():
            if group:
                selected_moves.append(group[0])
                move_type_counts[move_type] = 1
        
        # Second pass: add more moves respecting repetition limits
        remaining_candidates = []
        for move_type, group in move_groups.items():
            remaining_candidates.extend(group[1:])  # Skip first (already selected)
        
        # Sort remaining by score
        remaining_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        for candidate in remaining_candidates:
            move_type = self._get_move_type(candidate.move_candidate.move_label)
            current_count = move_type_counts.get(move_type, 0)
            
            if current_count < max_repetitions:
                selected_moves.append(candidate)
                move_type_counts[move_type] = current_count + 1
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(selected_moves)
        
        logger.info(f"Diversity selection complete: {len(selected_moves)} moves selected, diversity={diversity_score:.3f}")
        
        # If diversity is too low, add more variety
        if diversity_score < min_diversity_threshold:
            selected_moves = self._enhance_diversity(selected_moves, candidates, min_diversity_threshold)
        
        return selected_moves
    
    def _get_move_type(self, move_label: str) -> str:
        """Extract move type from move label for diversity grouping."""
        # Simple heuristic based on common Bachata move patterns
        move_label_lower = move_label.lower()
        
        if 'basic' in move_label_lower:
            return 'basic'
        elif 'turn' in move_label_lower:
            return 'turn'
        elif 'cross' in move_label_lower:
            return 'cross_body'
        elif 'dip' in move_label_lower:
            return 'dip'
        elif 'roll' in move_label_lower:
            return 'body_roll'
        elif 'styling' in move_label_lower or 'arm' in move_label_lower:
            return 'styling'
        elif 'shadow' in move_label_lower:
            return 'shadow'
        elif 'hammerlock' in move_label_lower:
            return 'hammerlock'
        else:
            return 'other'
    
    def _calculate_diversity_score(self, moves: List[RecommendationScore]) -> float:
        """Calculate diversity score for a set of moves."""
        if len(moves) <= 1:
            return 0.0
        
        # Count move types
        move_types = [self._get_move_type(move.move_candidate.move_label) for move in moves]
        unique_types = set(move_types)
        
        # Diversity based on type variety
        type_diversity = len(unique_types) / len(moves)
        
        # Diversity based on score distribution (avoid all high or all low scores)
        scores = [move.overall_score for move in moves]
        score_std = np.std(scores)
        score_diversity = min(1.0, score_std * 2)  # Normalize
        
        # Combined diversity score
        return (type_diversity * 0.7 + score_diversity * 0.3)
    
    def _enhance_diversity(self, 
                         selected_moves: List[RecommendationScore],
                         all_candidates: List[RecommendationScore],
                         target_diversity: float) -> List[RecommendationScore]:
        """Enhance diversity by adding more varied moves."""
        current_types = set(self._get_move_type(move.move_candidate.move_label) for move in selected_moves)
        
        # Find candidates with new move types
        for candidate in all_candidates:
            if candidate in selected_moves:
                continue
            
            move_type = self._get_move_type(candidate.move_candidate.move_label)
            if move_type not in current_types:
                selected_moves.append(candidate)
                current_types.add(move_type)
                
                # Check if we've reached target diversity
                diversity = self._calculate_diversity_score(selected_moves)
                if diversity >= target_diversity:
                    break
        
        return selected_moves
    
    def _align_with_musical_structure(self, 
                                    sections: List[MusicSection],
                                    candidates: List[RecommendationScore]) -> List[Tuple[MusicSection, List[RecommendationScore]]]:
        """
        Align move complexity with song sections (intro/verse/chorus/bridge).
        
        Args:
            sections: Musical sections from music analysis
            candidates: Available move candidates
            
        Returns:
            List of (section, assigned_moves) tuples
        """
        logger.info(f"Aligning {len(candidates)} moves with {len(sections)} musical sections")
        
        section_assignments = []
        
        for section in sections:
            # Filter moves appropriate for this section type
            section_moves = self._filter_moves_for_section(section, candidates)
            section_assignments.append((section, section_moves))
        
        return section_assignments
    
    def _filter_moves_for_section(self, 
                                section: MusicSection, 
                                candidates: List[RecommendationScore]) -> List[RecommendationScore]:
        """Filter moves appropriate for a specific musical section."""
        appropriate_moves = []
        
        for candidate in candidates:
            move_label = candidate.move_candidate.move_label.lower()
            
            # Check if move type is recommended for this section
            if any(rec_type in move_label for rec_type in section.recommended_move_types):
                appropriate_moves.append(candidate)
            # Also include moves with compatible energy levels
            elif self._is_energy_compatible(section.energy_level, candidate.move_candidate.energy_level):
                appropriate_moves.append(candidate)
        
        # If no specific matches, include all moves (fallback)
        if not appropriate_moves:
            appropriate_moves = candidates.copy()
        
        # Sort by score within section
        appropriate_moves.sort(key=lambda x: x.overall_score, reverse=True)
        
        return appropriate_moves
    
    def _is_energy_compatible(self, section_energy: float, move_energy: str) -> bool:
        """Check if move energy level is compatible with section energy."""
        # Map move energy to numeric
        energy_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        move_energy_value = energy_map.get(move_energy, 0.6)
        
        # Compatible if within reasonable range
        return abs(section_energy - move_energy_value) < 0.4
    
    def _optimize_sequence_dynamic_programming(self,
                                             section_assignments: List[Tuple[MusicSection, List[RecommendationScore]]],
                                             transition_matrix: Dict[Tuple[str, str], TransitionScore],
                                             request: OptimizationRequest) -> ChoreographySequence:
        """
        Optimize sequence using dynamic programming for smooth choreography flow.
        
        Args:
            section_assignments: Moves assigned to each musical section
            transition_matrix: Transition compatibility scores
            request: Optimization request parameters
            
        Returns:
            Optimized ChoreographySequence
        """
        logger.info("Optimizing sequence using dynamic programming")
        
        # Build sequence for each section
        optimized_moves = []
        transition_scores = []
        current_time = 0.0
        
        for i, (section, section_moves) in enumerate(section_assignments):
            section_duration = section.end_time - section.start_time
            
            # Select moves for this section based on duration and flow
            section_sequence = self._optimize_section_sequence(
                section_moves,
                section_duration,
                transition_matrix,
                optimized_moves[-1] if optimized_moves else None  # Previous move for transition
            )
            
            # Add to overall sequence
            for move_candidate in section_sequence:
                optimized_moves.append(move_candidate)
                
                # Calculate transition score if not first move
                if len(optimized_moves) > 1:
                    prev_move = optimized_moves[-2]
                    transition_score = self._get_transition_score(transition_matrix, prev_move, move_candidate)
                    if transition_score:
                        transition_scores.append(transition_score)
                
                current_time += move_candidate.analysis_result.duration
                
                # Stop if we've reached target duration
                if current_time >= request.target_duration:
                    break
            
            if current_time >= request.target_duration:
                break
        
        # Calculate overall scores
        total_score = np.mean([move.analysis_result.movement_complexity_score for move in optimized_moves])
        diversity_score = self._calculate_diversity_score([
            type('MockScore', (), {'move_candidate': move, 'overall_score': move.analysis_result.movement_complexity_score})()
            for move in optimized_moves
        ])
        flow_score = np.mean([ts.compatibility_score for ts in transition_scores]) if transition_scores else 0.8
        musical_alignment_score = 0.8  # Placeholder - could be calculated based on section alignment
        
        return ChoreographySequence(
            moves=optimized_moves,
            total_duration=current_time,
            total_score=total_score,
            diversity_score=diversity_score,
            flow_score=flow_score,
            musical_alignment_score=musical_alignment_score,
            transition_scores=transition_scores,
            section_alignment=section_assignments,
            optimization_method="dynamic_programming",
            iterations=1
        )
    
    def _optimize_section_sequence(self,
                                 section_moves: List[RecommendationScore],
                                 section_duration: float,
                                 transition_matrix: Dict[Tuple[str, str], TransitionScore],
                                 previous_move: Optional[MoveCandidate]) -> List[MoveCandidate]:
        """Optimize move sequence for a single musical section."""
        if not section_moves:
            return []
        
        selected_moves = []
        remaining_duration = section_duration
        
        # Start with highest scoring move that fits
        for move_score in section_moves:
            move_duration = move_score.move_candidate.analysis_result.duration
            
            # Check if move fits in remaining time
            if move_duration <= remaining_duration:
                # Check transition compatibility if there's a previous move
                if previous_move:
                    transition_score = self._get_transition_score(transition_matrix, previous_move, move_score.move_candidate)
                    if transition_score and transition_score.compatibility_score > 0.3:
                        selected_moves.append(move_score.move_candidate)
                        remaining_duration -= move_duration
                        break
                else:
                    # First move - no transition check needed
                    selected_moves.append(move_score.move_candidate)
                    remaining_duration -= move_duration
                    break
        
        # Fill remaining time with compatible moves
        while remaining_duration > 5.0 and section_moves:  # At least 5 seconds left
            best_move = None
            best_score = -1.0
            
            for move_score in section_moves:
                if move_score.move_candidate in selected_moves:
                    continue  # Skip already selected moves
                
                move_duration = move_score.move_candidate.analysis_result.duration
                if move_duration > remaining_duration:
                    continue  # Skip moves that don't fit
                
                # Calculate combined score (move quality + transition quality)
                combined_score = move_score.overall_score
                
                if selected_moves:
                    last_move = selected_moves[-1]
                    transition_score = self._get_transition_score(transition_matrix, last_move, move_score.move_candidate)
                    if transition_score:
                        combined_score = (combined_score + transition_score.compatibility_score) / 2
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_move = move_score.move_candidate
            
            if best_move:
                selected_moves.append(best_move)
                remaining_duration -= best_move.analysis_result.duration
            else:
                break  # No more suitable moves
        
        return selected_moves
    
    def get_transition_matrix_summary(self, transition_matrix: Dict[Tuple[str, str], TransitionScore]) -> Dict[str, Any]:
        """Get summary statistics of the transition matrix."""
        if not transition_matrix:
            return {}
        
        scores = [ts.compatibility_score for ts in transition_matrix.values()]
        smooth_transitions = sum(1 for ts in transition_matrix.values() if ts.is_smooth)
        pause_transitions = sum(1 for ts in transition_matrix.values() if ts.requires_pause)
        
        return {
            'total_transitions': len(transition_matrix),
            'mean_compatibility': np.mean(scores),
            'std_compatibility': np.std(scores),
            'min_compatibility': np.min(scores),
            'max_compatibility': np.max(scores),
            'smooth_transitions': smooth_transitions,
            'pause_transitions': pause_transitions,
            'smooth_percentage': smooth_transitions / len(transition_matrix) * 100,
            'pause_percentage': pause_transitions / len(transition_matrix) * 100
        }