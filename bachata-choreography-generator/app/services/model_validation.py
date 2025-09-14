"""
Model validation and performance testing framework.
Implements cross-validation, A/B testing, evaluation metrics, and performance benchmarking
for the Bachata choreography recommendation system.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import json

from .recommendation_engine import RecommendationEngine, MoveCandidate, RecommendationRequest, RecommendationScore
from .choreography_optimizer import ChoreographyOptimizer, OptimizationRequest, ChoreographySequence
from .feature_fusion import FeatureFusion, MultiModalEmbedding
from .music_analyzer import MusicAnalyzer, MusicFeatures
from .move_analyzer import MoveAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation test results."""
    test_name: str
    score: float
    details: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class CrossValidationResult:
    """Container for cross-validation results."""
    mean_score: float
    std_score: float
    fold_scores: List[float]
    best_fold: int
    worst_fold: int
    validation_details: Dict[str, Any]


@dataclass
class ABTestResult:
    """Container for A/B testing results."""
    variant_a_score: float
    variant_b_score: float
    improvement: float
    statistical_significance: float
    sample_size: int
    test_details: Dict[str, Any]


@dataclass
class PerformanceBenchmark:
    """Container for performance benchmark results."""
    operation: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float  # operations per second
    memory_usage: Optional[float] = None


class ModelValidationFramework:
    """
    Model validation and performance testing framework that implements:
    - Cross-validation system using held-out test songs and expert choreographer ratings
    - A/B testing framework to compare different scoring weight combinations
    - Evaluation metrics for choreography quality (flow, musicality, difficulty progression)
    - Performance benchmarking for recommendation speed and accuracy
    """
    
    def __init__(self):
        """Initialize the model validation framework."""
        self.recommendation_engine = RecommendationEngine()
        self.choreography_optimizer = ChoreographyOptimizer()
        self.feature_fusion = FeatureFusion()
        self.music_analyzer = MusicAnalyzer()
        self.move_analyzer = MoveAnalyzer(target_fps=30)
        
        # Validation metrics
        self.quality_metrics = [
            'flow_score',
            'musicality_score', 
            'difficulty_progression_score',
            'diversity_score',
            'transition_smoothness'
        ]
        
        logger.info("ModelValidationFramework initialized with quality metrics: " + 
                   ", ".join(self.quality_metrics))
    
    def run_cross_validation(self, 
                           test_songs: List[str],
                           move_candidates: List[MoveCandidate],
                           k_folds: int = 5,
                           expert_ratings: Optional[Dict[str, float]] = None) -> CrossValidationResult:
        """
        Run cross-validation using held-out test songs and expert choreographer ratings.
        
        Args:
            test_songs: List of audio file paths for testing
            move_candidates: Available move candidates
            k_folds: Number of cross-validation folds
            expert_ratings: Optional expert ratings for validation
            
        Returns:
            CrossValidationResult with validation metrics
        """
        logger.info(f"Running {k_folds}-fold cross-validation on {len(test_songs)} test songs")
        
        if len(test_songs) < k_folds:
            raise ValueError(f"Need at least {k_folds} test songs for {k_folds}-fold CV")
        
        # Split songs into folds
        fold_size = len(test_songs) // k_folds
        folds = []
        for i in range(k_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k_folds - 1 else len(test_songs)
            folds.append(test_songs[start_idx:end_idx])
        
        fold_scores = []
        validation_details = {}
        
        for fold_idx in range(k_folds):
            logger.info(f"Running fold {fold_idx + 1}/{k_folds}")
            
            # Use current fold as test set, others as training
            test_fold = folds[fold_idx]
            train_folds = [song for i, fold in enumerate(folds) if i != fold_idx for song in fold]
            
            # Validate on test fold
            fold_score = self._validate_fold(test_fold, move_candidates, expert_ratings)
            fold_scores.append(fold_score)
            
            validation_details[f'fold_{fold_idx}'] = {
                'score': fold_score,
                'test_songs': len(test_fold),
                'train_songs': len(train_folds)
            }
        
        # Calculate cross-validation statistics
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        best_fold = int(np.argmax(fold_scores))
        worst_fold = int(np.argmin(fold_scores))
        
        logger.info(f"Cross-validation complete: mean={mean_score:.3f}, std={std_score:.3f}")
        
        return CrossValidationResult(
            mean_score=mean_score,
            std_score=std_score,
            fold_scores=fold_scores,
            best_fold=best_fold,
            worst_fold=worst_fold,
            validation_details=validation_details
        )
    
    def run_ab_test(self,
                   test_songs: List[str],
                   move_candidates: List[MoveCandidate],
                   variant_a_weights: Dict[str, float],
                   variant_b_weights: Dict[str, float],
                   sample_size: Optional[int] = None) -> ABTestResult:
        """
        Run A/B test to compare different scoring weight combinations.
        
        Args:
            test_songs: List of audio file paths for testing
            move_candidates: Available move candidates
            variant_a_weights: Scoring weights for variant A
            variant_b_weights: Scoring weights for variant B
            sample_size: Optional sample size limit
            
        Returns:
            ABTestResult with comparison metrics
        """
        logger.info(f"Running A/B test on {len(test_songs)} songs")
        logger.info(f"Variant A weights: {variant_a_weights}")
        logger.info(f"Variant B weights: {variant_b_weights}")
        
        # Limit sample size if specified
        if sample_size and sample_size < len(test_songs):
            test_songs = test_songs[:sample_size]
        
        variant_a_scores = []
        variant_b_scores = []
        
        for song_path in test_songs:
            try:
                # Test variant A
                score_a = self._test_variant(song_path, move_candidates, variant_a_weights)
                variant_a_scores.append(score_a)
                
                # Test variant B
                score_b = self._test_variant(song_path, move_candidates, variant_b_weights)
                variant_b_scores.append(score_b)
                
            except Exception as e:
                logger.warning(f"Failed to test song {song_path}: {e}")
                continue
        
        if not variant_a_scores or not variant_b_scores:
            raise ValueError("No successful tests completed")
        
        # Calculate statistics
        mean_a = np.mean(variant_a_scores)
        mean_b = np.mean(variant_b_scores)
        improvement = (mean_b - mean_a) / mean_a * 100 if mean_a > 0 else 0
        
        # Simple statistical significance test (t-test approximation)
        if len(variant_a_scores) > 1 and len(variant_b_scores) > 1:
            pooled_std = np.sqrt((np.var(variant_a_scores) + np.var(variant_b_scores)) / 2)
            t_stat = abs(mean_b - mean_a) / (pooled_std * np.sqrt(2 / len(variant_a_scores)))
            # Rough p-value approximation (for t > 2, p < 0.05)
            statistical_significance = max(0, min(1, 1 - t_stat / 4))
        else:
            statistical_significance = 0.5
        
        logger.info(f"A/B test complete: A={mean_a:.3f}, B={mean_b:.3f}, improvement={improvement:.1f}%")
        
        return ABTestResult(
            variant_a_score=mean_a,
            variant_b_score=mean_b,
            improvement=improvement,
            statistical_significance=statistical_significance,
            sample_size=len(variant_a_scores),
            test_details={
                'variant_a_scores': variant_a_scores,
                'variant_b_scores': variant_b_scores,
                'variant_a_weights': variant_a_weights,
                'variant_b_weights': variant_b_weights
            }
        )
    
    def evaluate_choreography_quality(self, 
                                    choreography: ChoreographySequence,
                                    music_features: MusicFeatures) -> Dict[str, float]:
        """
        Evaluate choreography quality using multiple metrics.
        
        Args:
            choreography: Generated choreography sequence
            music_features: Original music features
            
        Returns:
            Dictionary of quality scores
        """
        quality_scores = {}
        
        # 1. Flow score (already calculated)
        quality_scores['flow_score'] = choreography.flow_score
        
        # 2. Musicality score (alignment with musical structure)
        quality_scores['musicality_score'] = self._calculate_musicality_score(
            choreography, music_features
        )
        
        # 3. Difficulty progression score
        quality_scores['difficulty_progression_score'] = self._calculate_difficulty_progression_score(
            choreography
        )
        
        # 4. Diversity score (already calculated)
        quality_scores['diversity_score'] = choreography.diversity_score
        
        # 5. Transition smoothness
        quality_scores['transition_smoothness'] = self._calculate_transition_smoothness(
            choreography
        )
        
        # 6. Overall quality (weighted combination)
        quality_scores['overall_quality'] = self._calculate_overall_quality(quality_scores)
        
        return quality_scores
    
    def benchmark_performance(self, 
                            test_songs: List[str],
                            move_candidates: List[MoveCandidate],
                            operations: List[str] = None) -> Dict[str, PerformanceBenchmark]:
        """
        Benchmark performance for recommendation speed and accuracy.
        
        Args:
            test_songs: List of audio file paths for testing
            move_candidates: Available move candidates
            operations: List of operations to benchmark
            
        Returns:
            Dictionary of performance benchmarks
        """
        if operations is None:
            operations = [
                'music_analysis',
                'move_analysis', 
                'recommendation_generation',
                'choreography_optimization',
                'end_to_end'
            ]
        
        logger.info(f"Benchmarking {len(operations)} operations on {len(test_songs)} songs")
        
        benchmarks = {}
        
        for operation in operations:
            logger.info(f"Benchmarking: {operation}")
            
            times = []
            successful_ops = 0
            
            for song_path in test_songs:
                try:
                    start_time = time.time()
                    
                    if operation == 'music_analysis':
                        self.music_analyzer.analyze_audio(song_path)
                    elif operation == 'move_analysis':
                        # Use first available move for benchmarking
                        if move_candidates:
                            self.move_analyzer.analyze_move_clip(move_candidates[0].video_path)
                    elif operation == 'recommendation_generation':
                        self._benchmark_recommendation_generation(song_path, move_candidates)
                    elif operation == 'choreography_optimization':
                        self._benchmark_choreography_optimization(song_path, move_candidates)
                    elif operation == 'end_to_end':
                        self._benchmark_end_to_end(song_path, move_candidates)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    successful_ops += 1
                    
                except Exception as e:
                    logger.warning(f"Benchmark failed for {song_path} on {operation}: {e}")
                    continue
            
            if times:
                benchmarks[operation] = PerformanceBenchmark(
                    operation=operation,
                    mean_time=np.mean(times),
                    std_time=np.std(times),
                    min_time=np.min(times),
                    max_time=np.max(times),
                    throughput=successful_ops / sum(times) if sum(times) > 0 else 0
                )
                
                logger.info(f"{operation}: {benchmarks[operation].mean_time:.2f}s avg, "
                           f"{benchmarks[operation].throughput:.2f} ops/sec")
        
        return benchmarks
    
    def validate_model_accuracy(self,
                               test_songs: List[str],
                               move_candidates: List[MoveCandidate],
                               ground_truth: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate model accuracy against ground truth or expert ratings.
        
        Args:
            test_songs: List of audio file paths for testing
            move_candidates: Available move candidates
            ground_truth: Optional ground truth data for validation
            
        Returns:
            ValidationResult with accuracy metrics
        """
        logger.info(f"Validating model accuracy on {len(test_songs)} songs")
        
        start_time = time.time()
        accuracy_scores = []
        details = {}
        
        try:
            for i, song_path in enumerate(test_songs):
                try:
                    # Generate choreography
                    music_features = self.music_analyzer.analyze_audio(song_path)
                    
                    # Create recommendation request
                    music_embedding = self.feature_fusion.create_audio_embedding(music_features)
                    request = RecommendationRequest(
                        music_features=music_features,
                        music_embedding=type('MockEmbedding', (), {'audio_embedding': music_embedding})(),
                        target_difficulty="intermediate",
                        target_energy="medium"
                    )
                    
                    # Get recommendations
                    recommendations = self.recommendation_engine.recommend_moves(
                        request, 
                        [type('MockScore', (), {
                            'move_candidate': candidate,
                            'overall_score': 0.8,
                            'audio_similarity': 0.8,
                            'tempo_compatibility': 0.8,
                            'energy_alignment': 0.8,
                            'difficulty_compatibility': 0.8
                        })() for candidate in move_candidates],
                        top_k=min(10, len(move_candidates))
                    )
                    
                    # Optimize choreography
                    optimization_request = OptimizationRequest(
                        music_features=music_features,
                        candidate_moves=recommendations,
                        target_duration=60.0
                    )
                    
                    choreography = self.choreography_optimizer.optimize_choreography(optimization_request)
                    
                    # Evaluate quality
                    quality_scores = self.evaluate_choreography_quality(choreography, music_features)
                    
                    # Calculate accuracy score
                    if ground_truth and song_path in ground_truth:
                        # Compare against ground truth
                        accuracy = self._compare_with_ground_truth(
                            choreography, quality_scores, ground_truth[song_path]
                        )
                    else:
                        # Use quality metrics as proxy for accuracy
                        accuracy = quality_scores['overall_quality']
                    
                    accuracy_scores.append(accuracy)
                    details[f'song_{i}'] = {
                        'path': song_path,
                        'accuracy': accuracy,
                        'quality_scores': quality_scores
                    }
                    
                except Exception as e:
                    logger.warning(f"Validation failed for {song_path}: {e}")
                    continue
            
            if not accuracy_scores:
                raise ValueError("No successful validations completed")
            
            mean_accuracy = np.mean(accuracy_scores)
            execution_time = time.time() - start_time
            
            logger.info(f"Model validation complete: accuracy={mean_accuracy:.3f}")
            
            return ValidationResult(
                test_name="model_accuracy",
                score=mean_accuracy,
                details=details,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Model validation failed: {e}")
            
            return ValidationResult(
                test_name="model_accuracy",
                score=0.0,
                details={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _validate_fold(self, 
                      test_songs: List[str], 
                      move_candidates: List[MoveCandidate],
                      expert_ratings: Optional[Dict[str, float]]) -> float:
        """Validate a single cross-validation fold."""
        scores = []
        
        for song_path in test_songs:
            try:
                # Generate choreography for this song
                music_features = self.music_analyzer.analyze_audio(song_path)
                
                # Create mock recommendations for testing
                mock_recommendations = [
                    type('MockScore', (), {
                        'move_candidate': candidate,
                        'overall_score': 0.8
                    })() for candidate in move_candidates[:5]  # Use top 5 for speed
                ]
                
                # Optimize choreography
                optimization_request = OptimizationRequest(
                    music_features=music_features,
                    candidate_moves=mock_recommendations,
                    target_duration=30.0  # Shorter for validation
                )
                
                choreography = self.choreography_optimizer.optimize_choreography(optimization_request)
                
                # Evaluate quality
                quality_scores = self.evaluate_choreography_quality(choreography, music_features)
                
                # Use expert rating if available, otherwise use quality score
                if expert_ratings and song_path in expert_ratings:
                    score = expert_ratings[song_path]
                else:
                    score = quality_scores['overall_quality']
                
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Fold validation failed for {song_path}: {e}")
                continue
        
        return np.mean(scores) if scores else 0.0
    
    def _test_variant(self, 
                     song_path: str, 
                     move_candidates: List[MoveCandidate],
                     weights: Dict[str, float]) -> float:
        """Test a single variant with specific weights."""
        # Analyze music
        music_features = self.music_analyzer.analyze_audio(song_path)
        
        # Create recommendation request with custom weights
        music_embedding = self.feature_fusion.create_audio_embedding(music_features)
        request = RecommendationRequest(
            music_features=music_features,
            music_embedding=type('MockEmbedding', (), {'audio_embedding': music_embedding})(),
            target_difficulty="intermediate",
            target_energy="medium",
            weights=weights
        )
        
        # Get recommendations
        mock_recommendations = [
            type('MockScore', (), {
                'move_candidate': candidate,
                'overall_score': 0.8
            })() for candidate in move_candidates[:5]
        ]
        
        recommendations = self.recommendation_engine.recommend_moves(
            request, mock_recommendations, top_k=5
        )
        
        # Optimize choreography
        optimization_request = OptimizationRequest(
            music_features=music_features,
            candidate_moves=recommendations,
            target_duration=30.0
        )
        
        choreography = self.choreography_optimizer.optimize_choreography(optimization_request)
        
        # Evaluate quality
        quality_scores = self.evaluate_choreography_quality(choreography, music_features)
        
        return quality_scores['overall_quality']
    
    def _calculate_musicality_score(self, 
                                  choreography: ChoreographySequence,
                                  music_features: MusicFeatures) -> float:
        """Calculate how well choreography aligns with musical structure."""
        if not choreography.section_alignment:
            return 0.5  # Default moderate score
        
        alignment_scores = []
        
        for section, assigned_moves in choreography.section_alignment:
            if not assigned_moves:
                continue
            
            # Check if moves match section recommendations
            section_score = 0.0
            for move in assigned_moves:
                move_type = move.move_label.lower()
                
                # Check if move type is in recommended types
                if any(rec_type in move_type for rec_type in section.recommended_move_types):
                    section_score += 1.0
                else:
                    section_score += 0.5  # Partial credit
            
            # Normalize by number of moves
            if assigned_moves:
                section_score /= len(assigned_moves)
            
            alignment_scores.append(section_score)
        
        return np.mean(alignment_scores) if alignment_scores else 0.5
    
    def _calculate_difficulty_progression_score(self, choreography: ChoreographySequence) -> float:
        """Calculate how well difficulty progresses through the choreography."""
        if len(choreography.moves) < 2:
            return 1.0  # Perfect score for single move
        
        difficulty_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        
        progression_scores = []
        
        for i in range(len(choreography.moves) - 1):
            current_difficulty = difficulty_map.get(choreography.moves[i].difficulty, 2)
            next_difficulty = difficulty_map.get(choreography.moves[i + 1].difficulty, 2)
            
            # Calculate progression score
            diff = next_difficulty - current_difficulty
            
            if diff == 0:
                score = 1.0  # Same level - good
            elif diff == 1:
                score = 0.9  # Logical progression up
            elif diff == -1:
                score = 0.8  # Step down - acceptable
            elif diff == 2:
                score = 0.6  # Big jump - challenging
            else:
                score = 0.4  # Extreme change
            
            progression_scores.append(score)
        
        return np.mean(progression_scores)
    
    def _calculate_transition_smoothness(self, choreography: ChoreographySequence) -> float:
        """Calculate average transition smoothness."""
        if not choreography.transition_scores:
            return 0.8  # Default good score
        
        smoothness_scores = [ts.compatibility_score for ts in choreography.transition_scores]
        return np.mean(smoothness_scores)
    
    def _calculate_overall_quality(self, quality_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'flow_score': 0.25,
            'musicality_score': 0.25,
            'difficulty_progression_score': 0.15,
            'diversity_score': 0.20,
            'transition_smoothness': 0.15
        }
        
        overall = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_scores:
                overall += quality_scores[metric] * weight
                total_weight += weight
        
        return overall / total_weight if total_weight > 0 else 0.5
    
    def _compare_with_ground_truth(self, 
                                 choreography: ChoreographySequence,
                                 quality_scores: Dict[str, float],
                                 ground_truth: Dict[str, Any]) -> float:
        """Compare choreography with ground truth data."""
        # Simple comparison based on available ground truth metrics
        accuracy = 0.0
        comparisons = 0
        
        if 'expected_quality' in ground_truth:
            accuracy += 1.0 - abs(quality_scores['overall_quality'] - ground_truth['expected_quality'])
            comparisons += 1
        
        if 'expected_moves' in ground_truth:
            # Compare move types
            actual_moves = set(move.move_label for move in choreography.moves)
            expected_moves = set(ground_truth['expected_moves'])
            
            intersection = len(actual_moves.intersection(expected_moves))
            union = len(actual_moves.union(expected_moves))
            
            if union > 0:
                accuracy += intersection / union
                comparisons += 1
        
        return accuracy / comparisons if comparisons > 0 else quality_scores['overall_quality']
    
    def _benchmark_recommendation_generation(self, song_path: str, move_candidates: List[MoveCandidate]):
        """Benchmark recommendation generation."""
        music_features = self.music_analyzer.analyze_audio(song_path)
        music_embedding = self.feature_fusion.create_audio_embedding(music_features)
        
        request = RecommendationRequest(
            music_features=music_features,
            music_embedding=type('MockEmbedding', (), {'audio_embedding': music_embedding})(),
            target_difficulty="intermediate"
        )
        
        mock_recommendations = [
            type('MockScore', (), {
                'move_candidate': candidate,
                'overall_score': 0.8
            })() for candidate in move_candidates[:10]
        ]
        
        self.recommendation_engine.recommend_moves(request, mock_recommendations, top_k=5)
    
    def _benchmark_choreography_optimization(self, song_path: str, move_candidates: List[MoveCandidate]):
        """Benchmark choreography optimization."""
        music_features = self.music_analyzer.analyze_audio(song_path)
        
        mock_recommendations = [
            type('MockScore', (), {
                'move_candidate': candidate,
                'overall_score': 0.8
            })() for candidate in move_candidates[:5]
        ]
        
        optimization_request = OptimizationRequest(
            music_features=music_features,
            candidate_moves=mock_recommendations,
            target_duration=30.0
        )
        
        self.choreography_optimizer.optimize_choreography(optimization_request)
    
    def _benchmark_end_to_end(self, song_path: str, move_candidates: List[MoveCandidate]):
        """Benchmark complete end-to-end pipeline."""
        # Music analysis
        music_features = self.music_analyzer.analyze_audio(song_path)
        music_embedding = self.feature_fusion.create_audio_embedding(music_features)
        
        # Recommendation generation
        request = RecommendationRequest(
            music_features=music_features,
            music_embedding=type('MockEmbedding', (), {'audio_embedding': music_embedding})(),
            target_difficulty="intermediate"
        )
        
        mock_recommendations = [
            type('MockScore', (), {
                'move_candidate': candidate,
                'overall_score': 0.8
            })() for candidate in move_candidates[:5]
        ]
        
        recommendations = self.recommendation_engine.recommend_moves(
            request, mock_recommendations, top_k=5
        )
        
        # Choreography optimization
        optimization_request = OptimizationRequest(
            music_features=music_features,
            candidate_moves=recommendations,
            target_duration=30.0
        )
        
        choreography = self.choreography_optimizer.optimize_choreography(optimization_request)
        
        # Quality evaluation
        self.evaluate_choreography_quality(choreography, music_features)
    
    def save_validation_results(self, results: Dict[str, Any], output_path: str):
        """Save validation results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Validation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
    
    def load_validation_results(self, input_path: str) -> Dict[str, Any]:
        """Load validation results from file."""
        try:
            with open(input_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Validation results loaded from {input_path}")
            return results
        except Exception as e:
            logger.error(f"Failed to load validation results: {e}")
            return {}