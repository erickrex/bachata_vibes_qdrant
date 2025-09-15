"""
Hyperparameter optimization for recommendation weights.
Implements grid search and optimization for scoring weights without heavy video processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from itertools import product
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    audio_weight_range: Tuple[float, float] = (0.1, 0.6)
    tempo_weight_range: Tuple[float, float] = (0.1, 0.4)
    energy_weight_range: Tuple[float, float] = (0.1, 0.4)
    difficulty_weight_range: Tuple[float, float] = (0.1, 0.4)
    
    # Grid search parameters
    grid_steps: int = 5
    max_combinations: int = 100
    
    # Validation parameters
    validation_split: float = 0.2
    cross_validation_folds: int = 3


@dataclass
class WeightConfiguration:
    """A specific weight configuration for testing."""
    audio_weight: float
    tempo_weight: float
    energy_weight: float
    difficulty_weight: float
    
    @property
    def total_weight(self) -> float:
        """Calculate total weight (should be close to 1.0)."""
        return self.audio_weight + self.tempo_weight + self.energy_weight + self.difficulty_weight
    
    def normalize(self) -> 'WeightConfiguration':
        """Normalize weights to sum to 1.0."""
        total = self.total_weight
        if total == 0:
            return WeightConfiguration(0.25, 0.25, 0.25, 0.25)
        
        return WeightConfiguration(
            audio_weight=self.audio_weight / total,
            tempo_weight=self.tempo_weight / total,
            energy_weight=self.energy_weight / total,
            difficulty_weight=self.difficulty_weight / total
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'audio': self.audio_weight,
            'tempo': self.tempo_weight,
            'energy': self.energy_weight,
            'difficulty': self.difficulty_weight
        }


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_weights: WeightConfiguration
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_stats: Dict[str, Any]


@dataclass
class ValidationExample:
    """Simplified validation example without heavy video processing."""
    clip_id: str
    move_label: str
    difficulty: str
    energy_level: str
    estimated_tempo: int
    
    # Simplified features (mock data for demonstration)
    audio_features: np.ndarray
    movement_complexity: float
    rhythm_score: float


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer for recommendation system weights.
    Uses simplified validation without heavy MediaPipe processing.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize the hyperparameter optimizer."""
        self.config = config or OptimizationConfig()
        self.validation_examples = []
        
        logger.info("HyperparameterOptimizer initialized")
    
    def create_mock_validation_data(self, annotations: List[Any]) -> List[ValidationExample]:
        """
        Create mock validation data from annotations for fast optimization.
        This replaces the heavy video processing with simulated features.
        """
        validation_examples = []
        
        for ann in annotations:
            # Create mock features based on annotation metadata
            audio_features = self._generate_mock_audio_features(ann)
            movement_complexity = self._estimate_movement_complexity(ann)
            rhythm_score = self._estimate_rhythm_score(ann)
            
            example = ValidationExample(
                clip_id=ann.clip_id,
                move_label=ann.move_label,
                difficulty=ann.difficulty,
                energy_level=ann.energy_level,
                estimated_tempo=ann.estimated_tempo,
                audio_features=audio_features,
                movement_complexity=movement_complexity,
                rhythm_score=rhythm_score
            )
            validation_examples.append(example)
        
        logger.info(f"Created {len(validation_examples)} mock validation examples")
        return validation_examples
    
    def _generate_mock_audio_features(self, annotation: Any) -> np.ndarray:
        """Generate mock audio features based on annotation metadata."""
        # Create features based on tempo and energy level
        tempo_normalized = annotation.estimated_tempo / 130.0  # Normalize around typical Bachata tempo
        
        energy_mapping = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        energy_score = energy_mapping.get(annotation.energy_level, 0.6)
        
        # Create a 128-dimensional mock audio feature vector
        base_features = np.random.normal(0, 0.1, 128)
        
        # Inject tempo and energy information
        base_features[0] = tempo_normalized
        base_features[1] = energy_score
        base_features[2] = tempo_normalized * energy_score  # Interaction term
        
        return base_features
    
    def _estimate_movement_complexity(self, annotation: Any) -> float:
        """Estimate movement complexity from annotation metadata."""
        difficulty_mapping = {'beginner': 0.3, 'intermediate': 0.6, 'advanced': 0.9}
        base_complexity = difficulty_mapping.get(annotation.difficulty, 0.6)
        
        # Adjust based on move type
        complex_moves = ['combination', 'body_roll', 'hammerlock', 'shadow_position']
        if any(move in annotation.move_label.lower() for move in complex_moves):
            base_complexity += 0.2
        
        return min(1.0, base_complexity)
    
    def _estimate_rhythm_score(self, annotation: Any) -> float:
        """Estimate rhythm compatibility from annotation metadata."""
        # Higher rhythm scores for moves that typically match musical rhythm
        rhythmic_moves = ['basic_step', 'forward_backward', 'cross_body_lead']
        if any(move in annotation.move_label.lower() for move in rhythmic_moves):
            return 0.8 + np.random.normal(0, 0.1)
        else:
            return 0.6 + np.random.normal(0, 0.15)
    
    def generate_weight_configurations(self) -> List[WeightConfiguration]:
        """Generate weight configurations for grid search."""
        config = self.config
        
        # Create ranges for each weight
        audio_range = np.linspace(config.audio_weight_range[0], config.audio_weight_range[1], config.grid_steps)
        tempo_range = np.linspace(config.tempo_weight_range[0], config.tempo_weight_range[1], config.grid_steps)
        energy_range = np.linspace(config.energy_weight_range[0], config.energy_weight_range[1], config.grid_steps)
        difficulty_range = np.linspace(config.difficulty_weight_range[0], config.difficulty_weight_range[1], config.grid_steps)
        
        # Generate all combinations
        all_combinations = list(product(audio_range, tempo_range, energy_range, difficulty_range))
        
        # Limit combinations if too many
        if len(all_combinations) > config.max_combinations:
            indices = np.random.choice(len(all_combinations), config.max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        # Create weight configurations and normalize
        configurations = []
        for audio_w, tempo_w, energy_w, diff_w in all_combinations:
            config_obj = WeightConfiguration(audio_w, tempo_w, energy_w, diff_w)
            normalized_config = config_obj.normalize()
            configurations.append(normalized_config)
        
        logger.info(f"Generated {len(configurations)} weight configurations")
        return configurations
    
    def evaluate_configuration(self, weights: WeightConfiguration, 
                             validation_examples: List[ValidationExample]) -> float:
        """
        Evaluate a weight configuration using simplified similarity scoring.
        Returns a score between 0 and 1 (higher is better).
        """
        total_score = 0.0
        num_comparisons = 0
        
        # Compare each example with a few others
        for i, example1 in enumerate(validation_examples):
            # Compare with next few examples (to keep it fast)
            for j in range(i + 1, min(i + 5, len(validation_examples))):
                example2 = validation_examples[j]
                
                # Calculate similarity score using current weights
                similarity_score = self._calculate_weighted_similarity(
                    example1, example2, weights
                )
                
                # Calculate ground truth similarity
                ground_truth = self._calculate_ground_truth_similarity(example1, example2)
                
                # Score is how close our weighted similarity is to ground truth
                score = 1.0 - abs(similarity_score - ground_truth)
                total_score += score
                num_comparisons += 1
        
        if num_comparisons == 0:
            return 0.0
        
        return total_score / num_comparisons
    
    def _calculate_weighted_similarity(self, example1: ValidationExample, 
                                     example2: ValidationExample,
                                     weights: WeightConfiguration) -> float:
        """Calculate weighted similarity between two examples."""
        # Audio similarity (cosine similarity of mock features)
        audio_sim = np.dot(example1.audio_features, example2.audio_features) / (
            np.linalg.norm(example1.audio_features) * np.linalg.norm(example2.audio_features)
        )
        audio_sim = max(0, audio_sim)  # Ensure non-negative
        
        # Tempo similarity (Gaussian)
        tempo_diff = abs(example1.estimated_tempo - example2.estimated_tempo)
        tempo_sim = np.exp(-(tempo_diff ** 2) / (2 * 15 ** 2))
        
        # Energy similarity
        energy_mapping = {'low': 0, 'medium': 1, 'high': 2}
        energy1 = energy_mapping.get(example1.energy_level, 1)
        energy2 = energy_mapping.get(example2.energy_level, 1)
        energy_sim = 1.0 - abs(energy1 - energy2) / 2.0
        
        # Difficulty similarity
        diff_mapping = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
        diff1 = diff_mapping.get(example1.difficulty, 1)
        diff2 = diff_mapping.get(example2.difficulty, 1)
        diff_sim = 1.0 - abs(diff1 - diff2) / 2.0
        
        # Weighted combination
        weighted_similarity = (
            weights.audio_weight * audio_sim +
            weights.tempo_weight * tempo_sim +
            weights.energy_weight * energy_sim +
            weights.difficulty_weight * diff_sim
        )
        
        return weighted_similarity
    
    def _calculate_ground_truth_similarity(self, example1: ValidationExample, 
                                         example2: ValidationExample) -> float:
        """Calculate ground truth similarity based on expert knowledge."""
        # Same move type = high similarity
        if example1.move_label == example2.move_label:
            base_similarity = 0.8
        else:
            # Check for related move types
            related_moves = {
                'basic_step': ['forward_backward'],
                'cross_body_lead': ['double_cross_body_lead'],
                'lady_right_turn': ['lady_left_turn'],
                'body_roll': ['arm_styling'],
            }
            
            move1_related = related_moves.get(example1.move_label, [])
            if example2.move_label in move1_related:
                base_similarity = 0.6
            else:
                base_similarity = 0.2
        
        # Adjust for difficulty and energy
        if example1.difficulty == example2.difficulty:
            base_similarity += 0.1
        
        if example1.energy_level == example2.energy_level:
            base_similarity += 0.1
        
        # Adjust for tempo compatibility
        tempo_diff = abs(example1.estimated_tempo - example2.estimated_tempo)
        if tempo_diff < 10:
            base_similarity += 0.1
        elif tempo_diff > 30:
            base_similarity -= 0.1
        
        return min(1.0, max(0.0, base_similarity))
    
    def optimize_weights(self, annotations: List[Any]) -> OptimizationResult:
        """
        Perform hyperparameter optimization to find best weights.
        
        Args:
            annotations: List of move annotations
            
        Returns:
            OptimizationResult with best weights and performance metrics
        """
        logger.info("Starting hyperparameter optimization...")
        
        # Create mock validation data (fast)
        validation_examples = self.create_mock_validation_data(annotations)
        
        # Generate weight configurations
        weight_configurations = self.generate_weight_configurations()
        
        # Evaluate each configuration
        results = []
        best_score = 0.0
        best_weights = None
        
        logger.info(f"Evaluating {len(weight_configurations)} weight configurations...")
        
        for i, weights in enumerate(weight_configurations):
            score = self.evaluate_configuration(weights, validation_examples)
            
            result = {
                'configuration_id': i,
                'weights': weights.to_dict(),
                'score': score,
                'normalized_weights': weights.to_dict()  # Already normalized
            }
            results.append(result)
            
            if score > best_score:
                best_score = score
                best_weights = weights
            
            if (i + 1) % 20 == 0:
                logger.info(f"Evaluated {i + 1}/{len(weight_configurations)} configurations. Best score so far: {best_score:.3f}")
        
        # Calculate optimization statistics
        scores = [r['score'] for r in results]
        optimization_stats = {
            'total_configurations': len(weight_configurations),
            'best_score': best_score,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'score_range': (np.min(scores), np.max(scores)),
            'top_10_percent_threshold': np.percentile(scores, 90)
        }
        
        logger.info(f"Optimization complete. Best score: {best_score:.3f}")
        logger.info(f"Best weights: {best_weights.to_dict()}")
        
        return OptimizationResult(
            best_weights=best_weights,
            best_score=best_score,
            all_results=results,
            optimization_stats=optimization_stats
        )
    
    def cross_validate_weights(self, weights: WeightConfiguration, 
                             validation_examples: List[ValidationExample]) -> Dict[str, float]:
        """
        Perform cross-validation on a specific weight configuration.
        
        Args:
            weights: Weight configuration to validate
            validation_examples: Validation examples
            
        Returns:
            Cross-validation metrics
        """
        n_folds = self.config.cross_validation_folds
        fold_size = len(validation_examples) // n_folds
        fold_scores = []
        
        for fold in range(n_folds):
            # Create train/test split for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else len(validation_examples)
            
            test_examples = validation_examples[start_idx:end_idx]
            train_examples = validation_examples[:start_idx] + validation_examples[end_idx:]
            
            # Evaluate on test fold
            fold_score = self.evaluate_configuration(weights, test_examples)
            fold_scores.append(fold_score)
        
        return {
            'mean_cv_score': np.mean(fold_scores),
            'std_cv_score': np.std(fold_scores),
            'cv_scores': fold_scores,
            'cv_confidence_interval': (
                np.mean(fold_scores) - 1.96 * np.std(fold_scores) / np.sqrt(n_folds),
                np.mean(fold_scores) + 1.96 * np.std(fold_scores) / np.sqrt(n_folds)
            )
        }
    
    def save_optimization_results(self, result: OptimizationResult, output_path: str):
        """Save optimization results to file."""
        output_data = {
            'best_weights': result.best_weights.to_dict(),
            'best_score': result.best_score,
            'optimization_stats': result.optimization_stats,
            'all_results': result.all_results,
            'config': {
                'audio_weight_range': self.config.audio_weight_range,
                'tempo_weight_range': self.config.tempo_weight_range,
                'energy_weight_range': self.config.energy_weight_range,
                'difficulty_weight_range': self.config.difficulty_weight_range,
                'grid_steps': self.config.grid_steps,
                'max_combinations': self.config.max_combinations
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {output_path}")
    
    def load_optimization_results(self, input_path: str) -> OptimizationResult:
        """Load optimization results from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        best_weights_dict = data['best_weights']
        best_weights = WeightConfiguration(
            audio_weight=best_weights_dict['audio'],
            tempo_weight=best_weights_dict['tempo'],
            energy_weight=best_weights_dict['energy'],
            difficulty_weight=best_weights_dict['difficulty']
        )
        
        return OptimizationResult(
            best_weights=best_weights,
            best_score=data['best_score'],
            all_results=data['all_results'],
            optimization_stats=data['optimization_stats']
        )


def main():
    """Main function for hyperparameter optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for recommendation weights")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--output_dir", default="data/optimization_results", help="Output directory")
    parser.add_argument("--grid_steps", type=int, default=5, help="Grid search steps")
    parser.add_argument("--max_combinations", type=int, default=100, help="Maximum combinations to test")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load annotations (simplified)
    from ..models.annotation_schema import MoveAnnotation
    import json
    
    annotations_file = Path(args.data_dir) / "bachata_annotations.json"
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    annotations = []
    for clip_data in data.get('clips', []):
        try:
            annotation = MoveAnnotation(**clip_data)
            annotations.append(annotation)
        except Exception as e:
            logger.warning(f"Failed to parse annotation: {e}")
    
    # Configure optimizer
    config = OptimizationConfig(
        grid_steps=args.grid_steps,
        max_combinations=args.max_combinations
    )
    
    optimizer = HyperparameterOptimizer(config)
    
    # Run optimization
    result = optimizer.optimize_weights(annotations)
    
    # Save results
    output_file = output_dir / "hyperparameter_optimization_results.json"
    optimizer.save_optimization_results(result, str(output_file))
    
    # Print summary
    print(f"✅ Hyperparameter optimization complete!")
    print(f"Best score: {result.best_score:.3f}")
    print(f"Best weights: {result.best_weights.to_dict()}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()