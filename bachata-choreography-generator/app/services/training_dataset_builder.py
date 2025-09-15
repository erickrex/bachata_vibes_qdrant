"""
Training dataset builder for comprehensive training data preparation.
Processes all annotated move clips through MediaPipe pose detection pipeline
and creates training datasets with positive/negative pairs for similarity learning.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
import pickle
from itertools import combinations

from .move_analyzer import MoveAnalyzer, MoveAnalysisResult
from .feature_fusion import FeatureFusion, MultiModalEmbedding
from .music_analyzer import MusicAnalyzer, MusicFeatures
from ..models.annotation_schema import MoveAnnotation

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example with features and labels."""
    clip_id: str
    video_path: str
    pose_embedding: np.ndarray
    movement_features: Dict[str, float]
    annotation_data: Dict[str, Any]
    analysis_quality: float


@dataclass
class SimilarityPair:
    """Pair of clips with similarity label for training."""
    clip1_id: str
    clip2_id: str
    clip1_embedding: np.ndarray
    clip2_embedding: np.ndarray
    similarity_label: float  # 0.0 to 1.0
    similarity_type: str     # 'move_type', 'difficulty', 'energy', 'tempo'
    metadata: Dict[str, Any]


@dataclass
class GroundTruthMatrix:
    """Ground truth similarity matrix based on expert annotations."""
    clip_ids: List[str]
    similarity_matrix: np.ndarray
    similarity_types: Dict[str, np.ndarray]  # Different similarity matrices by type
    metadata: Dict[str, Any]


@dataclass
class TrainingDataset:
    """Complete training dataset with examples and similarity pairs."""
    training_examples: List[TrainingExample]
    similarity_pairs: List[SimilarityPair]
    ground_truth_matrix: GroundTruthMatrix
    dataset_stats: Dict[str, Any]
    creation_timestamp: str


class TrainingDatasetBuilder:
    """
    Builds comprehensive training datasets from annotated move clips.
    Processes clips through MediaPipe pipeline and creates similarity learning data.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 output_dir: str = "data/training_datasets"):
        """
        Initialize the training dataset builder.
        
        Args:
            data_dir: Directory containing video clips and annotations
            output_dir: Directory to save training datasets
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize analyzers
        self.move_analyzer = MoveAnalyzer(target_fps=30)
        self.feature_fusion = FeatureFusion()
        self.music_analyzer = MusicAnalyzer()
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        logger.info(f"TrainingDatasetBuilder initialized with {len(self.annotations)} annotations")
    
    def build_comprehensive_training_dataset(self) -> TrainingDataset:
        """
        Build complete training dataset from all annotated move clips.
        
        Returns:
            TrainingDataset with processed examples and similarity pairs
        """
        logger.info("Building comprehensive training dataset...")
        
        # Step 1: Process all clips through MediaPipe pipeline
        training_examples = self._process_all_clips()
        
        # Step 2: Create ground truth similarity matrices
        ground_truth_matrix = self._create_ground_truth_similarity_matrix(training_examples)
        
        # Step 3: Generate positive/negative pairs for similarity learning
        similarity_pairs = self._generate_similarity_pairs(training_examples, ground_truth_matrix)
        
        # Step 4: Calculate dataset statistics
        dataset_stats = self._calculate_dataset_statistics(training_examples, similarity_pairs)
        
        # Create final dataset
        dataset = TrainingDataset(
            training_examples=training_examples,
            similarity_pairs=similarity_pairs,
            ground_truth_matrix=ground_truth_matrix,
            dataset_stats=dataset_stats,
            creation_timestamp=pd.Timestamp.now().isoformat()
        )
        
        # Save dataset
        self._save_training_dataset(dataset)
        
        logger.info(f"Training dataset created with {len(training_examples)} examples and {len(similarity_pairs)} pairs")
        return dataset
    
    def _load_annotations(self) -> List[MoveAnnotation]:
        """Load and validate annotations from JSON file."""
        annotations_file = self.data_dir / "bachata_annotations.json"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        annotations = []
        for clip_data in data.get('clips', []):
            try:
                annotation = MoveAnnotation(**clip_data)
                annotations.append(annotation)
            except Exception as e:
                logger.warning(f"Failed to parse annotation for {clip_data.get('clip_id', 'unknown')}: {e}")
        
        logger.info(f"Loaded {len(annotations)} valid annotations")
        return annotations
    
    def _process_all_clips(self) -> List[TrainingExample]:
        """Process all annotated clips through MediaPipe pose detection pipeline."""
        training_examples = []
        
        logger.info("Processing all clips through MediaPipe pipeline...")
        
        for annotation in tqdm(self.annotations, desc="Processing clips"):
            try:
                # Construct full video path
                video_path = self.data_dir / annotation.video_path
                
                if not video_path.exists():
                    logger.warning(f"Video file not found: {video_path}")
                    continue
                
                # Analyze move clip
                move_result = self.move_analyzer.analyze_move_clip(str(video_path))
                
                # Extract movement features for training
                movement_features = self._extract_movement_features(move_result)
                
                # Create training example
                example = TrainingExample(
                    clip_id=annotation.clip_id,
                    video_path=str(video_path),
                    pose_embedding=move_result.pose_embedding,
                    movement_features=movement_features,
                    annotation_data=annotation.dict(),  # Use .dict() for Pydantic models
                    analysis_quality=move_result.analysis_quality
                )
                
                training_examples.append(example)
                
            except Exception as e:
                logger.error(f"Failed to process clip {annotation.clip_id}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(training_examples)} clips")
        return training_examples
    
    def _extract_movement_features(self, move_result: MoveAnalysisResult) -> Dict[str, float]:
        """Extract key movement features for training."""
        dynamics = move_result.movement_dynamics
        
        features = {
            # Basic movement metrics
            'duration': move_result.duration,
            'movement_complexity_score': move_result.movement_complexity_score,
            'difficulty_score': move_result.difficulty_score,
            'analysis_quality': move_result.analysis_quality,
            'pose_detection_rate': move_result.pose_detection_rate,
            
            # Movement dynamics
            'spatial_coverage': dynamics.spatial_coverage,
            'rhythm_score': dynamics.rhythm_score,
            'complexity_score': dynamics.complexity_score,
            'footwork_area_coverage': dynamics.footwork_area_coverage,
            'upper_body_movement_range': dynamics.upper_body_movement_range,
            'rhythm_compatibility_score': dynamics.rhythm_compatibility_score,
            'movement_periodicity': dynamics.movement_periodicity,
            
            # Velocity and acceleration statistics
            'avg_velocity': np.mean(dynamics.velocity_profile),
            'max_velocity': np.max(dynamics.velocity_profile),
            'velocity_std': np.std(dynamics.velocity_profile),
            'avg_acceleration': np.mean(dynamics.acceleration_profile) if len(dynamics.acceleration_profile) > 0 else 0.0,
            'max_acceleration': np.max(dynamics.acceleration_profile) if len(dynamics.acceleration_profile) > 0 else 0.0,
            
            # Spatial distribution
            'upper_body_movement': dynamics.spatial_distribution.get('upper_body', 0.0),
            'lower_body_movement': dynamics.spatial_distribution.get('lower_body', 0.0),
            'arm_movement': dynamics.spatial_distribution.get('arms', 0.0),
            'leg_movement': dynamics.spatial_distribution.get('legs', 0.0),
            
            # Tempo compatibility
            'min_tempo_compatibility': move_result.tempo_compatibility_range[0],
            'max_tempo_compatibility': move_result.tempo_compatibility_range[1],
            'tempo_range_width': move_result.tempo_compatibility_range[1] - move_result.tempo_compatibility_range[0],
            
            # Transition and intensity features
            'num_transition_points': len(dynamics.transition_points),
            'avg_movement_intensity': np.mean(dynamics.movement_intensity_profile),
            'max_movement_intensity': np.max(dynamics.movement_intensity_profile),
            'intensity_variance': np.var(dynamics.movement_intensity_profile),
            
            # Energy level encoding
            'energy_level_low': 1.0 if dynamics.energy_level == 'low' else 0.0,
            'energy_level_medium': 1.0 if dynamics.energy_level == 'medium' else 0.0,
            'energy_level_high': 1.0 if dynamics.energy_level == 'high' else 0.0,
            
            # Movement direction encoding
            'movement_horizontal': 1.0 if 'horizontal' in dynamics.dominant_movement_direction else 0.0,
            'movement_vertical': 1.0 if 'vertical' in dynamics.dominant_movement_direction else 0.0,
        }
        
        return features
    
    def _create_ground_truth_similarity_matrix(self, training_examples: List[TrainingExample]) -> GroundTruthMatrix:
        """Create ground truth similarity matrices based on expert annotations."""
        n_clips = len(training_examples)
        clip_ids = [ex.clip_id for ex in training_examples]
        
        # Initialize similarity matrices
        overall_similarity = np.zeros((n_clips, n_clips))
        move_type_similarity = np.zeros((n_clips, n_clips))
        difficulty_similarity = np.zeros((n_clips, n_clips))
        energy_similarity = np.zeros((n_clips, n_clips))
        tempo_similarity = np.zeros((n_clips, n_clips))
        
        logger.info("Creating ground truth similarity matrices...")
        
        for i in range(n_clips):
            for j in range(n_clips):
                if i == j:
                    # Self-similarity is 1.0
                    overall_similarity[i, j] = 1.0
                    move_type_similarity[i, j] = 1.0
                    difficulty_similarity[i, j] = 1.0
                    energy_similarity[i, j] = 1.0
                    tempo_similarity[i, j] = 1.0
                else:
                    # Calculate similarities based on annotations
                    ex1 = training_examples[i]
                    ex2 = training_examples[j]
                    
                    # Move type similarity
                    move_sim = self._calculate_move_type_similarity(
                        ex1.annotation_data['move_label'],
                        ex2.annotation_data['move_label']
                    )
                    move_type_similarity[i, j] = move_sim
                    
                    # Difficulty similarity
                    diff_sim = self._calculate_difficulty_similarity(
                        ex1.annotation_data['difficulty'],
                        ex2.annotation_data['difficulty']
                    )
                    difficulty_similarity[i, j] = diff_sim
                    
                    # Energy similarity
                    energy_sim = self._calculate_energy_similarity(
                        ex1.annotation_data['energy_level'],
                        ex2.annotation_data['energy_level']
                    )
                    energy_similarity[i, j] = energy_sim
                    
                    # Tempo similarity
                    tempo_sim = self._calculate_tempo_similarity(
                        ex1.annotation_data['estimated_tempo'],
                        ex2.annotation_data['estimated_tempo']
                    )
                    tempo_similarity[i, j] = tempo_sim
                    
                    # Overall similarity (weighted combination)
                    overall_similarity[i, j] = (
                        0.4 * move_sim +
                        0.2 * diff_sim +
                        0.2 * energy_sim +
                        0.2 * tempo_sim
                    )
        
        similarity_types = {
            'move_type': move_type_similarity,
            'difficulty': difficulty_similarity,
            'energy': energy_similarity,
            'tempo': tempo_similarity
        }
        
        # Calculate matrix statistics
        metadata = {
            'n_clips': n_clips,
            'mean_similarity': np.mean(overall_similarity[np.triu_indices(n_clips, k=1)]),
            'std_similarity': np.std(overall_similarity[np.triu_indices(n_clips, k=1)]),
            'similarity_distribution': {
                'move_type': {
                    'mean': np.mean(move_type_similarity[np.triu_indices(n_clips, k=1)]),
                    'std': np.std(move_type_similarity[np.triu_indices(n_clips, k=1)])
                },
                'difficulty': {
                    'mean': np.mean(difficulty_similarity[np.triu_indices(n_clips, k=1)]),
                    'std': np.std(difficulty_similarity[np.triu_indices(n_clips, k=1)])
                },
                'energy': {
                    'mean': np.mean(energy_similarity[np.triu_indices(n_clips, k=1)]),
                    'std': np.std(energy_similarity[np.triu_indices(n_clips, k=1)])
                },
                'tempo': {
                    'mean': np.mean(tempo_similarity[np.triu_indices(n_clips, k=1)]),
                    'std': np.std(tempo_similarity[np.triu_indices(n_clips, k=1)])
                }
            }
        }
        
        return GroundTruthMatrix(
            clip_ids=clip_ids,
            similarity_matrix=overall_similarity,
            similarity_types=similarity_types,
            metadata=metadata
        )
    
    def _calculate_move_type_similarity(self, move1: str, move2: str) -> float:
        """Calculate similarity between move types."""
        if move1 == move2:
            return 1.0
        
        # Define move type hierarchies and similarities
        move_similarities = {
            # Basic moves are similar to each other
            ('basic_step', 'combination_basic_step'): 0.8,
            ('basic_step', 'forward_backward'): 0.6,
            
            # Turn moves are similar
            ('lady_right_turn', 'lady_left_turn'): 0.7,
            ('cross_body_lead', 'double_cross_body_lead'): 0.8,
            
            # Advanced moves have some similarity
            ('body_roll', 'arm_styling'): 0.4,
            ('hammerlock', 'shadow_position'): 0.3,
            ('dip', 'shadow_position'): 0.3,
            
            # Combination moves are somewhat similar to component moves
            ('combination', 'cross_body_lead'): 0.4,
            ('combination', 'lady_right_turn'): 0.4,
            ('combination', 'body_roll'): 0.4,
        }
        
        # Check both directions
        pair = (move1, move2)
        reverse_pair = (move2, move1)
        
        if pair in move_similarities:
            return move_similarities[pair]
        elif reverse_pair in move_similarities:
            return move_similarities[reverse_pair]
        else:
            return 0.1  # Default low similarity for different move types
    
    def _calculate_difficulty_similarity(self, diff1: str, diff2: str) -> float:
        """Calculate similarity between difficulty levels."""
        difficulty_order = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
        
        if diff1 == diff2:
            return 1.0
        
        d1 = difficulty_order.get(diff1, 1)
        d2 = difficulty_order.get(diff2, 1)
        
        # Linear similarity based on difficulty distance
        distance = abs(d1 - d2)
        return max(0.0, 1.0 - distance * 0.5)
    
    def _calculate_energy_similarity(self, energy1: str, energy2: str) -> float:
        """Calculate similarity between energy levels."""
        energy_order = {'low': 0, 'medium': 1, 'high': 2}
        
        if energy1 == energy2:
            return 1.0
        
        e1 = energy_order.get(energy1, 1)
        e2 = energy_order.get(energy2, 1)
        
        # Linear similarity based on energy distance
        distance = abs(e1 - e2)
        return max(0.0, 1.0 - distance * 0.5)
    
    def _calculate_tempo_similarity(self, tempo1: float, tempo2: float) -> float:
        """Calculate similarity between tempos."""
        tempo_diff = abs(tempo1 - tempo2)
        
        # Gaussian similarity with sigma=15 BPM
        similarity = np.exp(-(tempo_diff ** 2) / (2 * 15 ** 2))
        return similarity
    
    def _generate_similarity_pairs(self, 
                                 training_examples: List[TrainingExample],
                                 ground_truth_matrix: GroundTruthMatrix) -> List[SimilarityPair]:
        """Generate positive/negative pairs for similarity learning."""
        similarity_pairs = []
        
        logger.info("Generating similarity pairs for training...")
        
        n_clips = len(training_examples)
        
        # Generate pairs for each similarity type
        for sim_type, sim_matrix in ground_truth_matrix.similarity_types.items():
            
            # Generate positive pairs (high similarity)
            positive_threshold = 0.7
            positive_pairs = []
            
            for i in range(n_clips):
                for j in range(i + 1, n_clips):
                    if sim_matrix[i, j] >= positive_threshold:
                        pair = SimilarityPair(
                            clip1_id=training_examples[i].clip_id,
                            clip2_id=training_examples[j].clip_id,
                            clip1_embedding=training_examples[i].pose_embedding,
                            clip2_embedding=training_examples[j].pose_embedding,
                            similarity_label=sim_matrix[i, j],
                            similarity_type=sim_type,
                            metadata={
                                'pair_type': 'positive',
                                'clip1_annotation': training_examples[i].annotation_data,
                                'clip2_annotation': training_examples[j].annotation_data
                            }
                        )
                        positive_pairs.append(pair)
            
            # Generate negative pairs (low similarity)
            negative_threshold = 0.3
            negative_pairs = []
            
            for i in range(n_clips):
                for j in range(i + 1, n_clips):
                    if sim_matrix[i, j] <= negative_threshold:
                        pair = SimilarityPair(
                            clip1_id=training_examples[i].clip_id,
                            clip2_id=training_examples[j].clip_id,
                            clip1_embedding=training_examples[i].pose_embedding,
                            clip2_embedding=training_examples[j].pose_embedding,
                            similarity_label=sim_matrix[i, j],
                            similarity_type=sim_type,
                            metadata={
                                'pair_type': 'negative',
                                'clip1_annotation': training_examples[i].annotation_data,
                                'clip2_annotation': training_examples[j].annotation_data
                            }
                        )
                        negative_pairs.append(pair)
            
            # Balance positive and negative pairs
            max_pairs_per_type = min(len(positive_pairs), len(negative_pairs), 50)
            
            # Randomly sample to balance
            if len(positive_pairs) > max_pairs_per_type:
                positive_pairs = np.random.choice(positive_pairs, max_pairs_per_type, replace=False).tolist()
            if len(negative_pairs) > max_pairs_per_type:
                negative_pairs = np.random.choice(negative_pairs, max_pairs_per_type, replace=False).tolist()
            
            similarity_pairs.extend(positive_pairs)
            similarity_pairs.extend(negative_pairs)
            
            logger.info(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs for {sim_type}")
        
        logger.info(f"Total similarity pairs generated: {len(similarity_pairs)}")
        return similarity_pairs
    
    def _calculate_dataset_statistics(self, 
                                    training_examples: List[TrainingExample],
                                    similarity_pairs: List[SimilarityPair]) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics."""
        stats = {}
        
        # Basic counts
        stats['n_training_examples'] = len(training_examples)
        stats['n_similarity_pairs'] = len(similarity_pairs)
        
        # Annotation distribution
        move_labels = [ex.annotation_data['move_label'] for ex in training_examples]
        difficulties = [ex.annotation_data['difficulty'] for ex in training_examples]
        energy_levels = [ex.annotation_data['energy_level'] for ex in training_examples]
        tempos = [ex.annotation_data['estimated_tempo'] for ex in training_examples]
        
        stats['move_label_distribution'] = {label: move_labels.count(label) for label in set(move_labels)}
        stats['difficulty_distribution'] = {diff: difficulties.count(diff) for diff in set(difficulties)}
        stats['energy_level_distribution'] = {energy: energy_levels.count(energy) for energy in set(energy_levels)}
        
        if tempos:
            stats['tempo_statistics'] = {
                'mean': np.mean(tempos),
                'std': np.std(tempos),
                'min': np.min(tempos),
                'max': np.max(tempos),
                'median': np.median(tempos)
            }
        else:
            stats['tempo_statistics'] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        # Quality metrics
        analysis_qualities = [ex.analysis_quality for ex in training_examples]
        pose_detection_rates = [ex.movement_features.get('pose_detection_rate', 0.0) for ex in training_examples]
        
        if analysis_qualities:
            stats['quality_metrics'] = {
                'mean_analysis_quality': np.mean(analysis_qualities),
                'std_analysis_quality': np.std(analysis_qualities),
                'mean_pose_detection_rate': np.mean(pose_detection_rates),
                'std_pose_detection_rate': np.std(pose_detection_rates),
                'high_quality_clips': sum(1 for q in analysis_qualities if q >= 0.8),
                'low_quality_clips': sum(1 for q in analysis_qualities if q < 0.5)
            }
        else:
            stats['quality_metrics'] = {
                'mean_analysis_quality': 0.0,
                'std_analysis_quality': 0.0,
                'mean_pose_detection_rate': 0.0,
                'std_pose_detection_rate': 0.0,
                'high_quality_clips': 0,
                'low_quality_clips': 0
            }
        
        # Movement feature statistics
        movement_features = {}
        if training_examples:
            feature_names = list(training_examples[0].movement_features.keys())
            
            for feature_name in feature_names:
                values = [ex.movement_features[feature_name] for ex in training_examples]
                movement_features[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        stats['movement_feature_statistics'] = movement_features
        
        # Similarity pair statistics
        pair_types = [pair.similarity_type for pair in similarity_pairs]
        similarity_labels = [pair.similarity_label for pair in similarity_pairs]
        
        if similarity_pairs:
            stats['similarity_pair_statistics'] = {
                'pair_type_distribution': {ptype: pair_types.count(ptype) for ptype in set(pair_types)},
                'similarity_label_statistics': {
                    'mean': np.mean(similarity_labels),
                    'std': np.std(similarity_labels),
                    'min': np.min(similarity_labels),
                    'max': np.max(similarity_labels)
                },
                'positive_pairs': sum(1 for label in similarity_labels if label >= 0.7),
                'negative_pairs': sum(1 for label in similarity_labels if label <= 0.3),
                'neutral_pairs': sum(1 for label in similarity_labels if 0.3 < label < 0.7)
            }
        else:
            stats['similarity_pair_statistics'] = {
                'pair_type_distribution': {},
                'similarity_label_statistics': {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                },
                'positive_pairs': 0,
                'negative_pairs': 0,
                'neutral_pairs': 0
            }
        
        # Embedding statistics
        if training_examples:
            embeddings = np.vstack([ex.pose_embedding for ex in training_examples])
            stats['embedding_statistics'] = {
                'embedding_dimension': embeddings.shape[1],
                'mean_embedding_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
                'std_embedding_norm': np.std(np.linalg.norm(embeddings, axis=1)),
                'embedding_variance': np.mean(np.var(embeddings, axis=0)),
                'non_zero_dimensions': np.sum(np.var(embeddings, axis=0) > 1e-6)
            }
        else:
            stats['embedding_statistics'] = {
                'embedding_dimension': 0,
                'mean_embedding_norm': 0.0,
                'std_embedding_norm': 0.0,
                'embedding_variance': 0.0,
                'non_zero_dimensions': 0
            }
        
        return stats
    
    def _save_training_dataset(self, dataset: TrainingDataset):
        """Save training dataset to disk."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main dataset
        dataset_file = self.output_dir / f"training_dataset_{timestamp}.pkl"
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save human-readable statistics
        stats_file = self.output_dir / f"dataset_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(dataset.dataset_stats, f, indent=2, default=str)
        
        # Save similarity matrix as CSV for inspection
        matrix_file = self.output_dir / f"similarity_matrix_{timestamp}.csv"
        df = pd.DataFrame(
            dataset.ground_truth_matrix.similarity_matrix,
            index=dataset.ground_truth_matrix.clip_ids,
            columns=dataset.ground_truth_matrix.clip_ids
        )
        df.to_csv(matrix_file)
        
        # Save training examples as CSV
        examples_data = []
        for ex in dataset.training_examples:
            row = {
                'clip_id': ex.clip_id,
                'video_path': ex.video_path,
                'analysis_quality': ex.analysis_quality,
                **ex.movement_features,
                **ex.annotation_data
            }
            examples_data.append(row)
        
        examples_file = self.output_dir / f"training_examples_{timestamp}.csv"
        pd.DataFrame(examples_data).to_csv(examples_file, index=False)
        
        logger.info(f"Training dataset saved to {dataset_file}")
        logger.info(f"Dataset statistics saved to {stats_file}")
        logger.info(f"Similarity matrix saved to {matrix_file}")
        logger.info(f"Training examples saved to {examples_file}")
    
    def load_training_dataset(self, dataset_file: str) -> TrainingDataset:
        """Load training dataset from disk."""
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        
        logger.info(f"Loaded training dataset with {len(dataset.training_examples)} examples")
        return dataset
    
    def validate_training_dataset(self, dataset: TrainingDataset) -> Dict[str, Any]:
        """Validate training dataset quality and completeness."""
        validation_results = {}
        
        # Check data completeness
        validation_results['data_completeness'] = {
            'all_examples_have_embeddings': all(
                ex.pose_embedding is not None and len(ex.pose_embedding) > 0 
                for ex in dataset.training_examples
            ),
            'all_examples_have_features': all(
                len(ex.movement_features) > 0 
                for ex in dataset.training_examples
            ),
            'all_pairs_have_embeddings': all(
                pair.clip1_embedding is not None and pair.clip2_embedding is not None
                for pair in dataset.similarity_pairs
            )
        }
        
        # Check embedding quality
        embeddings = np.vstack([ex.pose_embedding for ex in dataset.training_examples])
        validation_results['embedding_quality'] = {
            'consistent_dimensions': len(set(len(ex.pose_embedding) for ex in dataset.training_examples)) == 1,
            'no_nan_values': not np.any(np.isnan(embeddings)),
            'no_inf_values': not np.any(np.isinf(embeddings)),
            'reasonable_variance': np.mean(np.var(embeddings, axis=0)) > 1e-6
        }
        
        # Check similarity distribution
        similarity_labels = [pair.similarity_label for pair in dataset.similarity_pairs]
        validation_results['similarity_distribution'] = {
            'has_positive_pairs': any(label >= 0.7 for label in similarity_labels),
            'has_negative_pairs': any(label <= 0.3 for label in similarity_labels),
            'balanced_distribution': abs(
                sum(1 for label in similarity_labels if label >= 0.7) - 
                sum(1 for label in similarity_labels if label <= 0.3)
            ) < len(similarity_labels) * 0.2
        }
        
        # Overall validation
        all_checks = []
        for category in validation_results.values():
            all_checks.extend(category.values())
        
        validation_results['overall_valid'] = all(all_checks)
        validation_results['validation_score'] = sum(all_checks) / len(all_checks)
        
        return validation_results


def main():
    """Main function to build training dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build comprehensive training dataset")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--output_dir", default="data/training_datasets", help="Output directory")
    
    args = parser.parse_args()
    
    # Build training dataset
    builder = TrainingDatasetBuilder(args.data_dir, args.output_dir)
    dataset = builder.build_comprehensive_training_dataset()
    
    # Validate dataset
    validation_results = builder.validate_training_dataset(dataset)
    print(f"Dataset validation results: {validation_results}")
    
    if validation_results['overall_valid']:
        print("✅ Training dataset successfully created and validated!")
    else:
        print("⚠️ Training dataset created but has validation issues")
        print(f"Validation score: {validation_results['validation_score']:.2f}")


if __name__ == "__main__":
    main()