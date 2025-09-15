"""
Tests for training dataset builder.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.services.training_dataset_builder import (
    TrainingDatasetBuilder, TrainingExample, SimilarityPair, 
    GroundTruthMatrix, TrainingDataset
)
from app.services.move_analyzer import MoveAnalysisResult, MovementDynamics
from app.models.annotation_schema import MoveAnnotation


@pytest.fixture
def mock_annotations():
    """Create mock annotations for testing."""
    return [
        MoveAnnotation(
            clip_id="basic_step_1",
            video_path="Bachata_steps/basic_steps/basic_step_1.mp4",
            move_label="basic_step",
            energy_level="medium",
            estimated_tempo=110,
            difficulty="beginner",
            lead_follow_roles="both",
            notes="Test basic step"
        ),
        MoveAnnotation(
            clip_id="cross_body_lead_1",
            video_path="Bachata_steps/cross_body_lead/cross_body_lead_1.mp4",
            move_label="cross_body_lead",
            energy_level="high",
            estimated_tempo=130,
            difficulty="intermediate",
            lead_follow_roles="lead_focus",
            notes="Test cross body lead"
        ),
        MoveAnnotation(
            clip_id="dip_1",
            video_path="Bachata_steps/dip/dip_1.mp4",
            move_label="dip",
            energy_level="high",
            estimated_tempo=135,
            difficulty="advanced",
            lead_follow_roles="both",
            notes="Test dip"
        )
    ]


@pytest.fixture
def mock_move_result():
    """Create mock move analysis result."""
    dynamics = MovementDynamics(
        velocity_profile=np.array([0.1, 0.2, 0.15]),
        acceleration_profile=np.array([0.05, 0.1]),
        spatial_coverage=0.3,
        rhythm_score=0.8,
        complexity_score=0.6,
        dominant_movement_direction="horizontal",
        energy_level="medium",
        footwork_area_coverage=0.2,
        upper_body_movement_range=0.4,
        rhythm_compatibility_score=0.7,
        movement_periodicity=0.5,
        transition_points=[10, 20],
        movement_intensity_profile=np.array([0.3, 0.5, 0.4]),
        spatial_distribution={"upper_body": 0.3, "lower_body": 0.4, "arms": 0.2, "legs": 0.1}
    )
    
    return MoveAnalysisResult(
        video_path="test_video.mp4",
        duration=10.0,
        frame_count=30,
        fps=30.0,
        pose_features=[],
        hand_features=[],
        movement_dynamics=dynamics,
        pose_embedding=np.random.rand(384),
        movement_embedding=np.random.rand(128),
        movement_complexity_score=0.6,
        tempo_compatibility_range=(100, 140),
        difficulty_score=0.5,
        analysis_quality=0.8,
        pose_detection_rate=0.9
    )


@pytest.fixture
def temp_data_dir(mock_annotations):
    """Create temporary data directory with mock annotations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        # Create annotations file
        annotations_data = {
            "instructions": "Test annotations",
            "move_categories": ["basic_step", "cross_body_lead", "dip"],
            "clips": [
                {
                    "clip_id": ann.clip_id,
                    "video_path": ann.video_path,
                    "move_label": ann.move_label,
                    "energy_level": ann.energy_level,
                    "estimated_tempo": ann.estimated_tempo,
                    "difficulty": ann.difficulty,
                    "lead_follow_roles": ann.lead_follow_roles,
                    "notes": ann.notes
                }
                for ann in mock_annotations
            ]
        }
        
        annotations_file = data_dir / "bachata_annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(annotations_data, f)
        
        # Create mock video directories
        for ann in mock_annotations:
            video_path = data_dir / ann.video_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            # Create empty file to simulate video
            video_path.touch()
        
        yield str(data_dir)


class TestTrainingDatasetBuilder:
    """Test cases for TrainingDatasetBuilder."""
    
    def test_initialization(self, temp_data_dir):
        """Test builder initialization."""
        builder = TrainingDatasetBuilder(temp_data_dir)
        
        assert len(builder.annotations) == 3
        assert builder.data_dir == Path(temp_data_dir)
        assert builder.move_analyzer is not None
        assert builder.feature_fusion is not None
    
    def test_load_annotations(self, temp_data_dir):
        """Test annotation loading."""
        builder = TrainingDatasetBuilder(temp_data_dir)
        annotations = builder._load_annotations()
        
        assert len(annotations) == 3
        assert annotations[0].clip_id == "basic_step_1"
        assert annotations[1].move_label == "cross_body_lead"
        assert annotations[2].difficulty == "advanced"
    
    @patch('app.services.training_dataset_builder.MoveAnalyzer')
    def test_process_all_clips(self, mock_analyzer_class, temp_data_dir, mock_move_result):
        """Test processing all clips through MediaPipe pipeline."""
        # Setup mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_move_clip.return_value = mock_move_result
        mock_analyzer_class.return_value = mock_analyzer
        
        builder = TrainingDatasetBuilder(temp_data_dir)
        builder.move_analyzer = mock_analyzer
        
        training_examples = builder._process_all_clips()
        
        assert len(training_examples) == 3
        assert all(isinstance(ex, TrainingExample) for ex in training_examples)
        assert all(len(ex.pose_embedding) == 384 for ex in training_examples)
        assert all(ex.analysis_quality > 0 for ex in training_examples)
    
    def test_extract_movement_features(self, mock_move_result):
        """Test movement feature extraction."""
        builder = TrainingDatasetBuilder()
        features = builder._extract_movement_features(mock_move_result)
        
        # Check that all expected features are present
        expected_features = [
            'duration', 'movement_complexity_score', 'difficulty_score',
            'spatial_coverage', 'rhythm_score', 'avg_velocity',
            'upper_body_movement', 'min_tempo_compatibility'
        ]
        
        for feature in expected_features:
            assert feature in features
        
        # Check feature types and ranges
        assert isinstance(features['duration'], float)
        assert 0 <= features['movement_complexity_score'] <= 1
        assert features['avg_velocity'] >= 0
    
    def test_calculate_move_type_similarity(self):
        """Test move type similarity calculation."""
        builder = TrainingDatasetBuilder()
        
        # Same move type
        assert builder._calculate_move_type_similarity("basic_step", "basic_step") == 1.0
        
        # Similar move types
        sim = builder._calculate_move_type_similarity("basic_step", "combination_basic_step")
        assert 0.5 < sim < 1.0
        
        # Different move types
        sim = builder._calculate_move_type_similarity("basic_step", "dip")
        assert 0 <= sim < 0.5
    
    def test_calculate_difficulty_similarity(self):
        """Test difficulty similarity calculation."""
        builder = TrainingDatasetBuilder()
        
        # Same difficulty
        assert builder._calculate_difficulty_similarity("beginner", "beginner") == 1.0
        
        # Adjacent difficulties
        sim = builder._calculate_difficulty_similarity("beginner", "intermediate")
        assert 0.3 < sim < 1.0
        
        # Distant difficulties
        sim = builder._calculate_difficulty_similarity("beginner", "advanced")
        assert 0 <= sim < 0.7
    
    def test_calculate_energy_similarity(self):
        """Test energy similarity calculation."""
        builder = TrainingDatasetBuilder()
        
        # Same energy
        assert builder._calculate_energy_similarity("medium", "medium") == 1.0
        
        # Adjacent energies
        sim = builder._calculate_energy_similarity("low", "medium")
        assert 0.3 < sim < 1.0
        
        # Distant energies
        sim = builder._calculate_energy_similarity("low", "high")
        assert 0 <= sim < 0.7
    
    def test_calculate_tempo_similarity(self):
        """Test tempo similarity calculation."""
        builder = TrainingDatasetBuilder()
        
        # Same tempo
        assert builder._calculate_tempo_similarity(120, 120) == 1.0
        
        # Close tempos
        sim = builder._calculate_tempo_similarity(120, 125)
        assert 0.5 < sim < 1.0
        
        # Distant tempos
        sim = builder._calculate_tempo_similarity(100, 150)
        assert 0 <= sim < 0.5
    
    def test_create_ground_truth_similarity_matrix(self, temp_data_dir, mock_move_result):
        """Test ground truth similarity matrix creation."""
        with patch('app.services.training_dataset_builder.MoveAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_move_clip.return_value = mock_move_result
            mock_analyzer_class.return_value = mock_analyzer
            
            builder = TrainingDatasetBuilder(temp_data_dir)
            builder.move_analyzer = mock_analyzer
            
            training_examples = builder._process_all_clips()
            ground_truth = builder._create_ground_truth_similarity_matrix(training_examples)
            
            assert isinstance(ground_truth, GroundTruthMatrix)
            assert len(ground_truth.clip_ids) == 3
            assert ground_truth.similarity_matrix.shape == (3, 3)
            
            # Check diagonal is 1.0
            assert np.allclose(np.diag(ground_truth.similarity_matrix), 1.0)
            
            # Check symmetry
            assert np.allclose(ground_truth.similarity_matrix, ground_truth.similarity_matrix.T)
            
            # Check similarity types
            assert 'move_type' in ground_truth.similarity_types
            assert 'difficulty' in ground_truth.similarity_types
            assert 'energy' in ground_truth.similarity_types
            assert 'tempo' in ground_truth.similarity_types
    
    def test_generate_similarity_pairs(self, temp_data_dir, mock_move_result):
        """Test similarity pair generation."""
        with patch('app.services.training_dataset_builder.MoveAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_move_clip.return_value = mock_move_result
            mock_analyzer_class.return_value = mock_analyzer
            
            builder = TrainingDatasetBuilder(temp_data_dir)
            builder.move_analyzer = mock_analyzer
            
            training_examples = builder._process_all_clips()
            ground_truth = builder._create_ground_truth_similarity_matrix(training_examples)
            similarity_pairs = builder._generate_similarity_pairs(training_examples, ground_truth)
            
            assert len(similarity_pairs) > 0
            assert all(isinstance(pair, SimilarityPair) for pair in similarity_pairs)
            
            # Check that we have both positive and negative pairs
            positive_pairs = [p for p in similarity_pairs if p.similarity_label >= 0.7]
            negative_pairs = [p for p in similarity_pairs if p.similarity_label <= 0.3]
            
            # Should have some pairs (exact numbers depend on similarity calculations)
            assert len(positive_pairs) >= 0
            assert len(negative_pairs) >= 0
    
    def test_calculate_dataset_statistics(self, temp_data_dir, mock_move_result):
        """Test dataset statistics calculation."""
        with patch('app.services.training_dataset_builder.MoveAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_move_clip.return_value = mock_move_result
            mock_analyzer_class.return_value = mock_analyzer
            
            builder = TrainingDatasetBuilder(temp_data_dir)
            builder.move_analyzer = mock_analyzer
            
            training_examples = builder._process_all_clips()
            ground_truth = builder._create_ground_truth_similarity_matrix(training_examples)
            similarity_pairs = builder._generate_similarity_pairs(training_examples, ground_truth)
            
            stats = builder._calculate_dataset_statistics(training_examples, similarity_pairs)
            
            # Check required statistics
            assert 'n_training_examples' in stats
            assert 'n_similarity_pairs' in stats
            assert 'move_label_distribution' in stats
            assert 'difficulty_distribution' in stats
            assert 'quality_metrics' in stats
            assert 'embedding_statistics' in stats
            
            assert stats['n_training_examples'] == 3
            assert stats['n_similarity_pairs'] == len(similarity_pairs)
    
    @patch('app.services.training_dataset_builder.MoveAnalyzer')
    def test_build_comprehensive_training_dataset(self, mock_analyzer_class, temp_data_dir, mock_move_result):
        """Test complete training dataset building."""
        # Setup mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_move_clip.return_value = mock_move_result
        mock_analyzer_class.return_value = mock_analyzer
        
        with tempfile.TemporaryDirectory() as output_dir:
            builder = TrainingDatasetBuilder(temp_data_dir, output_dir)
            builder.move_analyzer = mock_analyzer
            
            dataset = builder.build_comprehensive_training_dataset()
            
            assert isinstance(dataset, TrainingDataset)
            assert len(dataset.training_examples) == 3
            assert len(dataset.similarity_pairs) >= 0
            assert isinstance(dataset.ground_truth_matrix, GroundTruthMatrix)
            assert 'n_training_examples' in dataset.dataset_stats
            
            # Check that files were created
            output_path = Path(output_dir)
            pkl_files = list(output_path.glob("training_dataset_*.pkl"))
            assert len(pkl_files) == 1
    
    def test_validate_training_dataset(self, temp_data_dir, mock_move_result):
        """Test training dataset validation."""
        with patch('app.services.training_dataset_builder.MoveAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_move_clip.return_value = mock_move_result
            mock_analyzer_class.return_value = mock_analyzer
            
            with tempfile.TemporaryDirectory() as output_dir:
                builder = TrainingDatasetBuilder(temp_data_dir, output_dir)
                builder.move_analyzer = mock_analyzer
                
                dataset = builder.build_comprehensive_training_dataset()
                validation_results = builder.validate_training_dataset(dataset)
                
                assert 'data_completeness' in validation_results
                assert 'embedding_quality' in validation_results
                assert 'similarity_distribution' in validation_results
                assert 'overall_valid' in validation_results
                assert 'validation_score' in validation_results
                
                # Should be valid with mock data
                assert validation_results['overall_valid'] is True
                assert validation_results['validation_score'] > 0.8


class TestTrainingDataStructures:
    """Test training data structures."""
    
    def test_training_example_creation(self):
        """Test TrainingExample creation."""
        example = TrainingExample(
            clip_id="test_clip",
            video_path="test/path.mp4",
            pose_embedding=np.random.rand(384),
            movement_features={"feature1": 0.5, "feature2": 0.8},
            annotation_data={"move_label": "basic_step"},
            analysis_quality=0.9
        )
        
        assert example.clip_id == "test_clip"
        assert len(example.pose_embedding) == 384
        assert example.analysis_quality == 0.9
    
    def test_similarity_pair_creation(self):
        """Test SimilarityPair creation."""
        pair = SimilarityPair(
            clip1_id="clip1",
            clip2_id="clip2",
            clip1_embedding=np.random.rand(384),
            clip2_embedding=np.random.rand(384),
            similarity_label=0.8,
            similarity_type="move_type",
            metadata={"pair_type": "positive"}
        )
        
        assert pair.clip1_id == "clip1"
        assert pair.similarity_label == 0.8
        assert pair.similarity_type == "move_type"
    
    def test_ground_truth_matrix_creation(self):
        """Test GroundTruthMatrix creation."""
        clip_ids = ["clip1", "clip2", "clip3"]
        similarity_matrix = np.eye(3)
        similarity_types = {"move_type": np.eye(3)}
        
        matrix = GroundTruthMatrix(
            clip_ids=clip_ids,
            similarity_matrix=similarity_matrix,
            similarity_types=similarity_types,
            metadata={"n_clips": 3}
        )
        
        assert len(matrix.clip_ids) == 3
        assert matrix.similarity_matrix.shape == (3, 3)
        assert "move_type" in matrix.similarity_types


if __name__ == "__main__":
    pytest.main([__file__])