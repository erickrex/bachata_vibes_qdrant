"""
Tests for hyperparameter optimizer.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.hyperparameter_optimizer import (
    HyperparameterOptimizer, OptimizationConfig, WeightConfiguration,
    ValidationExample, OptimizationResult
)
from app.models.annotation_schema import MoveAnnotation


@pytest.fixture
def mock_annotations():
    """Create mock annotations for testing."""
    return [
        MoveAnnotation(
            clip_id="basic_step_1",
            video_path="test/basic_step_1.mp4",
            move_label="basic_step",
            energy_level="medium",
            estimated_tempo=110,
            difficulty="beginner",
            lead_follow_roles="both",
            notes="Test basic step"
        ),
        MoveAnnotation(
            clip_id="cross_body_lead_1",
            video_path="test/cross_body_lead_1.mp4",
            move_label="cross_body_lead",
            energy_level="high",
            estimated_tempo=130,
            difficulty="intermediate",
            lead_follow_roles="lead_focus",
            notes="Test cross body lead"
        ),
        MoveAnnotation(
            clip_id="dip_1",
            video_path="test/dip_1.mp4",
            move_label="dip",
            energy_level="high",
            estimated_tempo=135,
            difficulty="advanced",
            lead_follow_roles="both",
            notes="Test dip"
        )
    ]


@pytest.fixture
def optimization_config():
    """Create test optimization configuration."""
    return OptimizationConfig(
        grid_steps=3,
        max_combinations=10,
        validation_split=0.2,
        cross_validation_folds=2
    )


class TestWeightConfiguration:
    """Test cases for WeightConfiguration."""
    
    def test_weight_configuration_creation(self):
        """Test weight configuration creation."""
        weights = WeightConfiguration(0.4, 0.3, 0.2, 0.1)
        
        assert weights.audio_weight == 0.4
        assert weights.tempo_weight == 0.3
        assert weights.energy_weight == 0.2
        assert weights.difficulty_weight == 0.1
        assert weights.total_weight == 1.0
    
    def test_weight_normalization(self):
        """Test weight normalization."""
        weights = WeightConfiguration(0.8, 0.6, 0.4, 0.2)  # Sum = 2.0
        normalized = weights.normalize()
        
        assert abs(normalized.total_weight - 1.0) < 1e-6
        assert normalized.audio_weight == 0.4
        assert normalized.tempo_weight == 0.3
        assert normalized.energy_weight == 0.2
        assert normalized.difficulty_weight == 0.1
    
    def test_weight_to_dict(self):
        """Test weight conversion to dictionary."""
        weights = WeightConfiguration(0.4, 0.3, 0.2, 0.1)
        weight_dict = weights.to_dict()
        
        expected = {
            'audio': 0.4,
            'tempo': 0.3,
            'energy': 0.2,
            'difficulty': 0.1
        }
        
        assert weight_dict == expected


class TestHyperparameterOptimizer:
    """Test cases for HyperparameterOptimizer."""
    
    def test_initialization(self, optimization_config):
        """Test optimizer initialization."""
        optimizer = HyperparameterOptimizer(optimization_config)
        
        assert optimizer.config == optimization_config
        assert len(optimizer.validation_examples) == 0
    
    def test_create_mock_validation_data(self, optimization_config, mock_annotations):
        """Test mock validation data creation."""
        optimizer = HyperparameterOptimizer(optimization_config)
        validation_examples = optimizer.create_mock_validation_data(mock_annotations)
        
        assert len(validation_examples) == 3
        assert all(isinstance(ex, ValidationExample) for ex in validation_examples)
        assert all(len(ex.audio_features) == 128 for ex in validation_examples)
        assert all(0 <= ex.movement_complexity <= 1 for ex in validation_examples)
        assert all(0 <= ex.rhythm_score <= 1 for ex in validation_examples)
    
    def test_generate_weight_configurations(self, optimization_config):
        """Test weight configuration generation."""
        optimizer = HyperparameterOptimizer(optimization_config)
        configurations = optimizer.generate_weight_configurations()
        
        assert len(configurations) <= optimization_config.max_combinations
        assert all(isinstance(config, WeightConfiguration) for config in configurations)
        assert all(abs(config.total_weight - 1.0) < 1e-6 for config in configurations)
    
    def test_evaluate_configuration(self, optimization_config, mock_annotations):
        """Test configuration evaluation."""
        optimizer = HyperparameterOptimizer(optimization_config)
        validation_examples = optimizer.create_mock_validation_data(mock_annotations)
        
        weights = WeightConfiguration(0.4, 0.3, 0.2, 0.1)
        score = optimizer.evaluate_configuration(weights, validation_examples)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)
    
    def test_calculate_weighted_similarity(self, optimization_config, mock_annotations):
        """Test weighted similarity calculation."""
        optimizer = HyperparameterOptimizer(optimization_config)
        validation_examples = optimizer.create_mock_validation_data(mock_annotations)
        
        weights = WeightConfiguration(0.4, 0.3, 0.2, 0.1)
        similarity = optimizer._calculate_weighted_similarity(
            validation_examples[0], validation_examples[1], weights
        )
        
        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)
    
    def test_calculate_ground_truth_similarity(self, optimization_config, mock_annotations):
        """Test ground truth similarity calculation."""
        optimizer = HyperparameterOptimizer(optimization_config)
        validation_examples = optimizer.create_mock_validation_data(mock_annotations)
        
        # Same move type should have high similarity
        same_move_sim = optimizer._calculate_ground_truth_similarity(
            validation_examples[0], validation_examples[0]
        )
        assert same_move_sim >= 0.8
        
        # Different move types should have lower similarity
        diff_move_sim = optimizer._calculate_ground_truth_similarity(
            validation_examples[0], validation_examples[1]
        )
        assert diff_move_sim < same_move_sim
    
    def test_optimize_weights(self, optimization_config, mock_annotations):
        """Test weight optimization."""
        optimizer = HyperparameterOptimizer(optimization_config)
        result = optimizer.optimize_weights(mock_annotations)
        
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_weights, WeightConfiguration)
        assert 0 <= result.best_score <= 1
        assert len(result.all_results) > 0
        assert 'total_configurations' in result.optimization_stats
        assert 'best_score' in result.optimization_stats
    
    def test_cross_validate_weights(self, optimization_config, mock_annotations):
        """Test cross-validation."""
        optimizer = HyperparameterOptimizer(optimization_config)
        validation_examples = optimizer.create_mock_validation_data(mock_annotations)
        
        weights = WeightConfiguration(0.4, 0.3, 0.2, 0.1)
        cv_results = optimizer.cross_validate_weights(weights, validation_examples)
        
        assert 'mean_cv_score' in cv_results
        assert 'std_cv_score' in cv_results
        assert 'cv_scores' in cv_results
        assert 'cv_confidence_interval' in cv_results
        
        assert 0 <= cv_results['mean_cv_score'] <= 1
        assert len(cv_results['cv_scores']) == optimization_config.cross_validation_folds
    
    def test_save_and_load_optimization_results(self, optimization_config, mock_annotations):
        """Test saving and loading optimization results."""
        optimizer = HyperparameterOptimizer(optimization_config)
        result = optimizer.optimize_weights(mock_annotations)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save results
            optimizer.save_optimization_results(result, temp_path)
            assert Path(temp_path).exists()
            
            # Load results
            loaded_result = optimizer.load_optimization_results(temp_path)
            
            assert isinstance(loaded_result, OptimizationResult)
            assert loaded_result.best_score == result.best_score
            assert loaded_result.best_weights.to_dict() == result.best_weights.to_dict()
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


class TestValidationExample:
    """Test cases for ValidationExample."""
    
    def test_validation_example_creation(self):
        """Test validation example creation."""
        audio_features = np.random.rand(128)
        
        example = ValidationExample(
            clip_id="test_clip",
            move_label="basic_step",
            difficulty="beginner",
            energy_level="medium",
            estimated_tempo=120,
            audio_features=audio_features,
            movement_complexity=0.5,
            rhythm_score=0.8
        )
        
        assert example.clip_id == "test_clip"
        assert example.move_label == "basic_step"
        assert example.difficulty == "beginner"
        assert example.energy_level == "medium"
        assert example.estimated_tempo == 120
        assert len(example.audio_features) == 128
        assert example.movement_complexity == 0.5
        assert example.rhythm_score == 0.8


class TestOptimizationConfig:
    """Test cases for OptimizationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert config.audio_weight_range == (0.1, 0.6)
        assert config.tempo_weight_range == (0.1, 0.4)
        assert config.energy_weight_range == (0.1, 0.4)
        assert config.difficulty_weight_range == (0.1, 0.4)
        assert config.grid_steps == 5
        assert config.max_combinations == 100
        assert config.validation_split == 0.2
        assert config.cross_validation_folds == 3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            audio_weight_range=(0.2, 0.8),
            grid_steps=10,
            max_combinations=50
        )
        
        assert config.audio_weight_range == (0.2, 0.8)
        assert config.grid_steps == 10
        assert config.max_combinations == 50


if __name__ == "__main__":
    pytest.main([__file__])