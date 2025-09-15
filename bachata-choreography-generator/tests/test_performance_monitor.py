"""
Tests for performance monitor.
"""

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.performance_monitor import (
    PerformanceMonitor, RecommendationDecision, UserFeedback,
    PerformanceMetrics, ABTestConfig
)


@pytest.fixture
def temp_monitor_dir():
    """Create temporary monitoring directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def performance_monitor(temp_monitor_dir):
    """Create performance monitor instance."""
    return PerformanceMonitor(temp_monitor_dir)


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_initialization(self, temp_monitor_dir):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(temp_monitor_dir)
        
        assert monitor.data_dir == Path(temp_monitor_dir)
        assert len(monitor.recent_decisions) == 0
        assert len(monitor.recent_feedback) == 0
        assert monitor.current_algorithm_version == "1.0"
    
    def test_log_recommendation_decision(self, performance_monitor):
        """Test logging recommendation decisions."""
        decision_id = performance_monitor.log_recommendation_decision(
            input_song="test_song",
            recommended_moves=["move1", "move2", "move3"],
            recommendation_scores=[0.8, 0.7, 0.6],
            weights_used={'audio': 0.4, 'tempo': 0.3, 'energy': 0.2, 'difficulty': 0.1},
            processing_time_ms=50.0,
            song_features={'tempo': 120, 'energy': 0.8},
            user_id="test_user"
        )
        
        assert decision_id is not None
        assert len(performance_monitor.recent_decisions) == 1
        
        decision = performance_monitor.recent_decisions[0]
        assert decision.input_song == "test_song"
        assert decision.recommended_moves == ["move1", "move2", "move3"]
        assert decision.recommendation_scores == [0.8, 0.7, 0.6]
        assert decision.user_id == "test_user"
    
    def test_log_user_feedback(self, performance_monitor):
        """Test logging user feedback."""
        # First log a decision
        decision_id = performance_monitor.log_recommendation_decision(
            input_song="test_song",
            recommended_moves=["move1"],
            recommendation_scores=[0.8],
            weights_used={'audio': 0.5, 'tempo': 0.5},
            processing_time_ms=50.0,
            song_features={'tempo': 120}
        )
        
        # Then log feedback
        feedback_id = performance_monitor.log_user_feedback(
            decision_id=decision_id,
            feedback_type="rating",
            feedback_value=4.5,
            user_id="test_user"
        )
        
        assert feedback_id is not None
        assert len(performance_monitor.recent_feedback) == 1
        
        feedback = performance_monitor.recent_feedback[0]
        assert feedback.decision_id == decision_id
        assert feedback.feedback_type == "rating"
        assert feedback.feedback_value == 4.5
        assert feedback.user_id == "test_user"
    
    def test_calculate_performance_metrics_empty(self, performance_monitor):
        """Test calculating metrics with no data."""
        metrics = performance_monitor.calculate_performance_metrics()
        
        assert metrics.total_recommendations == 0
        assert metrics.avg_processing_time_ms == 0.0
        assert metrics.avg_recommendation_score == 0.0
        assert metrics.user_satisfaction_score == 0.0
    
    def test_calculate_performance_metrics_with_data(self, performance_monitor):
        """Test calculating metrics with sample data."""
        # Log some decisions
        decision_ids = []
        for i in range(3):
            decision_id = performance_monitor.log_recommendation_decision(
                input_song=f"song_{i}",
                recommended_moves=[f"move_{i}"],
                recommendation_scores=[0.8 + i * 0.1],
                weights_used={'audio': 0.5, 'tempo': 0.5},
                processing_time_ms=50.0 + i * 10,
                song_features={'tempo': 120 + i}
            )
            decision_ids.append(decision_id)
        
        # Log some feedback
        performance_monitor.log_user_feedback(
            decision_id=decision_ids[0],
            feedback_type="like"
        )
        performance_monitor.log_user_feedback(
            decision_id=decision_ids[1],
            feedback_type="rating",
            feedback_value=4.0
        )
        
        # Calculate metrics
        metrics = performance_monitor.calculate_performance_metrics()
        
        assert metrics.total_recommendations == 3
        assert metrics.avg_processing_time_ms == 60.0  # (50 + 60 + 70) / 3
        assert metrics.avg_recommendation_score == 0.9  # (0.8 + 0.9 + 1.0) / 3
        assert metrics.user_satisfaction_score > 0.5  # Should be positive with likes and ratings
        assert len(metrics.recommendation_distribution) == 3
    
    def test_ab_test_setup(self, performance_monitor):
        """Test A/B test setup."""
        test_config = ABTestConfig(
            test_id="test_1",
            test_name="Algorithm Comparison",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            algorithm_versions=["v1.0", "v2.0"],
            traffic_split={"v1.0": 0.5, "v2.0": 0.5},
            success_metrics=["user_satisfaction"],
            minimum_sample_size=100
        )
        
        performance_monitor.setup_ab_test(test_config)
        
        assert "test_1" in performance_monitor.active_ab_tests
        assert performance_monitor.active_ab_tests["test_1"] == test_config
    
    def test_get_algorithm_version_for_user(self, performance_monitor):
        """Test algorithm version assignment for A/B testing."""
        # Set up A/B test
        test_config = ABTestConfig(
            test_id="test_1",
            test_name="Algorithm Comparison",
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now() + timedelta(days=7),
            algorithm_versions=["v1.0", "v2.0"],
            traffic_split={"v1.0": 0.5, "v2.0": 0.5},
            success_metrics=["user_satisfaction"],
            minimum_sample_size=100
        )
        
        performance_monitor.setup_ab_test(test_config)
        
        # Test version assignment
        version1 = performance_monitor.get_algorithm_version_for_user("user_1")
        version2 = performance_monitor.get_algorithm_version_for_user("user_2")
        
        assert version1 in ["v1.0", "v2.0"]
        assert version2 in ["v1.0", "v2.0"]
        
        # Same user should get same version
        version1_again = performance_monitor.get_algorithm_version_for_user("user_1")
        assert version1 == version1_again
    
    def test_analyze_ab_test_results(self, performance_monitor):
        """Test A/B test results analysis."""
        # Set up A/B test
        test_config = ABTestConfig(
            test_id="test_1",
            test_name="Algorithm Comparison",
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now() + timedelta(hours=1),
            algorithm_versions=["v1.0", "v2.0"],
            traffic_split={"v1.0": 0.5, "v2.0": 0.5},
            success_metrics=["user_satisfaction"],
            minimum_sample_size=10
        )
        
        performance_monitor.setup_ab_test(test_config)
        
        # Log decisions for different versions
        performance_monitor.current_algorithm_version = "v1.0"
        decision_id_v1 = performance_monitor.log_recommendation_decision(
            input_song="song_1",
            recommended_moves=["move_1"],
            recommendation_scores=[0.8],
            weights_used={'audio': 0.5, 'tempo': 0.5},
            processing_time_ms=50.0,
            song_features={'tempo': 120}
        )
        
        performance_monitor.current_algorithm_version = "v2.0"
        decision_id_v2 = performance_monitor.log_recommendation_decision(
            input_song="song_2",
            recommended_moves=["move_2"],
            recommendation_scores=[0.9],
            weights_used={'audio': 0.5, 'tempo': 0.5},
            processing_time_ms=40.0,
            song_features={'tempo': 125}
        )
        
        # Log feedback
        performance_monitor.log_user_feedback(decision_id_v1, "rating", 3.0)
        performance_monitor.log_user_feedback(decision_id_v2, "rating", 4.0)
        
        # Analyze results
        analysis = performance_monitor.analyze_ab_test_results("test_1")
        
        assert analysis['test_id'] == "test_1"
        assert 'versions' in analysis
        assert 'v1.0' in analysis['versions']
        assert 'v2.0' in analysis['versions']
        
        v1_results = analysis['versions']['v1.0']
        v2_results = analysis['versions']['v2.0']
        
        assert v1_results['sample_size'] == 1
        assert v2_results['sample_size'] == 1
        assert v1_results['avg_processing_time_ms'] == 50.0
        assert v2_results['avg_processing_time_ms'] == 40.0
    
    def test_generate_performance_report(self, performance_monitor):
        """Test performance report generation."""
        # Log some sample data
        for i in range(5):
            decision_id = performance_monitor.log_recommendation_decision(
                input_song=f"song_{i}",
                recommended_moves=[f"move_{i}"],
                recommendation_scores=[0.8],
                weights_used={'audio': 0.5, 'tempo': 0.5},
                processing_time_ms=50.0,
                song_features={'tempo': 120}
            )
            
            if i % 2 == 0:
                performance_monitor.log_user_feedback(decision_id, "like")
        
        # Generate report
        report = performance_monitor.generate_performance_report(days=1)
        
        assert 'report_period' in report
        assert 'overall_metrics' in report
        assert 'daily_metrics' in report
        assert 'top_recommended_moves' in report
        assert 'feedback_trends' in report
        
        assert report['overall_metrics']['total_recommendations'] == 5
        assert len(report['top_recommended_moves']) <= 10
        assert 'like' in report['feedback_trends']
    
    def test_persistence(self, temp_monitor_dir):
        """Test data persistence and loading."""
        # Create monitor and log some data
        monitor1 = PerformanceMonitor(temp_monitor_dir)
        
        decision_id = monitor1.log_recommendation_decision(
            input_song="test_song",
            recommended_moves=["move1"],
            recommendation_scores=[0.8],
            weights_used={'audio': 0.5, 'tempo': 0.5},
            processing_time_ms=50.0,
            song_features={'tempo': 120}
        )
        
        monitor1.log_user_feedback(decision_id, "like")
        
        # Force persistence
        monitor1._persist_decisions()
        monitor1._persist_feedback()
        
        # Create new monitor and check if data is loaded
        monitor2 = PerformanceMonitor(temp_monitor_dir)
        
        assert len(monitor2.recent_decisions) == 1
        assert len(monitor2.recent_feedback) == 1
        assert monitor2.recent_decisions[0].input_song == "test_song"
        assert monitor2.recent_feedback[0].feedback_type == "like"


class TestDataStructures:
    """Test data structure classes."""
    
    def test_recommendation_decision_creation(self):
        """Test RecommendationDecision creation."""
        decision = RecommendationDecision(
            decision_id="test_id",
            timestamp=datetime.now(),
            user_id="test_user",
            input_song="test_song",
            recommended_moves=["move1", "move2"],
            recommendation_scores=[0.8, 0.7],
            algorithm_version="1.0",
            weights_used={'audio': 0.5, 'tempo': 0.5},
            processing_time_ms=50.0,
            song_features={'tempo': 120}
        )
        
        assert decision.decision_id == "test_id"
        assert decision.user_id == "test_user"
        assert decision.input_song == "test_song"
        assert len(decision.recommended_moves) == 2
        assert len(decision.recommendation_scores) == 2
    
    def test_user_feedback_creation(self):
        """Test UserFeedback creation."""
        feedback = UserFeedback(
            feedback_id="feedback_id",
            decision_id="decision_id",
            timestamp=datetime.now(),
            user_id="test_user",
            feedback_type="rating",
            feedback_value=4.5,
            move_id="move1"
        )
        
        assert feedback.feedback_id == "feedback_id"
        assert feedback.decision_id == "decision_id"
        assert feedback.feedback_type == "rating"
        assert feedback.feedback_value == 4.5
        assert feedback.move_id == "move1"
    
    def test_ab_test_config_creation(self):
        """Test ABTestConfig creation."""
        config = ABTestConfig(
            test_id="test_1",
            test_name="Algorithm Test",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            algorithm_versions=["v1.0", "v2.0"],
            traffic_split={"v1.0": 0.5, "v2.0": 0.5},
            success_metrics=["satisfaction"],
            minimum_sample_size=100
        )
        
        assert config.test_id == "test_1"
        assert config.test_name == "Algorithm Test"
        assert len(config.algorithm_versions) == 2
        assert config.traffic_split["v1.0"] == 0.5
        assert config.minimum_sample_size == 100


if __name__ == "__main__":
    pytest.main([__file__])