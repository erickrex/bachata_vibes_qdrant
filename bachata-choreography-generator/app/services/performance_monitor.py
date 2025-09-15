"""
Model performance monitoring and continuous improvement system.
Tracks recommendation decisions, user feedback, and system performance metrics.
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RecommendationDecision:
    """Record of a recommendation decision made by the system."""
    decision_id: str
    timestamp: datetime
    user_id: Optional[str]
    input_song: str
    recommended_moves: List[str]
    recommendation_scores: List[float]
    algorithm_version: str
    weights_used: Dict[str, float]
    processing_time_ms: float
    
    # Context information
    song_features: Dict[str, Any]
    user_preferences: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


@dataclass
class UserFeedback:
    """User feedback on recommendations."""
    feedback_id: str
    decision_id: str
    timestamp: datetime
    user_id: Optional[str]
    feedback_type: str  # 'rating', 'like', 'dislike', 'skip', 'use'
    feedback_value: Optional[float]  # Rating value if applicable
    move_id: Optional[str]  # Specific move feedback
    comments: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """System performance metrics for a time period."""
    period_start: datetime
    period_end: datetime
    total_recommendations: int
    avg_processing_time_ms: float
    avg_recommendation_score: float
    user_satisfaction_score: float
    algorithm_version: str
    
    # Detailed metrics
    recommendation_distribution: Dict[str, int]
    score_distribution: Dict[str, float]
    feedback_summary: Dict[str, int]
    error_count: int
    uptime_percentage: float


@dataclass
class ABTestConfig:
    """Configuration for A/B testing different algorithm versions."""
    test_id: str
    test_name: str
    start_date: datetime
    end_date: datetime
    algorithm_versions: List[str]
    traffic_split: Dict[str, float]  # version -> percentage
    success_metrics: List[str]
    minimum_sample_size: int


class PerformanceMonitor:
    """
    Performance monitoring system for the recommendation engine.
    Tracks decisions, feedback, and performance metrics.
    """
    
    def __init__(self, data_dir: str = "data/monitoring"):
        """Initialize the performance monitor."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffers for recent data
        self.recent_decisions = deque(maxlen=1000)
        self.recent_feedback = deque(maxlen=1000)
        self.recent_metrics = deque(maxlen=100)
        
        # Performance tracking
        self.current_algorithm_version = "1.0"
        self.active_ab_tests = {}
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"PerformanceMonitor initialized with data_dir: {data_dir}")
    
    def log_recommendation_decision(self, 
                                  input_song: str,
                                  recommended_moves: List[str],
                                  recommendation_scores: List[float],
                                  weights_used: Dict[str, float],
                                  processing_time_ms: float,
                                  song_features: Dict[str, Any],
                                  user_id: Optional[str] = None,
                                  user_preferences: Optional[Dict[str, Any]] = None,
                                  session_id: Optional[str] = None) -> str:
        """
        Log a recommendation decision.
        
        Returns:
            decision_id: Unique identifier for this decision
        """
        decision_id = str(uuid.uuid4())
        
        decision = RecommendationDecision(
            decision_id=decision_id,
            timestamp=datetime.now(),
            user_id=user_id,
            input_song=input_song,
            recommended_moves=recommended_moves,
            recommendation_scores=recommendation_scores,
            algorithm_version=self.current_algorithm_version,
            weights_used=weights_used,
            processing_time_ms=processing_time_ms,
            song_features=song_features,
            user_preferences=user_preferences,
            session_id=session_id
        )
        
        # Add to recent decisions buffer
        self.recent_decisions.append(decision)
        
        # Persist to disk periodically
        if len(self.recent_decisions) % 10 == 0:
            self._persist_decisions()
        
        logger.debug(f"Logged recommendation decision {decision_id}")
        return decision_id
    
    def log_user_feedback(self,
                         decision_id: str,
                         feedback_type: str,
                         feedback_value: Optional[float] = None,
                         move_id: Optional[str] = None,
                         comments: Optional[str] = None,
                         user_id: Optional[str] = None) -> str:
        """
        Log user feedback on a recommendation.
        
        Returns:
            feedback_id: Unique identifier for this feedback
        """
        feedback_id = str(uuid.uuid4())
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            decision_id=decision_id,
            timestamp=datetime.now(),
            user_id=user_id,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            move_id=move_id,
            comments=comments
        )
        
        # Add to recent feedback buffer
        self.recent_feedback.append(feedback)
        
        # Persist to disk periodically
        if len(self.recent_feedback) % 10 == 0:
            self._persist_feedback()
        
        logger.debug(f"Logged user feedback {feedback_id} for decision {decision_id}")
        return feedback_id
    
    def calculate_performance_metrics(self, 
                                    start_time: Optional[datetime] = None,
                                    end_time: Optional[datetime] = None) -> PerformanceMetrics:
        """
        Calculate performance metrics for a time period.
        
        Args:
            start_time: Start of time period (default: 24 hours ago)
            end_time: End of time period (default: now)
            
        Returns:
            PerformanceMetrics for the specified period
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # Filter decisions and feedback for the time period
        period_decisions = [
            d for d in self.recent_decisions 
            if start_time <= d.timestamp <= end_time
        ]
        
        period_feedback = [
            f for f in self.recent_feedback 
            if start_time <= f.timestamp <= end_time
        ]
        
        if not period_decisions:
            return PerformanceMetrics(
                period_start=start_time,
                period_end=end_time,
                total_recommendations=0,
                avg_processing_time_ms=0.0,
                avg_recommendation_score=0.0,
                user_satisfaction_score=0.0,
                algorithm_version=self.current_algorithm_version,
                recommendation_distribution={},
                score_distribution={},
                feedback_summary={},
                error_count=0,
                uptime_percentage=100.0
            )
        
        # Calculate basic metrics
        total_recommendations = len(period_decisions)
        avg_processing_time = np.mean([d.processing_time_ms for d in period_decisions])
        
        # Calculate average recommendation scores
        all_scores = []
        for decision in period_decisions:
            all_scores.extend(decision.recommendation_scores)
        avg_recommendation_score = np.mean(all_scores) if all_scores else 0.0
        
        # Calculate recommendation distribution
        recommendation_distribution = defaultdict(int)
        for decision in period_decisions:
            for move in decision.recommended_moves:
                recommendation_distribution[move] += 1
        
        # Calculate score distribution by quartiles
        if all_scores:
            score_distribution = {
                'min': float(np.min(all_scores)),
                'q25': float(np.percentile(all_scores, 25)),
                'median': float(np.median(all_scores)),
                'q75': float(np.percentile(all_scores, 75)),
                'max': float(np.max(all_scores)),
                'mean': float(np.mean(all_scores)),
                'std': float(np.std(all_scores))
            }
        else:
            score_distribution = {}
        
        # Calculate feedback summary
        feedback_summary = defaultdict(int)
        satisfaction_scores = []
        
        for feedback in period_feedback:
            feedback_summary[feedback.feedback_type] += 1
            
            # Convert feedback to satisfaction score
            if feedback.feedback_type == 'rating' and feedback.feedback_value is not None:
                satisfaction_scores.append(feedback.feedback_value)
            elif feedback.feedback_type == 'like':
                satisfaction_scores.append(1.0)
            elif feedback.feedback_type == 'dislike':
                satisfaction_scores.append(0.0)
            elif feedback.feedback_type == 'use':
                satisfaction_scores.append(1.0)
            elif feedback.feedback_type == 'skip':
                satisfaction_scores.append(0.2)
        
        user_satisfaction_score = np.mean(satisfaction_scores) if satisfaction_scores else 0.5
        
        metrics = PerformanceMetrics(
            period_start=start_time,
            period_end=end_time,
            total_recommendations=total_recommendations,
            avg_processing_time_ms=avg_processing_time,
            avg_recommendation_score=avg_recommendation_score,
            user_satisfaction_score=user_satisfaction_score,
            algorithm_version=self.current_algorithm_version,
            recommendation_distribution=dict(recommendation_distribution),
            score_distribution=score_distribution,
            feedback_summary=dict(feedback_summary),
            error_count=0,  # Would be tracked separately
            uptime_percentage=100.0  # Would be tracked separately
        )
        
        # Add to recent metrics
        self.recent_metrics.append(metrics)
        
        logger.info(f"Calculated performance metrics for period {start_time} to {end_time}")
        return metrics
    
    def setup_ab_test(self, test_config: ABTestConfig):
        """Set up an A/B test configuration."""
        self.active_ab_tests[test_config.test_id] = test_config
        logger.info(f"Set up A/B test: {test_config.test_name}")
    
    def get_algorithm_version_for_user(self, user_id: Optional[str] = None) -> str:
        """
        Get the algorithm version to use for a user (for A/B testing).
        
        Args:
            user_id: User identifier
            
        Returns:
            Algorithm version to use
        """
        # Simple A/B testing logic
        for test_id, test_config in self.active_ab_tests.items():
            if test_config.start_date <= datetime.now() <= test_config.end_date:
                # Use hash of user_id to consistently assign users to versions
                if user_id:
                    user_hash = hash(user_id) % 100
                    cumulative_percentage = 0
                    for version, percentage in test_config.traffic_split.items():
                        cumulative_percentage += percentage * 100
                        if user_hash < cumulative_percentage:
                            return version
                
                # Default to first version if no user_id
                return list(test_config.algorithm_versions)[0]
        
        return self.current_algorithm_version
    
    def analyze_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Analyze results of an A/B test.
        
        Args:
            test_id: A/B test identifier
            
        Returns:
            Analysis results
        """
        if test_id not in self.active_ab_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        test_config = self.active_ab_tests[test_id]
        
        # Filter decisions for the test period
        test_decisions = [
            d for d in self.recent_decisions
            if (test_config.start_date <= d.timestamp <= test_config.end_date and
                d.algorithm_version in test_config.algorithm_versions)
        ]
        
        # Group by algorithm version
        version_results = defaultdict(list)
        for decision in test_decisions:
            version_results[decision.algorithm_version].append(decision)
        
        # Calculate metrics for each version
        analysis = {
            'test_id': test_id,
            'test_name': test_config.test_name,
            'test_period': {
                'start': test_config.start_date.isoformat(),
                'end': test_config.end_date.isoformat()
            },
            'versions': {}
        }
        
        for version, decisions in version_results.items():
            if not decisions:
                continue
            
            # Calculate metrics for this version
            processing_times = [d.processing_time_ms for d in decisions]
            all_scores = []
            for d in decisions:
                all_scores.extend(d.recommendation_scores)
            
            # Get feedback for these decisions
            decision_ids = {d.decision_id for d in decisions}
            version_feedback = [
                f for f in self.recent_feedback
                if f.decision_id in decision_ids
            ]
            
            satisfaction_scores = []
            for feedback in version_feedback:
                if feedback.feedback_type == 'rating' and feedback.feedback_value is not None:
                    satisfaction_scores.append(feedback.feedback_value)
                elif feedback.feedback_type == 'like':
                    satisfaction_scores.append(1.0)
                elif feedback.feedback_type == 'dislike':
                    satisfaction_scores.append(0.0)
            
            analysis['versions'][version] = {
                'sample_size': len(decisions),
                'avg_processing_time_ms': np.mean(processing_times),
                'avg_recommendation_score': np.mean(all_scores) if all_scores else 0.0,
                'user_satisfaction_score': np.mean(satisfaction_scores) if satisfaction_scores else 0.5,
                'feedback_count': len(version_feedback)
            }
        
        logger.info(f"Analyzed A/B test results for {test_id}")
        return analysis
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            Performance report
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_performance_metrics(start_time, end_time)
        
        # Calculate daily metrics
        daily_metrics = []
        for i in range(days):
            day_start = start_time + timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            day_metrics = self.calculate_performance_metrics(day_start, day_end)
            daily_metrics.append({
                'date': day_start.date().isoformat(),
                'total_recommendations': day_metrics.total_recommendations,
                'avg_processing_time_ms': day_metrics.avg_processing_time_ms,
                'user_satisfaction_score': day_metrics.user_satisfaction_score
            })
        
        # Top recommended moves
        move_counts = defaultdict(int)
        for decision in self.recent_decisions:
            if start_time <= decision.timestamp <= end_time:
                for move in decision.recommended_moves:
                    move_counts[move] += 1
        
        top_moves = sorted(move_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Recent feedback trends
        feedback_trends = defaultdict(int)
        for feedback in self.recent_feedback:
            if start_time <= feedback.timestamp <= end_time:
                feedback_trends[feedback.feedback_type] += 1
        
        report = {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'days': days
            },
            'overall_metrics': asdict(overall_metrics),
            'daily_metrics': daily_metrics,
            'top_recommended_moves': top_moves,
            'feedback_trends': dict(feedback_trends),
            'active_ab_tests': list(self.active_ab_tests.keys()),
            'algorithm_version': self.current_algorithm_version,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Generated performance report for {days} days")
        return report
    
    def _load_existing_data(self):
        """Load existing monitoring data from disk."""
        # Load recent decisions
        decisions_file = self.data_dir / "recent_decisions.json"
        if decisions_file.exists():
            try:
                with open(decisions_file, 'r') as f:
                    decisions_data = json.load(f)
                
                for decision_data in decisions_data:
                    decision_data['timestamp'] = datetime.fromisoformat(decision_data['timestamp'])
                    decision = RecommendationDecision(**decision_data)
                    self.recent_decisions.append(decision)
                
                logger.info(f"Loaded {len(self.recent_decisions)} recent decisions")
            except Exception as e:
                logger.warning(f"Failed to load recent decisions: {e}")
        
        # Load recent feedback
        feedback_file = self.data_dir / "recent_feedback.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                
                for feedback_item in feedback_data:
                    feedback_item['timestamp'] = datetime.fromisoformat(feedback_item['timestamp'])
                    feedback = UserFeedback(**feedback_item)
                    self.recent_feedback.append(feedback)
                
                logger.info(f"Loaded {len(self.recent_feedback)} recent feedback items")
            except Exception as e:
                logger.warning(f"Failed to load recent feedback: {e}")
    
    def _persist_decisions(self):
        """Persist recent decisions to disk."""
        decisions_file = self.data_dir / "recent_decisions.json"
        
        try:
            decisions_data = []
            for decision in self.recent_decisions:
                decision_dict = asdict(decision)
                decision_dict['timestamp'] = decision.timestamp.isoformat()
                decisions_data.append(decision_dict)
            
            with open(decisions_file, 'w') as f:
                json.dump(decisions_data, f, indent=2, default=str)
            
            logger.debug(f"Persisted {len(decisions_data)} decisions to disk")
        except Exception as e:
            logger.error(f"Failed to persist decisions: {e}")
    
    def _persist_feedback(self):
        """Persist recent feedback to disk."""
        feedback_file = self.data_dir / "recent_feedback.json"
        
        try:
            feedback_data = []
            for feedback in self.recent_feedback:
                feedback_dict = asdict(feedback)
                feedback_dict['timestamp'] = feedback.timestamp.isoformat()
                feedback_data.append(feedback_dict)
            
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2, default=str)
            
            logger.debug(f"Persisted {len(feedback_data)} feedback items to disk")
        except Exception as e:
            logger.error(f"Failed to persist feedback: {e}")
    
    def save_performance_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save performance report to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report_file = self.data_dir / filename
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved performance report to {report_file}")


def main():
    """Main function for performance monitoring demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance monitoring demo")
    parser.add_argument("--data_dir", default="data/monitoring", help="Monitoring data directory")
    parser.add_argument("--days", type=int, default=7, help="Days to include in report")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = PerformanceMonitor(args.data_dir)
    
    # Generate some sample data
    print("Generating sample monitoring data...")
    
    for i in range(50):
        # Log sample recommendation
        decision_id = monitor.log_recommendation_decision(
            input_song=f"song_{i % 10}",
            recommended_moves=[f"move_{j}" for j in range(i % 3 + 1, i % 3 + 4)],
            recommendation_scores=[0.8 + np.random.normal(0, 0.1) for _ in range(3)],
            weights_used={'audio': 0.4, 'tempo': 0.3, 'energy': 0.2, 'difficulty': 0.1},
            processing_time_ms=50 + np.random.normal(0, 10),
            song_features={'tempo': 120 + i, 'energy': 0.5 + i * 0.01},
            user_id=f"user_{i % 5}"
        )
        
        # Log sample feedback (for some decisions)
        if i % 3 == 0:
            monitor.log_user_feedback(
                decision_id=decision_id,
                feedback_type='rating',
                feedback_value=3.5 + np.random.normal(0, 1),
                user_id=f"user_{i % 5}"
            )
    
    # Generate performance report
    print(f"Generating performance report for {args.days} days...")
    report = monitor.generate_performance_report(args.days)
    
    # Save report
    monitor.save_performance_report(report)
    
    # Print summary
    print(f"✅ Performance monitoring demo complete!")
    print(f"Total recommendations: {report['overall_metrics']['total_recommendations']}")
    print(f"Average processing time: {report['overall_metrics']['avg_processing_time_ms']:.1f}ms")
    print(f"User satisfaction score: {report['overall_metrics']['user_satisfaction_score']:.2f}")
    print(f"Report saved to: {monitor.data_dir}")


if __name__ == "__main__":
    main()