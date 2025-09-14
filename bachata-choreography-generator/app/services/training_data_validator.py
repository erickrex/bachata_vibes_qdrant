"""
Training data validation and quality assurance service.
Provides comprehensive validation for video annotations, pose detection confidence,
annotation consistency, and training data statistics.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, Counter
# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
from datetime import datetime

from .annotation_validator import AnnotationValidator
from .move_analyzer import MoveAnalyzer
from ..models.annotation_schema import (
    AnnotationCollection, 
    MoveAnnotation, 
    QualityStandards,
    DifficultyLevel,
    EnergyLevel,
    MoveCategory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PoseQualityMetrics:
    """Metrics for pose detection quality assessment."""
    clip_id: str
    video_path: str
    pose_detection_rate: float  # Percentage of frames with successful pose detection
    average_confidence: float  # Average pose detection confidence
    confidence_std: float  # Standard deviation of confidence scores
    low_confidence_frames: int  # Number of frames with confidence < 0.5
    quality_score: float  # Overall quality score (0-1)
    issues: List[str]  # List of quality issues identified


@dataclass
class AnnotationConsistencyCheck:
    """Results of annotation consistency validation."""
    clip_id: str
    move_label: str
    predicted_category: Optional[str]  # Category predicted from pose analysis
    predicted_difficulty: Optional[str]  # Difficulty predicted from movement complexity
    predicted_energy: Optional[str]  # Energy predicted from movement intensity
    consistency_score: float  # Overall consistency score (0-1)
    inconsistencies: List[str]  # List of inconsistencies found


@dataclass
class TrainingDataStatistics:
    """Comprehensive statistics for training data."""
    total_clips: int
    total_duration: float  # Total duration in seconds
    
    # Distribution statistics
    category_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    energy_distribution: Dict[str, int]
    tempo_distribution: Dict[str, int]  # Binned tempo distribution
    
    # Quality statistics
    average_pose_quality: float
    clips_with_quality_issues: int
    clips_with_consistency_issues: int
    
    # Balance analysis
    category_balance_score: float  # How balanced the categories are (0-1)
    difficulty_balance_score: float
    energy_balance_score: float
    
    # Recommendations
    recommendations: List[str]


class TrainingDataValidator:
    """
    Comprehensive training data validation and quality assurance service.
    Combines annotation validation, pose quality assessment, and consistency checking.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.annotation_validator = AnnotationValidator(data_dir)
        self.move_analyzer = MoveAnalyzer(target_fps=15)  # Lower FPS for faster validation
        
        # Quality thresholds
        self.min_pose_detection_rate = 0.8  # 80% of frames should have pose detection
        self.min_average_confidence = 0.6   # Average confidence should be > 0.6
        self.max_low_confidence_frames = 0.2  # Max 20% of frames with low confidence
        
        logger.info("TrainingDataValidator initialized")
    
    def validate_pose_quality(self, annotation_file: str = "bachata_annotations.json") -> List[PoseQualityMetrics]:
        """
        Validate pose detection quality for all clips in the annotation file.
        
        Returns:
            List of PoseQualityMetrics for each clip
        """
        logger.info("Starting pose quality validation...")
        
        # Load annotations
        collection = self.annotation_validator.load_annotations(annotation_file)
        
        pose_quality_results = []
        
        for i, clip in enumerate(collection.clips):
            logger.info(f"Analyzing pose quality for clip {i+1}/{len(collection.clips)}: {clip.clip_id}")
            
            try:
                # Analyze the move clip
                analysis_result = self.move_analyzer.analyze_move_clip(
                    str(self.data_dir / clip.video_path)
                )
                
                # Calculate quality metrics
                pose_detection_rate = analysis_result.pose_detection_rate
                
                # Calculate confidence statistics
                confidences = [pf.confidence for pf in analysis_result.pose_features]
                average_confidence = np.mean(confidences) if confidences else 0.0
                confidence_std = np.std(confidences) if len(confidences) > 1 else 0.0
                low_confidence_frames = sum(1 for c in confidences if c < 0.5)
                
                # Calculate overall quality score
                quality_score = self._calculate_pose_quality_score(
                    pose_detection_rate, average_confidence, low_confidence_frames, 
                    len(analysis_result.pose_features)
                )
                
                # Identify issues
                issues = []
                if pose_detection_rate < self.min_pose_detection_rate:
                    issues.append(f"Low pose detection rate: {pose_detection_rate:.2f}")
                if average_confidence < self.min_average_confidence:
                    issues.append(f"Low average confidence: {average_confidence:.2f}")
                if low_confidence_frames / len(confidences) > self.max_low_confidence_frames:
                    issues.append(f"Too many low confidence frames: {low_confidence_frames}")
                
                pose_quality_results.append(PoseQualityMetrics(
                    clip_id=clip.clip_id,
                    video_path=clip.video_path,
                    pose_detection_rate=pose_detection_rate,
                    average_confidence=average_confidence,
                    confidence_std=confidence_std,
                    low_confidence_frames=low_confidence_frames,
                    quality_score=quality_score,
                    issues=issues
                ))
                
            except Exception as e:
                logger.error(f"Error analyzing clip {clip.clip_id}: {str(e)}")
                pose_quality_results.append(PoseQualityMetrics(
                    clip_id=clip.clip_id,
                    video_path=clip.video_path,
                    pose_detection_rate=0.0,
                    average_confidence=0.0,
                    confidence_std=0.0,
                    low_confidence_frames=0,
                    quality_score=0.0,
                    issues=[f"Analysis failed: {str(e)}"]
                ))
        
        logger.info(f"Pose quality validation complete. Analyzed {len(pose_quality_results)} clips.")
        return pose_quality_results
    
    def validate_annotation_consistency(self, annotation_file: str = "bachata_annotations.json") -> List[AnnotationConsistencyCheck]:
        """
        Validate annotation consistency by comparing annotations with extracted features.
        
        Returns:
            List of AnnotationConsistencyCheck for each clip
        """
        logger.info("Starting annotation consistency validation...")
        
        # Load annotations
        collection = self.annotation_validator.load_annotations(annotation_file)
        
        consistency_results = []
        
        for i, clip in enumerate(collection.clips):
            logger.info(f"Checking consistency for clip {i+1}/{len(collection.clips)}: {clip.clip_id}")
            
            try:
                # Analyze the move clip
                analysis_result = self.move_analyzer.analyze_move_clip(
                    str(self.data_dir / clip.video_path)
                )
                
                # Predict category from movement analysis
                predicted_category = self._predict_category_from_movement(analysis_result, clip.move_label)
                
                # Predict difficulty from movement complexity
                predicted_difficulty = self._predict_difficulty_from_complexity(
                    analysis_result.movement_complexity_score,
                    analysis_result.difficulty_score
                )
                
                # Predict energy from movement intensity
                predicted_energy = self._predict_energy_from_intensity(analysis_result.movement_dynamics)
                
                # Calculate consistency score and identify inconsistencies
                consistency_score, inconsistencies = self._calculate_consistency_score(
                    clip, predicted_category, predicted_difficulty, predicted_energy
                )
                
                consistency_results.append(AnnotationConsistencyCheck(
                    clip_id=clip.clip_id,
                    move_label=clip.move_label,
                    predicted_category=predicted_category,
                    predicted_difficulty=predicted_difficulty,
                    predicted_energy=predicted_energy,
                    consistency_score=consistency_score,
                    inconsistencies=inconsistencies
                ))
                
            except Exception as e:
                logger.error(f"Error checking consistency for clip {clip.clip_id}: {str(e)}")
                consistency_results.append(AnnotationConsistencyCheck(
                    clip_id=clip.clip_id,
                    move_label=clip.move_label,
                    predicted_category=None,
                    predicted_difficulty=None,
                    predicted_energy=None,
                    consistency_score=0.0,
                    inconsistencies=[f"Analysis failed: {str(e)}"]
                ))
        
        logger.info(f"Annotation consistency validation complete. Checked {len(consistency_results)} clips.")
        return consistency_results
    
    def generate_training_statistics(self, annotation_file: str = "bachata_annotations.json") -> TrainingDataStatistics:
        """
        Generate comprehensive training data statistics.
        
        Returns:
            TrainingDataStatistics with detailed analysis
        """
        logger.info("Generating training data statistics...")
        
        # Load annotations
        collection = self.annotation_validator.load_annotations(annotation_file)
        
        # Basic statistics
        total_clips = len(collection.clips)
        total_duration = 0.0
        
        # Distribution counters
        category_counts = Counter()
        difficulty_counts = Counter()
        energy_counts = Counter()
        tempo_bins = defaultdict(int)
        
        # Quality metrics (will be populated if pose quality validation is run)
        pose_quality_scores = []
        clips_with_quality_issues = 0
        clips_with_consistency_issues = 0
        
        for clip in collection.clips:
            # Update duration (estimate from video if not available)
            if hasattr(clip, 'duration_seconds') and clip.duration_seconds:
                total_duration += clip.duration_seconds
            else:
                # Estimate duration from video file
                try:
                    video_validation = self.annotation_validator.validate_video_file(clip.video_path)
                    total_duration += video_validation.get('duration', 10.0)  # Default 10s
                except:
                    total_duration += 10.0  # Default estimate
            
            # Update distributions
            if clip.category:
                category_counts[clip.category.value] += 1
            else:
                # Derive from move_label
                category = clip._derive_category_from_label(clip.move_label)
                if category:
                    category_counts[category.value] += 1
                else:
                    category_counts['unknown'] += 1
            
            difficulty_counts[clip.difficulty.value] += 1
            energy_counts[clip.energy_level.value] += 1
            
            # Bin tempo (10 BPM bins)
            tempo_bin = (clip.estimated_tempo // 10) * 10
            tempo_bins[f"{tempo_bin}-{tempo_bin+9}"] += 1
        
        # Calculate balance scores
        category_balance_score = self._calculate_balance_score(list(category_counts.values()))
        difficulty_balance_score = self._calculate_balance_score(list(difficulty_counts.values()))
        energy_balance_score = self._calculate_balance_score(list(energy_counts.values()))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            category_counts, difficulty_counts, energy_counts, total_clips
        )
        
        return TrainingDataStatistics(
            total_clips=total_clips,
            total_duration=total_duration,
            category_distribution=dict(category_counts),
            difficulty_distribution=dict(difficulty_counts),
            energy_distribution=dict(energy_counts),
            tempo_distribution=dict(tempo_bins),
            average_pose_quality=np.mean(pose_quality_scores) if pose_quality_scores else 0.0,
            clips_with_quality_issues=clips_with_quality_issues,
            clips_with_consistency_issues=clips_with_consistency_issues,
            category_balance_score=category_balance_score,
            difficulty_balance_score=difficulty_balance_score,
            energy_balance_score=energy_balance_score,
            recommendations=recommendations
        )
    
    def create_training_dashboard(self, 
                                annotation_file: str = "bachata_annotations.json",
                                output_dir: str = "data/validation_reports") -> str:
        """
        Create a comprehensive training data statistics dashboard with visualizations.
        
        Returns:
            Path to the generated dashboard HTML file
        """
        logger.info("Creating training data dashboard...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate statistics
        stats = self.generate_training_statistics(annotation_file)
        
        # Create visualizations (if matplotlib available)
        if VISUALIZATION_AVAILABLE:
            self._create_distribution_plots(stats, output_path)
        
        # Generate HTML dashboard
        dashboard_path = output_path / "training_dashboard.html"
        self._generate_dashboard_html(stats, dashboard_path)
        
        logger.info(f"Training dashboard created: {dashboard_path}")
        return str(dashboard_path)
    
    def run_comprehensive_validation(self, 
                                   annotation_file: str = "bachata_annotations.json",
                                   output_dir: str = "data/validation_reports") -> Dict[str, Any]:
        """
        Run comprehensive validation including all quality checks and generate reports.
        
        Returns:
            Dictionary with all validation results
        """
        logger.info("Starting comprehensive training data validation...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run all validations
        basic_validation = self.annotation_validator.validate_collection(annotation_file)
        pose_quality_results = self.validate_pose_quality(annotation_file)
        consistency_results = self.validate_annotation_consistency(annotation_file)
        statistics = self.generate_training_statistics(annotation_file)
        
        # Update statistics with quality results
        statistics.average_pose_quality = np.mean([pq.quality_score for pq in pose_quality_results])
        statistics.clips_with_quality_issues = sum(1 for pq in pose_quality_results if pq.issues)
        statistics.clips_with_consistency_issues = sum(1 for cr in consistency_results if cr.inconsistencies)
        
        # Generate comprehensive report
        report_path = output_path / "comprehensive_validation_report.md"
        self._generate_comprehensive_report(
            basic_validation, pose_quality_results, consistency_results, statistics, report_path
        )
        
        # Create dashboard
        dashboard_path = self.create_training_dashboard(annotation_file, output_dir)
        
        # Save detailed results as JSON
        results_path = output_path / "validation_results.json"
        detailed_results = {
            "timestamp": datetime.now().isoformat(),
            "basic_validation": basic_validation,
            "pose_quality_results": [asdict(pq) for pq in pose_quality_results],
            "consistency_results": [asdict(cr) for cr in consistency_results],
            "statistics": asdict(statistics)
        }
        
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        logger.info("Comprehensive validation complete!")
        
        return {
            "basic_validation": basic_validation,
            "pose_quality_results": pose_quality_results,
            "consistency_results": consistency_results,
            "statistics": statistics,
            "report_path": str(report_path),
            "dashboard_path": dashboard_path,
            "results_path": str(results_path)
        }
    
    # Helper methods
    
    def _calculate_pose_quality_score(self, detection_rate: float, avg_confidence: float, 
                                    low_conf_frames: int, total_frames: int) -> float:
        """Calculate overall pose quality score."""
        # Weight factors
        detection_weight = 0.4
        confidence_weight = 0.4
        consistency_weight = 0.2
        
        # Normalize metrics
        detection_score = min(1.0, detection_rate / self.min_pose_detection_rate)
        confidence_score = min(1.0, avg_confidence / self.min_average_confidence)
        consistency_score = max(0.0, 1.0 - (low_conf_frames / total_frames) / self.max_low_confidence_frames)
        
        return (detection_weight * detection_score + 
                confidence_weight * confidence_score + 
                consistency_weight * consistency_score)
    
    def _predict_category_from_movement(self, analysis_result, move_label: str) -> Optional[str]:
        """Predict move category from movement analysis."""
        # Simple heuristic-based prediction
        movement_dynamics = analysis_result.movement_dynamics
        
        # Check for specific movement patterns
        if movement_dynamics.dominant_movement_direction == "horizontal":
            if "cross" in move_label.lower():
                return "cross_body_lead"
            elif "forward" in move_label.lower() or "backward" in move_label.lower():
                return "forward_backward"
        
        if movement_dynamics.complexity_score > 0.7:
            if "combination" in move_label.lower():
                return "combination"
            elif "body_roll" in move_label.lower():
                return "body_roll"
        
        if "turn" in move_label.lower():
            if "right" in move_label.lower():
                return "lady_right_turn"
            elif "left" in move_label.lower():
                return "lady_left_turn"
        
        if "basic" in move_label.lower():
            return "basic_step"
        
        return None
    
    def _predict_difficulty_from_complexity(self, complexity_score: float, difficulty_score: float) -> str:
        """Predict difficulty level from movement complexity."""
        combined_score = (complexity_score + difficulty_score) / 2
        
        if combined_score < 0.3:
            return "beginner"
        elif combined_score < 0.7:
            return "intermediate"
        else:
            return "advanced"
    
    def _predict_energy_from_intensity(self, movement_dynamics) -> str:
        """Predict energy level from movement intensity."""
        # Use multiple indicators
        velocity_avg = np.mean(movement_dynamics.velocity_profile) if len(movement_dynamics.velocity_profile) > 0 else 0
        intensity_avg = np.mean(movement_dynamics.movement_intensity_profile) if len(movement_dynamics.movement_intensity_profile) > 0 else 0
        
        combined_intensity = (velocity_avg * 0.6 + intensity_avg * 0.4)
        
        if combined_intensity < 0.02:
            return "low"
        elif combined_intensity < 0.05:
            return "medium"
        else:
            return "high"
    
    def _calculate_consistency_score(self, clip: MoveAnnotation, predicted_category: Optional[str],
                                   predicted_difficulty: str, predicted_energy: str) -> Tuple[float, List[str]]:
        """Calculate consistency score and identify inconsistencies."""
        inconsistencies = []
        score_components = []
        
        # Category consistency
        actual_category = clip.category.value if clip.category else None
        if predicted_category and actual_category:
            if predicted_category == actual_category:
                score_components.append(1.0)
            else:
                score_components.append(0.0)
                inconsistencies.append(f"Category mismatch: annotated '{actual_category}', predicted '{predicted_category}'")
        
        # Difficulty consistency
        if predicted_difficulty == clip.difficulty.value:
            score_components.append(1.0)
        else:
            score_components.append(0.0)
            inconsistencies.append(f"Difficulty mismatch: annotated '{clip.difficulty.value}', predicted '{predicted_difficulty}'")
        
        # Energy consistency
        if predicted_energy == clip.energy_level.value:
            score_components.append(1.0)
        else:
            score_components.append(0.0)
            inconsistencies.append(f"Energy mismatch: annotated '{clip.energy_level.value}', predicted '{predicted_energy}'")
        
        # Calculate overall score
        consistency_score = np.mean(score_components) if score_components else 0.0
        
        return consistency_score, inconsistencies
    
    def _calculate_balance_score(self, counts: List[int]) -> float:
        """Calculate balance score for a distribution (higher = more balanced)."""
        if not counts or len(counts) <= 1:
            return 1.0
        
        total = sum(counts)
        if total == 0:
            return 1.0
        
        # Calculate entropy-based balance score
        proportions = [c / total for c in counts]
        entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
        max_entropy = np.log2(len(counts))
        
        return entropy / max_entropy if max_entropy > 0 else 1.0
    
    def _generate_recommendations(self, category_counts: Counter, difficulty_counts: Counter,
                                energy_counts: Counter, total_clips: int) -> List[str]:
        """Generate recommendations for improving training data."""
        recommendations = []
        
        # Category balance recommendations
        if len(category_counts) > 0:
            min_count = min(category_counts.values())
            max_count = max(category_counts.values())
            if max_count > 3 * min_count:
                underrepresented = [cat for cat, count in category_counts.items() if count == min_count]
                recommendations.append(f"Consider adding more clips for underrepresented categories: {', '.join(underrepresented)}")
        
        # Difficulty balance recommendations
        beginner_ratio = difficulty_counts.get('beginner', 0) / total_clips
        if beginner_ratio < 0.2:
            recommendations.append("Consider adding more beginner-level clips for better difficulty balance")
        elif beginner_ratio > 0.6:
            recommendations.append("Consider adding more intermediate/advanced clips for better difficulty balance")
        
        # Energy balance recommendations
        energy_values = list(energy_counts.values())
        if len(energy_values) > 1 and max(energy_values) > 2 * min(energy_values):
            recommendations.append("Consider balancing energy levels across clips")
        
        # Quality recommendations
        if total_clips < 50:
            recommendations.append("Consider expanding the dataset to at least 50 clips for better model training")
        
        return recommendations
    
    def _create_distribution_plots(self, stats: TrainingDataStatistics, output_path: Path):
        """Create distribution plots for the dashboard."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available. Skipping plot generation.")
            return
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Category distribution
        categories = list(stats.category_distribution.keys())
        category_counts = list(stats.category_distribution.values())
        axes[0, 0].bar(categories, category_counts, color='skyblue')
        axes[0, 0].set_title('Move Category Distribution')
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Number of Clips')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Difficulty distribution
        difficulties = list(stats.difficulty_distribution.keys())
        difficulty_counts = list(stats.difficulty_distribution.values())
        axes[0, 1].pie(difficulty_counts, labels=difficulties, autopct='%1.1f%%', colors=['lightgreen', 'orange', 'lightcoral'])
        axes[0, 1].set_title('Difficulty Level Distribution')
        
        # Energy distribution
        energies = list(stats.energy_distribution.keys())
        energy_counts = list(stats.energy_distribution.values())
        axes[1, 0].bar(energies, energy_counts, color='lightpink')
        axes[1, 0].set_title('Energy Level Distribution')
        axes[1, 0].set_xlabel('Energy Level')
        axes[1, 0].set_ylabel('Number of Clips')
        
        # Tempo distribution
        tempos = list(stats.tempo_distribution.keys())
        tempo_counts = list(stats.tempo_distribution.values())
        axes[1, 1].bar(tempos, tempo_counts, color='lightyellow')
        axes[1, 1].set_title('Tempo Range Distribution')
        axes[1, 1].set_xlabel('Tempo Range (BPM)')
        axes[1, 1].set_ylabel('Number of Clips')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distribution_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_dashboard_html(self, stats: TrainingDataStatistics, output_path: Path):
        """Generate HTML dashboard."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bachata Training Data Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
                .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Bachata Training Data Dashboard</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Dataset Overview</h3>
                    <p><strong>Total Clips:</strong> {stats.total_clips}</p>
                    <p><strong>Total Duration:</strong> {stats.total_duration:.1f} seconds ({stats.total_duration/60:.1f} minutes)</p>
                    <p><strong>Average Pose Quality:</strong> {stats.average_pose_quality:.2f}</p>
                </div>
                
                <div class="stat-card">
                    <h3>Quality Metrics</h3>
                    <p><strong>Clips with Quality Issues:</strong> {stats.clips_with_quality_issues}</p>
                    <p><strong>Clips with Consistency Issues:</strong> {stats.clips_with_consistency_issues}</p>
                    <p><strong>Overall Quality:</strong> {'Good' if stats.clips_with_quality_issues < stats.total_clips * 0.1 else 'Needs Attention'}</p>
                </div>
                
                <div class="stat-card">
                    <h3>Balance Scores</h3>
                    <p><strong>Category Balance:</strong> {stats.category_balance_score:.2f}</p>
                    <p><strong>Difficulty Balance:</strong> {stats.difficulty_balance_score:.2f}</p>
                    <p><strong>Energy Balance:</strong> {stats.energy_balance_score:.2f}</p>
                </div>
            </div>
            
            {'<div class="chart-container"><h2>Distribution Analysis</h2><img src="distribution_plots.png" alt="Distribution Plots" style="max-width: 100%; height: auto;"></div>' if VISUALIZATION_AVAILABLE else '<div class="chart-container"><h2>Distribution Analysis</h2><p><em>Visualization not available (matplotlib/seaborn not installed)</em></p></div>'}
            
            <h2>Detailed Distributions</h2>
            
            <h3>Category Distribution</h3>
            <table>
                <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
                {self._generate_table_rows(stats.category_distribution, stats.total_clips)}
            </table>
            
            <h3>Difficulty Distribution</h3>
            <table>
                <tr><th>Difficulty</th><th>Count</th><th>Percentage</th></tr>
                {self._generate_table_rows(stats.difficulty_distribution, stats.total_clips)}
            </table>
            
            <h3>Energy Distribution</h3>
            <table>
                <tr><th>Energy Level</th><th>Count</th><th>Percentage</th></tr>
                {self._generate_table_rows(stats.energy_distribution, stats.total_clips)}
            </table>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in stats.recommendations)}
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_table_rows(self, distribution: Dict[str, int], total: int) -> str:
        """Generate HTML table rows for distribution data."""
        rows = []
        for key, count in distribution.items():
            percentage = (count / total) * 100 if total > 0 else 0
            rows.append(f"<tr><td>{key}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>")
        return ''.join(rows)
    
    def _generate_comprehensive_report(self, basic_validation, pose_quality_results, 
                                     consistency_results, statistics, output_path: Path):
        """Generate comprehensive markdown report."""
        report_lines = [
            "# Comprehensive Training Data Validation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- **Total Clips:** {statistics.total_clips}",
            f"- **Total Duration:** {statistics.total_duration:.1f} seconds ({statistics.total_duration/60:.1f} minutes)",
            f"- **Average Pose Quality:** {statistics.average_pose_quality:.2f}/1.0",
            f"- **Clips with Quality Issues:** {statistics.clips_with_quality_issues}/{statistics.total_clips} ({statistics.clips_with_quality_issues/statistics.total_clips*100:.1f}%)",
            f"- **Clips with Consistency Issues:** {statistics.clips_with_consistency_issues}/{statistics.total_clips} ({statistics.clips_with_consistency_issues/statistics.total_clips*100:.1f}%)",
            "",
            "## Dataset Balance Analysis",
            f"- **Category Balance Score:** {statistics.category_balance_score:.2f}/1.0",
            f"- **Difficulty Balance Score:** {statistics.difficulty_balance_score:.2f}/1.0", 
            f"- **Energy Balance Score:** {statistics.energy_balance_score:.2f}/1.0",
            "",
            "## Distribution Analysis",
            "",
            "### Category Distribution",
        ]
        
        for category, count in statistics.category_distribution.items():
            percentage = (count / statistics.total_clips) * 100
            report_lines.append(f"- {category}: {count} clips ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "### Difficulty Distribution",
        ])
        
        for difficulty, count in statistics.difficulty_distribution.items():
            percentage = (count / statistics.total_clips) * 100
            report_lines.append(f"- {difficulty}: {count} clips ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "### Energy Distribution",
        ])
        
        for energy, count in statistics.energy_distribution.items():
            percentage = (count / statistics.total_clips) * 100
            report_lines.append(f"- {energy}: {count} clips ({percentage:.1f}%)")
        
        # Add quality issues
        quality_issues = [pq for pq in pose_quality_results if pq.issues]
        if quality_issues:
            report_lines.extend([
                "",
                "## Pose Quality Issues",
                ""
            ])
            for pq in quality_issues:
                report_lines.append(f"### {pq.clip_id}")
                report_lines.append(f"- Quality Score: {pq.quality_score:.2f}")
                report_lines.append(f"- Detection Rate: {pq.pose_detection_rate:.2f}")
                report_lines.append(f"- Average Confidence: {pq.average_confidence:.2f}")
                for issue in pq.issues:
                    report_lines.append(f"- Issue: {issue}")
                report_lines.append("")
        
        # Add consistency issues
        consistency_issues = [cr for cr in consistency_results if cr.inconsistencies]
        if consistency_issues:
            report_lines.extend([
                "",
                "## Annotation Consistency Issues",
                ""
            ])
            for cr in consistency_issues:
                report_lines.append(f"### {cr.clip_id}")
                report_lines.append(f"- Consistency Score: {cr.consistency_score:.2f}")
                for inconsistency in cr.inconsistencies:
                    report_lines.append(f"- {inconsistency}")
                report_lines.append("")
        
        # Add recommendations
        if statistics.recommendations:
            report_lines.extend([
                "",
                "## Recommendations",
                ""
            ])
            for rec in statistics.recommendations:
                report_lines.append(f"- {rec}")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))


def main():
    """Main function for running training data validation from command line."""
    validator = TrainingDataValidator()
    results = validator.run_comprehensive_validation()
    
    print("Comprehensive validation complete!")
    print(f"Report: {results['report_path']}")
    print(f"Dashboard: {results['dashboard_path']}")
    print(f"Results: {results['results_path']}")


if __name__ == "__main__":
    main()