"""
Annotation validation service for Bachata move clips.
Validates video quality standards and annotation completeness.
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from pydantic import ValidationError

# Optional cv2 import for video validation
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..models.annotation_schema import (
    AnnotationCollection, 
    MoveAnnotation, 
    QualityStandards,
    DifficultyLevel,
    EnergyLevel
)


class AnnotationValidator:
    """Validates annotation data and video quality standards."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.video_base_path = self.data_dir / "Bachata_steps"  # Base path for video files
        
    def load_annotations(self, annotation_file: str = "bachata_annotations.json") -> AnnotationCollection:
        """Load and validate annotation collection from JSON file."""
        annotation_path = self.data_dir / annotation_file
        
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        try:
            return AnnotationCollection(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid annotation format: {e}")
    
    def validate_video_file(self, video_path: str) -> Dict[str, any]:
        """
        Validate a single video file against quality standards.
        
        Returns:
            Dict with validation results including:
            - exists: bool
            - duration: float (seconds)
            - resolution: tuple (width, height)
            - fps: float
            - quality_checks: dict
        """
        # Handle path - if video_path already includes Bachata_steps, use data_dir directly
        if video_path.startswith("Bachata_steps/"):
            full_path = self.data_dir / video_path
        else:
            full_path = self.video_base_path / video_path
        
        result = {
            "video_path": video_path,
            "exists": False,
            "duration": 0.0,
            "resolution": (0, 0),
            "fps": 0.0,
            "quality_checks": {
                "duration_valid": False,
                "resolution_adequate": False,
                "fps_adequate": False,
                "file_accessible": False
            },
            "issues": []
        }
        
        # Check if file exists
        if not full_path.exists():
            result["issues"].append(f"Video file not found: {full_path}")
            return result
        
        result["exists"] = True
        result["quality_checks"]["file_accessible"] = True
        
        # Basic file size check
        try:
            file_size = full_path.stat().st_size
            if file_size < 1024:  # Less than 1KB
                result["issues"].append("File appears to be empty or corrupted")
                return result
        except Exception as e:
            result["issues"].append(f"Cannot access file: {str(e)}")
            return result
        
        # Video validation with cv2 if available
        if CV2_AVAILABLE:
            try:
                # Open video file
                cap = cv2.VideoCapture(str(full_path))
                
                if not cap.isOpened():
                    result["issues"].append("Cannot open video file")
                    return result
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                duration = frame_count / fps if fps > 0 else 0
                
                result["duration"] = duration
                result["resolution"] = (width, height)
                result["fps"] = fps
                
                # Validate duration
                if QualityStandards.MIN_DURATION <= duration <= QualityStandards.MAX_DURATION:
                    result["quality_checks"]["duration_valid"] = True
                else:
                    result["issues"].append(
                        f"Duration {duration:.1f}s outside valid range "
                        f"({QualityStandards.MIN_DURATION}-{QualityStandards.MAX_DURATION}s)"
                    )
                
                # Validate resolution (minimum 480p)
                if width >= 640 and height >= 480:
                    result["quality_checks"]["resolution_adequate"] = True
                else:
                    result["issues"].append(f"Resolution {width}x{height} below minimum 640x480")
                
                # Validate FPS (minimum 24fps)
                if fps >= 24:
                    result["quality_checks"]["fps_adequate"] = True
                else:
                    result["issues"].append(f"FPS {fps} below minimum 24")
                
                cap.release()
                
            except Exception as e:
                result["issues"].append(f"Error processing video: {str(e)}")
        else:
            # Basic validation without cv2
            result["issues"].append("OpenCV not available - limited video validation")
            # Assume basic quality checks pass for existing files
            result["quality_checks"]["duration_valid"] = True
            result["quality_checks"]["resolution_adequate"] = True  
            result["quality_checks"]["fps_adequate"] = True
        
        return result
    
    def validate_annotation_data(self, annotation: MoveAnnotation) -> Dict[str, any]:
        """
        Validate annotation data completeness and consistency.
        
        Returns:
            Dict with validation results
        """
        result = {
            "clip_id": annotation.clip_id,
            "data_checks": {
                "required_fields_present": True,
                "tempo_in_range": False,
                "difficulty_appropriate": True,
                "energy_level_valid": True
            },
            "issues": [],
            "warnings": []
        }
        
        # Check tempo range
        if QualityStandards.MIN_TEMPO <= annotation.estimated_tempo <= QualityStandards.MAX_TEMPO:
            result["data_checks"]["tempo_in_range"] = True
        else:
            result["issues"].append(
                f"Tempo {annotation.estimated_tempo} outside valid range "
                f"({QualityStandards.MIN_TEMPO}-{QualityStandards.MAX_TEMPO} BPM)"
            )
        
        # Check for missing optional but important fields
        if not annotation.notes or len(annotation.notes.strip()) < 10:
            result["warnings"].append("Notes field is empty or too brief")
        
        # Validate difficulty vs move complexity
        advanced_moves = ["body_roll", "hammerlock", "shadow_position", "combination", "arm_styling"]
        if any(move in annotation.move_label.lower() for move in advanced_moves):
            if annotation.difficulty == DifficultyLevel.BEGINNER:
                result["warnings"].append(
                    f"Move '{annotation.move_label}' marked as beginner but typically advanced"
                )
        
        return result
    
    def validate_collection(self, annotation_file: str = "bachata_annotations.json") -> Dict[str, any]:
        """
        Validate entire annotation collection including video files.
        
        Returns:
            Comprehensive validation report
        """
        try:
            collection = self.load_annotations(annotation_file)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection_stats": {},
                "video_validation": [],
                "annotation_validation": []
            }
        
        # Collection statistics
        stats = {
            "total_clips": collection.total_clips,
            "categories": len(collection.move_categories),
            "difficulty_distribution": {},
            "energy_distribution": {},
            "tempo_range": {"min": float('inf'), "max": 0}
        }
        
        # Calculate distributions
        for clip in collection.clips:
            # Difficulty distribution
            diff = clip.difficulty
            stats["difficulty_distribution"][diff] = stats["difficulty_distribution"].get(diff, 0) + 1
            
            # Energy distribution
            energy = clip.energy_level
            stats["energy_distribution"][energy] = stats["energy_distribution"].get(energy, 0) + 1
            
            # Tempo range
            tempo = clip.estimated_tempo
            stats["tempo_range"]["min"] = min(stats["tempo_range"]["min"], tempo)
            stats["tempo_range"]["max"] = max(stats["tempo_range"]["max"], tempo)
        
        # Validate each video file
        video_results = []
        for clip in collection.clips:
            video_result = self.validate_video_file(clip.video_path)
            video_results.append(video_result)
        
        # Validate each annotation
        annotation_results = []
        for clip in collection.clips:
            annotation_result = self.validate_annotation_data(clip)
            annotation_results.append(annotation_result)
        
        # Summary statistics
        video_issues = sum(1 for r in video_results if r["issues"])
        annotation_issues = sum(1 for r in annotation_results if r["issues"])
        
        return {
            "success": True,
            "collection_stats": stats,
            "video_validation": video_results,
            "annotation_validation": annotation_results,
            "summary": {
                "total_clips": collection.total_clips,
                "videos_with_issues": video_issues,
                "annotations_with_issues": annotation_issues,
                "overall_quality": "good" if (video_issues + annotation_issues) < collection.total_clips * 0.1 else "needs_attention"
            }
        }
    
    def generate_validation_report(self, annotation_file: str = "bachata_annotations.json") -> str:
        """Generate a human-readable validation report."""
        validation_result = self.validate_collection(annotation_file)
        
        if not validation_result["success"]:
            return f"Validation failed: {validation_result['error']}"
        
        stats = validation_result["collection_stats"]
        summary = validation_result["summary"]
        
        report = []
        report.append("# Bachata Annotation Collection Validation Report")
        report.append("")
        report.append("## Collection Overview")
        report.append(f"- Total clips: {stats['total_clips']}")
        report.append(f"- Categories: {stats['categories']}")
        report.append(f"- Tempo range: {stats['tempo_range']['min']}-{stats['tempo_range']['max']} BPM")
        report.append("")
        
        report.append("## Difficulty Distribution")
        for difficulty, count in stats["difficulty_distribution"].items():
            percentage = (count / stats['total_clips']) * 100
            report.append(f"- {difficulty}: {count} clips ({percentage:.1f}%)")
        report.append("")
        
        report.append("## Energy Distribution")
        for energy, count in stats["energy_distribution"].items():
            percentage = (count / stats['total_clips']) * 100
            report.append(f"- {energy}: {count} clips ({percentage:.1f}%)")
        report.append("")
        
        report.append("## Quality Assessment")
        report.append(f"- Videos with issues: {summary['videos_with_issues']}")
        report.append(f"- Annotations with issues: {summary['annotations_with_issues']}")
        report.append(f"- Overall quality: {summary['overall_quality']}")
        report.append("")
        
        # Detailed issues
        if summary['videos_with_issues'] > 0:
            report.append("## Video Issues")
            for result in validation_result["video_validation"]:
                if result["issues"]:
                    report.append(f"### {result['video_path']}")
                    for issue in result["issues"]:
                        report.append(f"- {issue}")
            report.append("")
        
        if summary['annotations_with_issues'] > 0:
            report.append("## Annotation Issues")
            for result in validation_result["annotation_validation"]:
                if result["issues"]:
                    report.append(f"### {result['clip_id']}")
                    for issue in result["issues"]:
                        report.append(f"- {issue}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function for running validation from command line."""
    validator = AnnotationValidator()
    report = validator.generate_validation_report()
    print(report)


if __name__ == "__main__":
    main()