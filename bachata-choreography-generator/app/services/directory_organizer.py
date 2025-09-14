"""
Directory organizer for Bachata move clips.
Organizes existing video files into structured categories.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from ..models.annotation_schema import AnnotationCollection, MoveCategory


class DirectoryOrganizer:
    """Organizes video files into structured directory layout."""
    
    def __init__(self, base_video_path: str = None, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        if base_video_path is None:
            self.base_video_path = self.data_dir / "Bachata_steps"
        else:
            self.base_video_path = Path(base_video_path)
        
        # Category mapping for organization
        self.category_directories = {
            "basic_step": "basic_moves",
            "cross_body_lead": "partner_work", 
            "double_cross_body_lead": "partner_work",
            "lady_right_turn": "turns_spins",
            "lady_left_turn": "turns_spins",
            "forward_backward": "basic_moves",
            "dip": "advanced",
            "body_roll": "styling",
            "hammerlock": "advanced",
            "shadow_position": "advanced", 
            "combination": "advanced",
            "arm_styling": "styling"
        }
        
        # Standard directory structure
        self.standard_categories = [
            "basic_moves",
            "partner_work", 
            "turns_spins",
            "styling",
            "advanced"
        ]
    
    def load_annotations(self, annotation_file: str = "bachata_annotations.json") -> AnnotationCollection:
        """Load annotation collection from JSON file."""
        annotation_path = self.data_dir / annotation_file
        
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        return AnnotationCollection(**data)
    
    def analyze_current_structure(self) -> Dict[str, any]:
        """Analyze the current directory structure."""
        if not self.base_video_path.exists():
            return {
                "exists": False,
                "error": f"Base video path does not exist: {self.base_video_path}"
            }
        
        structure = {
            "exists": True,
            "directories": [],
            "files": [],
            "total_files": 0,
            "organization_status": "unorganized"
        }
        
        # Scan current structure
        for item in self.base_video_path.rglob("*"):
            if item.is_dir():
                rel_path = item.relative_to(self.base_video_path)
                structure["directories"].append(str(rel_path))
            elif item.is_file() and item.suffix.lower() in ['.mp4', '.avi', '.mov']:
                rel_path = item.relative_to(self.base_video_path)
                structure["files"].append(str(rel_path))
                structure["total_files"] += 1
        
        # Check if already organized
        has_standard_dirs = all(
            any(cat in d for d in structure["directories"]) 
            for cat in self.standard_categories
        )
        
        if has_standard_dirs:
            structure["organization_status"] = "organized"
        elif len(structure["directories"]) > 0:
            structure["organization_status"] = "partially_organized"
        
        return structure
    
    def create_directory_structure(self, dry_run: bool = True) -> Dict[str, any]:
        """
        Create the standard directory structure for organizing clips.
        
        Args:
            dry_run: If True, only simulate the creation without making changes
        """
        result = {
            "success": True,
            "created_directories": [],
            "existing_directories": [],
            "errors": []
        }
        
        for category in self.standard_categories:
            category_path = self.base_video_path / category
            
            if category_path.exists():
                result["existing_directories"].append(str(category_path))
            else:
                if not dry_run:
                    try:
                        category_path.mkdir(parents=True, exist_ok=True)
                        result["created_directories"].append(str(category_path))
                    except Exception as e:
                        result["errors"].append(f"Failed to create {category_path}: {str(e)}")
                        result["success"] = False
                else:
                    result["created_directories"].append(str(category_path) + " (dry run)")
        
        return result
    
    def organize_clips_by_annotations(self, annotation_file: str = "bachata_annotations.json", 
                                    dry_run: bool = True) -> Dict[str, any]:
        """
        Organize video clips based on annotation data.
        
        Args:
            annotation_file: Path to annotation JSON file
            dry_run: If True, only simulate the organization without moving files
        """
        try:
            collection = self.load_annotations(annotation_file)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load annotations: {str(e)}",
                "moves": []
            }
        
        result = {
            "success": True,
            "moves": [],
            "errors": [],
            "summary": {
                "total_clips": len(collection.clips),
                "moved": 0,
                "already_organized": 0,
                "missing_files": 0
            }
        }
        
        # First create directory structure
        dir_result = self.create_directory_structure(dry_run)
        if not dir_result["success"]:
            result["errors"].extend(dir_result["errors"])
        
        # Process each clip
        for clip in collection.clips:
            move_result = self._organize_single_clip(clip, dry_run)
            result["moves"].append(move_result)
            
            if move_result["status"] == "moved":
                result["summary"]["moved"] += 1
            elif move_result["status"] == "already_organized":
                result["summary"]["already_organized"] += 1
            elif move_result["status"] == "missing":
                result["summary"]["missing_files"] += 1
            
            if move_result["error"]:
                result["errors"].append(move_result["error"])
        
        return result
    
    def _organize_single_clip(self, clip, dry_run: bool = True) -> Dict[str, any]:
        """Organize a single clip based on its annotation."""
        # Handle path - if video_path already includes Bachata_steps, use data_dir directly
        if clip.video_path.startswith("Bachata_steps/"):
            current_path = self.data_dir / clip.video_path
        else:
            current_path = self.base_video_path / clip.video_path
        
        # Determine target category
        move_label = clip.move_label.lower()
        target_category = None
        
        for label_key, category in self.category_directories.items():
            if label_key in move_label:
                target_category = category
                break
        
        if not target_category:
            target_category = "advanced"  # Default fallback
        
        # Construct target path
        filename = Path(clip.video_path).name
        target_path = self.base_video_path / target_category / filename
        
        result = {
            "clip_id": clip.clip_id,
            "current_path": str(current_path),
            "target_path": str(target_path),
            "target_category": target_category,
            "status": "unknown",
            "error": None
        }
        
        # Check if source file exists
        if not current_path.exists():
            result["status"] = "missing"
            result["error"] = f"Source file not found: {current_path}"
            return result
        
        # Check if already in correct location
        if current_path == target_path:
            result["status"] = "already_organized"
            return result
        
        # Check if target already exists
        if target_path.exists():
            result["status"] = "target_exists"
            result["error"] = f"Target file already exists: {target_path}"
            return result
        
        # Move the file
        if not dry_run:
            try:
                # Ensure target directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_path), str(target_path))
                result["status"] = "moved"
            except Exception as e:
                result["status"] = "error"
                result["error"] = f"Failed to move file: {str(e)}"
        else:
            result["status"] = "moved"
            result["target_path"] += " (dry run)"
        
        return result
    
    def generate_organization_report(self, annotation_file: str = "bachata_annotations.json") -> str:
        """Generate a report showing current organization status and proposed changes."""
        
        # Analyze current structure
        current_structure = self.analyze_current_structure()
        
        # Simulate organization
        organization_result = self.organize_clips_by_annotations(annotation_file, dry_run=True)
        
        report = []
        report.append("# Directory Organization Report")
        report.append("")
        
        # Current status
        report.append("## Current Structure")
        if current_structure["exists"]:
            report.append(f"- Total video files: {current_structure['total_files']}")
            report.append(f"- Organization status: {current_structure['organization_status']}")
            report.append(f"- Existing directories: {len(current_structure['directories'])}")
            
            if current_structure["directories"]:
                report.append("\n### Current Directories:")
                for directory in sorted(current_structure["directories"]):
                    report.append(f"- {directory}")
        else:
            report.append(f"- Error: {current_structure['error']}")
        
        report.append("")
        
        # Proposed organization
        if organization_result["success"]:
            summary = organization_result["summary"]
            report.append("## Proposed Organization")
            report.append(f"- Total clips to process: {summary['total_clips']}")
            report.append(f"- Files to move: {summary['moved']}")
            report.append(f"- Already organized: {summary['already_organized']}")
            report.append(f"- Missing files: {summary['missing_files']}")
            report.append("")
            
            # Category breakdown
            category_counts = {}
            for move in organization_result["moves"]:
                if move["status"] in ["moved", "already_organized"]:
                    cat = move["target_category"]
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            report.append("### Files per Category:")
            for category in self.standard_categories:
                count = category_counts.get(category, 0)
                report.append(f"- {category}: {count} clips")
            
            # Issues
            if organization_result["errors"]:
                report.append("\n### Issues Found:")
                for error in organization_result["errors"][:10]:  # Limit to first 10
                    report.append(f"- {error}")
                if len(organization_result["errors"]) > 10:
                    report.append(f"- ... and {len(organization_result['errors']) - 10} more issues")
        
        return "\n".join(report)


def main():
    """Main function for running organization from command line."""
    organizer = DirectoryOrganizer()
    report = organizer.generate_organization_report()
    print(report)


if __name__ == "__main__":
    main()