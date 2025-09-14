"""
Annotation interface for managing Bachata move clip annotations.
Provides tools for creating, editing, and exporting annotations.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..models.annotation_schema import (
    AnnotationCollection, 
    MoveAnnotation, 
    DifficultyLevel,
    EnergyLevel,
    MoveCategory
)


class AnnotationInterface:
    """Interface for managing move clip annotations."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_annotations(self, annotation_file: str = "bachata_annotations.json") -> AnnotationCollection:
        """Load existing annotations from JSON file."""
        annotation_path = self.data_dir / annotation_file
        
        if not annotation_path.exists():
            # Create empty collection if file doesn't exist
            return AnnotationCollection(
                instructions="Annotations for bachata video training clips",
                move_categories=[],
                clips=[]
            )
        
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        return AnnotationCollection(**data)
    
    def save_annotations(self, collection: AnnotationCollection, 
                        annotation_file: str = "bachata_annotations.json") -> bool:
        """Save annotations to JSON file."""
        annotation_path = self.data_dir / annotation_file
        
        try:
            # Convert to dict for JSON serialization
            data = {
                "instructions": collection.instructions,
                "move_categories": collection.move_categories,
                "clips": [
                    {
                        "clip_id": clip.clip_id,
                        "video_path": clip.video_path,
                        "move_label": clip.move_label,
                        "energy_level": clip.energy_level,
                        "estimated_tempo": clip.estimated_tempo,
                        "difficulty": clip.difficulty,
                        "lead_follow_roles": clip.lead_follow_roles,
                        "notes": clip.notes
                    }
                    for clip in collection.clips
                ]
            }
            
            with open(annotation_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False
    
    def export_to_csv(self, annotation_file: str = "bachata_annotations.json",
                     csv_file: str = "bachata_annotations.csv") -> bool:
        """Export annotations to CSV format for easy editing."""
        try:
            collection = self.load_annotations(annotation_file)
            csv_path = self.data_dir / csv_file
            
            # Prepare data for CSV
            csv_data = []
            for clip in collection.clips:
                row = {
                    'clip_id': clip.clip_id,
                    'video_path': clip.video_path,
                    'move_label': clip.move_label,
                    'energy_level': clip.energy_level,
                    'estimated_tempo': clip.estimated_tempo,
                    'difficulty': clip.difficulty,
                    'lead_follow_roles': clip.lead_follow_roles,
                    'notes': clip.notes,
                    # Additional fields for enhancement
                    'duration_seconds': getattr(clip, 'duration_seconds', ''),
                    'video_quality': getattr(clip, 'video_quality', 'good'),
                    'lighting_quality': getattr(clip, 'lighting_quality', 'good'),
                    'full_body_visible': getattr(clip, 'full_body_visible', True),
                    'tempo_range_min': getattr(clip, 'tempo_range_min', ''),
                    'tempo_range_max': getattr(clip, 'tempo_range_max', ''),
                    'compatible_moves': ','.join(getattr(clip, 'compatible_moves', [])),
                    'annotator': getattr(clip, 'annotator', ''),
                    'annotation_date': getattr(clip, 'annotation_date', '')
                }
                csv_data.append(row)
            
            # Write to CSV using built-in csv module if pandas not available
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
            else:
                # Fallback to built-in csv module
                if csv_data:
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                        writer.writeheader()
                        writer.writerows(csv_data)
            
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def import_from_csv(self, csv_file: str = "bachata_annotations.csv",
                       annotation_file: str = "bachata_annotations.json") -> bool:
        """Import annotations from CSV file."""
        try:
            csv_path = self.data_dir / csv_file
            
            if not csv_path.exists():
                print(f"CSV file not found: {csv_path}")
                return False
            
            # Read CSV
            if PANDAS_AVAILABLE:
                df = pd.read_csv(csv_path)
                rows = df.to_dict('records')
            else:
                # Fallback to built-in csv module
                rows = []
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            
            # Convert to annotation objects
            clips = []
            for row in rows:
                # Handle compatible_moves field
                compatible_moves = []
                compatible_moves_str = row.get('compatible_moves', '')
                if compatible_moves_str and str(compatible_moves_str).strip() and str(compatible_moves_str) != 'nan':
                    compatible_moves = [m.strip() for m in str(compatible_moves_str).split(',') if m.strip()]
                
                # Helper function to safely get string value
                def safe_str(value):
                    if value is None or str(value).lower() in ['nan', 'none', '']:
                        return ''
                    return str(value).strip()
                
                # Helper function to check if value is empty/invalid
                def is_empty(value):
                    return value is None or str(value).lower() in ['nan', 'none', ''] or str(value).strip() == ''
                
                clip_data = {
                    'clip_id': safe_str(row['clip_id']),
                    'video_path': safe_str(row['video_path']),
                    'move_label': safe_str(row['move_label']),
                    'energy_level': safe_str(row['energy_level']),
                    'estimated_tempo': int(float(row['estimated_tempo'])),
                    'difficulty': safe_str(row['difficulty']),
                    'lead_follow_roles': safe_str(row['lead_follow_roles']),
                    'notes': safe_str(row['notes'])
                }
                
                # Add optional fields if present
                optional_fields = [
                    'duration_seconds', 'video_quality', 'lighting_quality', 
                    'full_body_visible', 'tempo_range_min', 'tempo_range_max',
                    'annotator', 'annotation_date'
                ]
                
                for field in optional_fields:
                    value = row.get(field, '')
                    if not is_empty(value):
                        try:
                            if field in ['tempo_range_min', 'tempo_range_max']:
                                clip_data[field] = int(float(value))
                            elif field == 'duration_seconds':
                                clip_data[field] = float(value)
                            elif field == 'full_body_visible':
                                clip_data[field] = str(value).lower() in ['true', '1', 'yes']
                            else:
                                clip_data[field] = safe_str(value)
                        except (ValueError, TypeError):
                            # Skip invalid values
                            continue
                
                if compatible_moves:
                    clip_data['compatible_moves'] = compatible_moves
                
                clips.append(MoveAnnotation(**clip_data))
            
            # Create collection
            move_categories = list(set(clip.move_label for clip in clips))
            collection = AnnotationCollection(
                instructions="Annotations for bachata video training clips",
                move_categories=move_categories,
                clips=clips
            )
            
            # Save to JSON
            return self.save_annotations(collection, annotation_file)
            
        except Exception as e:
            print(f"Error importing from CSV: {e}")
            return False
    
    def create_annotation_template(self, template_file: str = "annotation_template.csv") -> bool:
        """Create a CSV template for manual annotation."""
        template_path = self.data_dir / template_file
        
        # Template headers with descriptions
        headers = [
            'clip_id',  # Unique identifier (e.g., basic_step_1)
            'video_path',  # Path to video file (e.g., Bachata_steps/basic_steps/basic_step_1.mp4)
            'move_label',  # Move name (e.g., basic_step, cross_body_lead)
            'energy_level',  # low, medium, high
            'estimated_tempo',  # BPM (80-160)
            'difficulty',  # beginner, intermediate, advanced
            'lead_follow_roles',  # lead_focus, follow_focus, both
            'notes',  # Description of the move
            'duration_seconds',  # Duration in seconds (optional)
            'video_quality',  # good, fair, poor (optional)
            'lighting_quality',  # good, fair, poor (optional)
            'full_body_visible',  # TRUE/FALSE (optional)
            'tempo_range_min',  # Minimum compatible BPM (optional)
            'tempo_range_max',  # Maximum compatible BPM (optional)
            'compatible_moves',  # Comma-separated list of compatible move IDs (optional)
            'annotator',  # Person who created annotation (optional)
            'annotation_date'  # Date of annotation (optional)
        ]
        
        # Create sample rows
        sample_data = [
            {
                'clip_id': 'example_move_1',
                'video_path': 'Bachata_steps/basic_moves/example_move_1.mp4',
                'move_label': 'basic_step',
                'energy_level': 'medium',
                'estimated_tempo': 120,
                'difficulty': 'beginner',
                'lead_follow_roles': 'both',
                'notes': 'Basic bachata step with weight transfer',
                'duration_seconds': 8.0,
                'video_quality': 'good',
                'lighting_quality': 'good',
                'full_body_visible': True,
                'tempo_range_min': 110,
                'tempo_range_max': 130,
                'compatible_moves': 'cross_body_lead_1,forward_backward_1',
                'annotator': 'Your Name',
                'annotation_date': datetime.now().strftime('%Y-%m-%d')
            }
        ]
        
        try:
            # Write template using available method
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(sample_data)
                df.to_csv(template_path, index=False)
            else:
                # Fallback to built-in csv module
                with open(template_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(sample_data)
            
            # Create instructions file
            instructions_path = self.data_dir / "annotation_instructions.md"
            instructions = self._create_annotation_instructions()
            with open(instructions_path, 'w') as f:
                f.write(instructions)
            
            return True
        except Exception as e:
            print(f"Error creating template: {e}")
            return False
    
    def _create_annotation_instructions(self) -> str:
        """Create detailed instructions for manual annotation."""
        return """# Bachata Move Annotation Instructions

## Overview
This guide explains how to annotate Bachata move clips for the choreography generator.

## Required Fields

### clip_id
- Unique identifier for each clip
- Format: `{move_type}_{number}` (e.g., basic_step_1, cross_body_lead_2)
- Must be unique across all clips

### video_path
- Relative path to the video file
- Format: `Bachata_steps/{category}/{filename}.mp4`
- Categories: basic_moves, partner_work, turns_spins, styling, advanced

### move_label
- Primary move name/type
- Use consistent naming (e.g., basic_step, cross_body_lead, lady_right_turn)
- This determines the move category

### energy_level
- Overall energy/intensity of the move
- Options: `low`, `medium`, `high`
- Consider speed, complexity, and physical intensity

### estimated_tempo
- Compatible BPM range for this move
- Range: 80-160 BPM
- Consider the natural rhythm of the move

### difficulty
- Technical difficulty level
- Options: `beginner`, `intermediate`, `advanced`
- Consider coordination, balance, and lead/follow complexity

### lead_follow_roles
- Which role is emphasized in the clip
- Options: `lead_focus`, `follow_focus`, `both`
- Use `both` when both partners are equally featured

### notes
- Detailed description of the move
- Include timing, key elements, and any special considerations
- Minimum 10 characters recommended

## Optional Fields

### duration_seconds
- Actual clip duration in seconds
- Will be validated against video file

### video_quality / lighting_quality
- Assessment of technical quality
- Options: `good`, `fair`, `poor`

### full_body_visible
- Whether dancers' full bodies are visible
- Options: `TRUE`, `FALSE`

### tempo_range_min / tempo_range_max
- Specific BPM compatibility range
- Must be within 80-160 BPM
- tempo_range_max must be >= tempo_range_min

### compatible_moves
- List of move IDs that work well in sequence
- Comma-separated (e.g., `basic_step_1,cross_body_lead_2`)

### annotator
- Name of person creating the annotation

### annotation_date
- Date of annotation (YYYY-MM-DD format)

## Quality Standards

### Video Requirements
- Duration: 5-20 seconds
- Resolution: Minimum 640x480
- Frame rate: Minimum 24fps
- Full body visible
- Good lighting
- Stable camera

### Annotation Requirements
- All required fields must be completed
- Tempo must be within 80-160 BPM range
- Notes should be descriptive and informative
- Difficulty should match move complexity

## Tips for Good Annotations

1. **Be Consistent**: Use the same terminology across similar moves
2. **Be Descriptive**: Include key elements that make the move unique
3. **Consider Context**: Think about how moves connect in sequences
4. **Validate Quality**: Ensure video meets technical standards
5. **Double-Check**: Review annotations for accuracy and completeness

## Common Move Categories

- **basic_step**: Fundamental bachata steps
- **cross_body_lead**: Partner crosses in front of lead
- **lady_right_turn / lady_left_turn**: Follower turns
- **forward_backward**: Linear movement patterns
- **dip**: Dramatic dipping movements
- **body_roll**: Sensual body movements
- **hammerlock**: Arm positioning moves
- **shadow_position**: Side-by-side positioning
- **combination**: Complex sequences
- **arm_styling**: Decorative arm movements
"""
    
    def add_annotation(self, clip_data: Dict, annotation_file: str = "bachata_annotations.json") -> bool:
        """Add a new annotation to the collection."""
        try:
            collection = self.load_annotations(annotation_file)
            
            # Create new annotation
            new_clip = MoveAnnotation(**clip_data)
            
            # Check for duplicate clip_id
            existing_ids = [clip.clip_id for clip in collection.clips]
            if new_clip.clip_id in existing_ids:
                print(f"Error: clip_id '{new_clip.clip_id}' already exists")
                return False
            
            # Add to collection
            collection.clips.append(new_clip)
            
            # Update move categories
            if new_clip.move_label not in collection.move_categories:
                collection.move_categories.append(new_clip.move_label)
            
            # Save updated collection
            return self.save_annotations(collection, annotation_file)
            
        except Exception as e:
            print(f"Error adding annotation: {e}")
            return False
    
    def update_annotation(self, clip_id: str, updates: Dict, 
                         annotation_file: str = "bachata_annotations.json") -> bool:
        """Update an existing annotation."""
        try:
            collection = self.load_annotations(annotation_file)
            
            # Find and update clip
            for i, clip in enumerate(collection.clips):
                if clip.clip_id == clip_id:
                    # Create updated clip data
                    clip_dict = clip.dict()
                    clip_dict.update(updates)
                    
                    # Replace with updated clip
                    collection.clips[i] = MoveAnnotation(**clip_dict)
                    
                    # Save updated collection
                    return self.save_annotations(collection, annotation_file)
            
            print(f"Error: clip_id '{clip_id}' not found")
            return False
            
        except Exception as e:
            print(f"Error updating annotation: {e}")
            return False


def main():
    """Main function for running annotation interface from command line."""
    interface = AnnotationInterface()
    
    # Create template and instructions
    print("Creating annotation template...")
    if interface.create_annotation_template():
        print("✓ Template created: data/annotation_template.csv")
        print("✓ Instructions created: data/annotation_instructions.md")
    
    # Export existing annotations to CSV
    print("\nExporting existing annotations to CSV...")
    if interface.export_to_csv():
        print("✓ Annotations exported: data/bachata_annotations.csv")
    else:
        print("! No existing annotations found or export failed")


if __name__ == "__main__":
    main()