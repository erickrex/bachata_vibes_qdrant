"""
Annotation schema for Bachata move clips.
Defines the structure and validation for move clip annotations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class DifficultyLevel(str, Enum):
    """Difficulty levels for Bachata moves."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"


class EnergyLevel(str, Enum):
    """Energy levels for move clips."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MoveCategory(str, Enum):
    """Categories for organizing move clips based on existing data structure."""
    BASIC_STEP = "basic_step"
    CROSS_BODY_LEAD = "cross_body_lead"
    LADY_RIGHT_TURN = "lady_right_turn"
    DIPS = "dips"
    FORWARD_BACKWARD = "forward_backward"
    DOUBLE_CROSS_BODY_LEAD = "double_cross_body_lead"
    LADY_LEFT_TURN = "lady_left_turn"
    BODY_ROLL = "body_roll"
    HAMMERLOCK = "hammerlock"
    SHADOW_POSITION = "shadow_position"
    COMBINATION = "combination"
    ARM_STYLING = "arm_styling"


class MoveAnnotation(BaseModel):
    """
    Annotation schema for a single Bachata move clip.
    Matches existing data structure with enhancements.
    """
    
    # Required fields (matching existing structure)
    clip_id: str = Field(..., description="Unique identifier for the move clip")
    video_path: str = Field(..., description="Path to the video file")
    move_label: str = Field(..., description="Descriptive name of the move")
    energy_level: EnergyLevel = Field(..., description="Energy level of the move")
    estimated_tempo: int = Field(..., ge=80, le=160, description="Estimated BPM compatibility (80-160)")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level")
    lead_follow_roles: str = Field(..., description="Focus on lead, follow, or both")
    notes: str = Field(..., description="Additional notes about the move")
    
    # Enhanced metadata (optional for backward compatibility)
    category: Optional[MoveCategory] = Field(None, description="Move category derived from move_label")
    duration_seconds: Optional[float] = Field(None, ge=3.0, le=20.0, description="Clip duration in seconds")
    
    # Technical metadata for quality validation
    video_quality: Optional[str] = Field("good", description="Video quality assessment")
    lighting_quality: Optional[str] = Field("good", description="Lighting quality assessment")
    full_body_visible: Optional[bool] = Field(True, description="Whether full body is visible")
    
    # Compatibility and transition info
    tempo_range_min: Optional[int] = Field(None, ge=80, le=160, description="Minimum compatible BPM")
    tempo_range_max: Optional[int] = Field(None, ge=80, le=160, description="Maximum compatible BPM")
    compatible_moves: Optional[List[str]] = Field(default_factory=list, description="List of compatible move IDs for transitions")
    
    # Annotation metadata
    annotator: Optional[str] = Field(None, description="Person who created the annotation")
    annotation_date: Optional[str] = Field(None, description="Date of annotation")
    
    def __init__(self, **data):
        """Initialize with automatic category derivation from move_label."""
        super().__init__(**data)
        if not self.category and self.move_label:
            self.category = self._derive_category_from_label(self.move_label)
    
    def _derive_category_from_label(self, label: str) -> Optional[MoveCategory]:
        """Derive category from move label."""
        label_lower = label.lower()
        
        category_mapping = {
            "basic_step": MoveCategory.BASIC_STEP,
            "cross_body_lead": MoveCategory.CROSS_BODY_LEAD,
            "lady_right_turn": MoveCategory.LADY_RIGHT_TURN,
            "lady_left_turn": MoveCategory.LADY_LEFT_TURN,
            "dip": MoveCategory.DIPS,
            "forward_backward": MoveCategory.FORWARD_BACKWARD,
            "body_roll": MoveCategory.BODY_ROLL,
            "hammerlock": MoveCategory.HAMMERLOCK,
            "shadow_position": MoveCategory.SHADOW_POSITION,
            "combination": MoveCategory.COMBINATION,
            "arm_styling": MoveCategory.ARM_STYLING
        }
        
        for key, category in category_mapping.items():
            if key in label_lower:
                return category
        
        return None

    @validator('tempo_range_max')
    def validate_tempo_range(cls, v, values):
        """Ensure tempo_range_max >= tempo_range_min if both are provided."""
        if v is not None and 'tempo_range_min' in values and values['tempo_range_min'] is not None:
            if v < values['tempo_range_min']:
                raise ValueError('tempo_range_max must be >= tempo_range_min')
        return v

    @validator('estimated_tempo')
    def validate_estimated_tempo_in_range(cls, v, values):
        """Ensure estimated_tempo is within the specified range if provided."""
        if 'tempo_range_min' in values and values['tempo_range_min'] is not None:
            if v < values['tempo_range_min']:
                raise ValueError('estimated_tempo must be >= tempo_range_min')
        if 'tempo_range_max' in values and values['tempo_range_max'] is not None:
            if v > values['tempo_range_max']:
                raise ValueError('estimated_tempo must be <= tempo_range_max')
        return v


class AnnotationCollection(BaseModel):
    """Collection of move annotations matching existing JSON structure."""
    
    instructions: str = Field(..., description="Instructions for the annotation collection")
    move_categories: List[str] = Field(..., description="List of available move categories")
    clips: List[MoveAnnotation] = Field(..., description="List of move annotations")
    
    # Additional metadata
    collection_name: Optional[str] = Field("Bachata Move Clips", description="Name of the annotation collection")
    version: Optional[str] = Field("1.0", description="Version of the annotation schema")
    
    @property
    def total_clips(self) -> int:
        """Total number of clips in collection."""
        return len(self.clips)
    
    def get_clips_by_category(self, category: str) -> List[MoveAnnotation]:
        """Get all clips for a specific category."""
        return [clip for clip in self.clips if clip.move_label == category or 
                (clip.category and clip.category.value == category)]
    
    def get_clips_by_difficulty(self, difficulty: DifficultyLevel) -> List[MoveAnnotation]:
        """Get all clips for a specific difficulty level."""
        return [clip for clip in self.clips if clip.difficulty == difficulty]
    
    def get_clips_by_tempo_range(self, min_tempo: int, max_tempo: int) -> List[MoveAnnotation]:
        """Get clips within a specific tempo range."""
        return [clip for clip in self.clips if min_tempo <= clip.estimated_tempo <= max_tempo]


# Quality standards for validation
class QualityStandards:
    """Quality standards for move clip validation."""
    
    MIN_DURATION = 5.0  # seconds
    MAX_DURATION = 20.0  # seconds
    MIN_TEMPO = 80  # BPM
    MAX_TEMPO = 160  # BPM
    
    REQUIRED_QUALITY_CHECKS = [
        "full_body_visible",
        "good_lighting", 
        "clear_movement",
        "stable_camera"
    ]
    
    CATEGORY_REQUIREMENTS = {
        MoveCategory.BASIC_STEP: {
            "max_difficulty": DifficultyLevel.INTERMEDIATE,
            "required_elements": ["basic_step", "weight_transfer"]
        },
        MoveCategory.CROSS_BODY_LEAD: {
            "min_difficulty": DifficultyLevel.BEGINNER,
            "required_elements": ["lead_follow_connection"]
        },
        MoveCategory.LADY_RIGHT_TURN: {
            "min_difficulty": DifficultyLevel.INTERMEDIATE,
            "required_elements": ["rotation", "balance"]
        },
        MoveCategory.LADY_LEFT_TURN: {
            "min_difficulty": DifficultyLevel.INTERMEDIATE,
            "required_elements": ["rotation", "balance"]
        },
        MoveCategory.ARM_STYLING: {
            "min_difficulty": DifficultyLevel.BEGINNER,
            "required_elements": ["body_movement", "expression"]
        },
        MoveCategory.BODY_ROLL: {
            "min_difficulty": DifficultyLevel.ADVANCED,
            "required_elements": ["complex_coordination"]
        },
        MoveCategory.COMBINATION: {
            "min_difficulty": DifficultyLevel.ADVANCED,
            "required_elements": ["complex_coordination"]
        }
    }