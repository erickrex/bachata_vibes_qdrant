"""
Data models for video generation and choreography sequences.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class TransitionType(str, Enum):
    """Types of transitions between video clips."""
    CUT = "cut"
    CROSSFADE = "crossfade"
    FADE_BLACK = "fade_black"


class SelectedMove(BaseModel):
    """Represents a selected move clip for choreography generation."""
    
    clip_id: str = Field(..., description="Unique identifier for the move clip")
    video_path: str = Field(..., description="Path to the video file")
    start_time: float = Field(..., ge=0, description="Start time in final video (seconds)")
    duration: float = Field(..., gt=0, description="Duration of the clip (seconds)")
    transition_type: TransitionType = Field(default=TransitionType.CUT, description="Transition type to next clip")
    
    # Optional metadata for advanced processing
    original_duration: Optional[float] = Field(None, description="Original clip duration before trimming")
    trim_start: Optional[float] = Field(0, description="Seconds to trim from start of original clip")
    trim_end: Optional[float] = Field(0, description="Seconds to trim from end of original clip")
    volume_adjustment: Optional[float] = Field(1.0, description="Volume adjustment factor (0.0-2.0)")


class ChoreographySequence(BaseModel):
    """Complete choreography sequence with selected moves."""
    
    moves: List[SelectedMove] = Field(..., description="List of selected moves in sequence")
    total_duration: float = Field(..., gt=0, description="Total duration of choreography (seconds)")
    difficulty_level: str = Field(..., description="Overall difficulty level")
    
    # Audio information
    audio_path: Optional[str] = Field(None, description="Path to the audio file")
    audio_tempo: Optional[float] = Field(None, description="Detected tempo of the audio (BPM)")
    
    # Generation metadata
    generation_timestamp: Optional[str] = Field(None, description="When the sequence was generated")
    generation_parameters: Optional[dict] = Field(default_factory=dict, description="Parameters used for generation")
    
    @property
    def move_count(self) -> int:
        """Number of moves in the sequence."""
        return len(self.moves)
    
    def get_moves_by_transition_type(self, transition_type: TransitionType) -> List[SelectedMove]:
        """Get moves that use a specific transition type."""
        return [move for move in self.moves if move.transition_type == transition_type]


class VideoGenerationConfig(BaseModel):
    """Configuration for video generation process."""
    
    # Output settings
    output_path: str = Field(..., description="Path for the generated video file")
    output_format: str = Field(default="mp4", description="Output video format")
    video_codec: str = Field(default="libx264", description="Video codec to use")
    audio_codec: str = Field(default="aac", description="Audio codec to use")
    
    # Quality settings
    video_bitrate: str = Field(default="2M", description="Video bitrate")
    audio_bitrate: str = Field(default="128k", description="Audio bitrate")
    frame_rate: int = Field(default=30, description="Output frame rate")
    resolution: Optional[str] = Field(None, description="Output resolution (e.g., '1920x1080')")
    
    # Processing settings
    transition_duration: float = Field(default=0.5, description="Duration of transitions in seconds")
    fade_duration: float = Field(default=0.3, description="Duration of fade effects in seconds")
    temp_dir: str = Field(default="data/temp", description="Directory for temporary files")
    
    # Advanced settings
    preserve_aspect_ratio: bool = Field(default=True, description="Preserve original aspect ratio")
    add_audio_overlay: bool = Field(default=True, description="Add original audio track")
    normalize_audio: bool = Field(default=True, description="Normalize audio levels")
    cleanup_temp_files: bool = Field(default=True, description="Clean up temporary files after generation")


class VideoGenerationResult(BaseModel):
    """Result of video generation process."""
    
    success: bool = Field(..., description="Whether generation was successful")
    output_path: Optional[str] = Field(None, description="Path to generated video file")
    duration: Optional[float] = Field(None, description="Duration of generated video")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    
    # Processing information
    processing_time: Optional[float] = Field(None, description="Time taken to generate video (seconds)")
    clips_processed: Optional[int] = Field(None, description="Number of clips processed")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages during generation")
    
    # Technical details
    ffmpeg_command: Optional[str] = Field(None, description="FFmpeg command used for generation")
    temp_files_created: List[str] = Field(default_factory=list, description="Temporary files created during processing")