"""
Video generation service for creating choreography videos.
Handles stitching together move clips using FFmpeg.
"""

import subprocess
import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import json

from ..models.video_models import (
    ChoreographySequence, 
    SelectedMove, 
    VideoGenerationConfig, 
    VideoGenerationResult,
    TransitionType
)

logger = logging.getLogger(__name__)


class VideoGenerationError(Exception):
    """Custom exception for video generation errors."""
    pass


class VideoGenerator:
    """
    Service for generating choreography videos by stitching together move clips.
    Uses FFmpeg for video processing and concatenation.
    """
    
    def __init__(self, config: Optional[VideoGenerationConfig] = None):
        """
        Initialize the video generator.
        
        Args:
            config: Configuration for video generation. Uses defaults if not provided.
        """
        self.config = config or VideoGenerationConfig(output_path="data/temp/output.mp4")
        self._ensure_temp_directory()
        self._check_ffmpeg_availability()
    
    def _ensure_temp_directory(self) -> None:
        """Ensure the temporary directory exists."""
        temp_dir = Path(self.config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_ffmpeg_availability(self) -> None:
        """Check if FFmpeg is available on the system."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                raise VideoGenerationError("FFmpeg is not available or not working properly")
            logger.info("FFmpeg is available and working")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise VideoGenerationError(f"FFmpeg is not installed or not in PATH: {e}")
    
    def export_sequence_metadata(
        self,
        sequence: ChoreographySequence,
        output_path: str,
        music_features: Optional[dict] = None,
        additional_info: Optional[dict] = None
    ) -> str:
        """
        Export detailed choreography sequence metadata to JSON.
        
        Args:
            sequence: The choreography sequence
            output_path: Path where the JSON metadata should be saved
            music_features: Optional music analysis features
            additional_info: Additional metadata to include
            
        Returns:
            Path to the saved JSON metadata file
        """
        try:
            # Create detailed metadata
            metadata = {
                "sequence_info": {
                    "total_moves": len(sequence.moves),
                    "total_duration": sequence.total_duration,
                    "difficulty_level": sequence.difficulty_level,
                    "audio_tempo": sequence.audio_tempo,
                    "generation_timestamp": sequence.generation_timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
                    "generation_parameters": sequence.generation_parameters or {}
                },
                "moves_used": [
                    {
                        "clip_id": move.clip_id,
                        "video_path": move.video_path,
                        "move_name": Path(move.video_path).stem,
                        "start_time": move.start_time,
                        "duration": move.duration,
                        "end_time": move.start_time + move.duration,
                        "transition_type": move.transition_type.value,
                        "original_duration": move.original_duration,
                        "trim_start": move.trim_start or 0,
                        "trim_end": move.trim_end or 0,
                        "volume_adjustment": move.volume_adjustment or 1.0
                    }
                    for move in sequence.moves
                ],
                "audio_info": {
                    "audio_path": sequence.audio_path,
                    "tempo": sequence.audio_tempo
                },
                "music_analysis": music_features or {},
                "video_config": {
                    "output_path": self.config.output_path,
                    "resolution": self.config.resolution,
                    "video_bitrate": self.config.video_bitrate,
                    "audio_bitrate": self.config.audio_bitrate,
                    "frame_rate": self.config.frame_rate
                }
            }
            
            # Add additional info if provided
            if additional_info:
                metadata.update(additional_info)
            
            # Save metadata to JSON file
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Sequence metadata exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export sequence metadata: {e}")
            return ""

    def generate_choreography_video(
        self, 
        sequence: ChoreographySequence,
        audio_path: Optional[str] = None,
        music_features: Optional[dict] = None
    ) -> VideoGenerationResult:
        """
        Generate a choreography video from a sequence of selected moves.
        
        Args:
            sequence: The choreography sequence with selected moves
            audio_path: Optional path to audio file to overlay
            music_features: Optional music analysis features for beat synchronization
            
        Returns:
            VideoGenerationResult with success status and output information
        """
        start_time = time.time()
        temp_files = []
        
        try:
            logger.info(f"Starting video generation with {len(sequence.moves)} moves")
            
            # Validate input moves
            self._validate_sequence(sequence)
            
            # Create file list for FFmpeg concatenation with beat synchronization
            if music_features and music_features.get('beat_positions'):
                concat_file_path = self._create_beat_synchronized_concat_file(
                    sequence, music_features, temp_files
                )
            else:
                concat_file_path = self._create_concat_file(sequence, temp_files)
            
            # Generate the video with enhanced audio synchronization
            output_path = self._concatenate_videos_with_audio_sync(
                concat_file_path, audio_path, music_features
            )
            
            # Get output file information
            duration, file_size = self._get_output_info(output_path)
            
            processing_time = time.time() - start_time
            
            # Clean up temporary files if requested
            if self.config.cleanup_temp_files:
                self._cleanup_temp_files(temp_files)
            
            logger.info(f"Video generation completed successfully in {processing_time:.2f}s")
            
            return VideoGenerationResult(
                success=True,
                output_path=output_path,
                duration=duration,
                file_size=file_size,
                processing_time=processing_time,
                clips_processed=len(sequence.moves),
                temp_files_created=temp_files
            )
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            
            # Clean up on error
            if self.config.cleanup_temp_files:
                self._cleanup_temp_files(temp_files)
            
            return VideoGenerationResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                temp_files_created=temp_files
            )
    
    def _validate_sequence(self, sequence: ChoreographySequence) -> None:
        """
        Validate the choreography sequence.
        
        Args:
            sequence: The sequence to validate
            
        Raises:
            VideoGenerationError: If validation fails
        """
        if not sequence.moves:
            raise VideoGenerationError("Sequence must contain at least one move")
        
        for i, move in enumerate(sequence.moves):
            if not os.path.exists(move.video_path):
                raise VideoGenerationError(f"Video file not found: {move.video_path}")
            
            if move.duration <= 0:
                raise VideoGenerationError(f"Move {i} has invalid duration: {move.duration}")
    
    def _create_concat_file(
        self, 
        sequence: ChoreographySequence, 
        temp_files: List[str]
    ) -> str:
        """
        Create a concatenation file for FFmpeg.
        
        Args:
            sequence: The choreography sequence
            temp_files: List to track temporary files
            
        Returns:
            Path to the concatenation file
        """
        concat_file_path = os.path.join(self.config.temp_dir, f"concat_{int(time.time())}.txt")
        temp_files.append(concat_file_path)
        
        with open(concat_file_path, 'w') as f:
            for move in sequence.moves:
                # Convert to absolute path for FFmpeg
                abs_path = os.path.abspath(move.video_path)
                f.write(f"file '{abs_path}'\n")
        
        logger.info(f"Created concatenation file: {concat_file_path}")
        return concat_file_path
    
    def _create_beat_synchronized_concat_file(
        self, 
        sequence: ChoreographySequence, 
        music_features: dict,
        temp_files: List[str]
    ) -> str:
        """
        Create a concatenation file with beat-synchronized timing adjustments.
        
        Args:
            sequence: The choreography sequence
            music_features: Music analysis features including beat positions
            temp_files: List to track temporary files
            
        Returns:
            Path to the concatenation file with timing adjustments
        """
        concat_file_path = os.path.join(self.config.temp_dir, f"beat_sync_concat_{int(time.time())}.txt")
        temp_files.append(concat_file_path)
        
        beat_positions = music_features.get('beat_positions', [])
        tempo = music_features.get('tempo', 120)
        
        logger.info(f"Creating beat-synchronized concat file with {len(beat_positions)} beats at {tempo} BPM")
        
        with open(concat_file_path, 'w') as f:
            for i, move in enumerate(sequence.moves):
                abs_path = os.path.abspath(move.video_path)
                
                # Calculate beat-aligned duration if we have beat information
                if beat_positions and i < len(beat_positions) - 1:
                    # Find the closest beat positions for this move's timing
                    move_start_beat = self._find_closest_beat(move.start_time, beat_positions)
                    move_end_time = move.start_time + move.duration
                    move_end_beat = self._find_closest_beat(move_end_time, beat_positions)
                    
                    # Calculate beat-aligned duration
                    if move_end_beat > move_start_beat:
                        beat_aligned_duration = beat_positions[move_end_beat] - beat_positions[move_start_beat]
                        
                        # Only adjust if the difference is reasonable (within 20% of original)
                        duration_diff = abs(beat_aligned_duration - move.duration)
                        if duration_diff / move.duration < 0.2:
                            logger.debug(f"Beat-aligning move {i}: {move.duration:.2f}s -> {beat_aligned_duration:.2f}s")
                            
                            # Write with duration filter for beat alignment
                            f.write(f"file '{abs_path}'\n")
                            f.write(f"duration {beat_aligned_duration:.3f}\n")
                            continue
                
                # Default: use original duration
                f.write(f"file '{abs_path}'\n")
        
        logger.info(f"Created beat-synchronized concatenation file: {concat_file_path}")
        return concat_file_path
    
    def _find_closest_beat(self, time_position: float, beat_positions: List[float]) -> int:
        """
        Find the index of the beat position closest to the given time.
        
        Args:
            time_position: Time position in seconds
            beat_positions: List of beat positions in seconds
            
        Returns:
            Index of the closest beat position
        """
        if not beat_positions:
            return 0
        
        min_distance = float('inf')
        closest_index = 0
        
        for i, beat_time in enumerate(beat_positions):
            distance = abs(beat_time - time_position)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index
    
    def _concatenate_videos_with_audio_sync(
        self, 
        concat_file_path: str, 
        audio_path: Optional[str] = None,
        music_features: Optional[dict] = None
    ) -> str:
        """
        Concatenate videos using FFmpeg with enhanced audio synchronization.
        
        Args:
            concat_file_path: Path to the concatenation file
            audio_path: Optional audio file to overlay
            music_features: Optional music analysis features for synchronization
            
        Returns:
            Path to the output video file
        """
        output_path = self.config.output_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build FFmpeg command for simple concatenation
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",  # Copy streams without re-encoding for speed
            "-y",  # Overwrite output file
            output_path
        ]
        
        # Enhanced audio overlay with synchronization and web optimization
        if audio_path and os.path.exists(audio_path) and self.config.add_audio_overlay:
            cmd = self._build_audio_sync_command(
                concat_file_path, audio_path, output_path, music_features
            )
        else:
            # Optimize for web playback even without audio overlay
            cmd = self._build_web_optimized_command(concat_file_path, output_path)
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"FFmpeg failed with return code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"
                raise VideoGenerationError(error_msg)
            
            logger.info("Video concatenation completed successfully")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise VideoGenerationError("FFmpeg process timed out")
        except Exception as e:
            raise VideoGenerationError(f"FFmpeg execution failed: {e}")
    
    def _get_output_info(self, output_path: str) -> Tuple[Optional[float], Optional[int]]:
        """
        Get information about the generated video file.
        
        Args:
            output_path: Path to the output video
            
        Returns:
            Tuple of (duration in seconds, file size in bytes)
        """
        try:
            # Get file size
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else None
            
            # Get duration using ffprobe
            duration = None
            try:
                cmd = [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    info = json.loads(result.stdout)
                    duration = float(info.get("format", {}).get("duration", 0))
            except Exception as e:
                logger.warning(f"Could not get video duration: {e}")
            
            return duration, file_size
            
        except Exception as e:
            logger.warning(f"Could not get output file info: {e}")
            return None, None
    
    def _cleanup_temp_files(self, temp_files: List[str]) -> None:
        """
        Clean up temporary files.
        
        Args:
            temp_files: List of temporary file paths to clean up
        """
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary file {temp_file}: {e}")
    
    def _build_audio_sync_command(
        self, 
        concat_file_path: str, 
        audio_path: str, 
        output_path: str,
        music_features: Optional[dict] = None
    ) -> List[str]:
        """
        Build FFmpeg command for audio-synchronized video generation.
        
        Args:
            concat_file_path: Path to concatenation file
            audio_path: Path to audio file
            output_path: Output video path
            music_features: Optional music features for advanced sync
            
        Returns:
            FFmpeg command as list of strings
        """
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-i", audio_path,
            
            # Video encoding optimized for web
            "-c:v", self.config.video_codec,
            "-preset", "medium",  # Balance between speed and compression
            "-crf", "23",  # Constant rate factor for good quality
            "-maxrate", self.config.video_bitrate,
            "-bufsize", "4M",  # Buffer size for rate control
            
            # Audio encoding
            "-c:a", self.config.audio_codec,
            "-b:a", self.config.audio_bitrate,
            "-ar", "44100",  # Standard sample rate
            
            # Stream mapping
            "-map", "0:v:0",  # Video from concatenated clips
            "-map", "1:a:0",  # Audio from audio file
            
            # Synchronization and timing
            "-vsync", "cfr",  # Constant frame rate for web compatibility
            "-r", str(self.config.frame_rate),  # Explicit frame rate
            
            # Web optimization
            "-movflags", "+faststart",  # Enable progressive download
            "-pix_fmt", "yuv420p",  # Ensure compatibility with all browsers
            
            # Duration handling
            "-shortest",  # End when shortest stream ends
            
            # Audio processing
        ]
        
        # Add audio normalization if enabled
        if self.config.normalize_audio:
            cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
        
        # Add resolution if specified
        if self.config.resolution:
            cmd.extend(["-s", self.config.resolution])
        
        # Beat synchronization adjustments
        if music_features and music_features.get('tempo'):
            tempo = music_features['tempo']
            # Add subtle tempo-based frame rate adjustment for better sync
            if 90 <= tempo <= 150:  # Typical Bachata range
                # Slight frame rate adjustment based on tempo
                adjusted_fps = self.config.frame_rate * (tempo / 120.0) * 0.02 + self.config.frame_rate * 0.98
                cmd[cmd.index(str(self.config.frame_rate))] = f"{adjusted_fps:.2f}"
        
        cmd.extend(["-y", output_path])
        return cmd
    
    def _build_web_optimized_command(self, concat_file_path: str, output_path: str) -> List[str]:
        """
        Build FFmpeg command optimized for web playback without audio overlay.
        
        Args:
            concat_file_path: Path to concatenation file
            output_path: Output video path
            
        Returns:
            FFmpeg command as list of strings
        """
        return [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            
            # Web-optimized video encoding
            "-c:v", self.config.video_codec,
            "-preset", "medium",
            "-crf", "23",
            "-maxrate", self.config.video_bitrate,
            "-bufsize", "4M",
            
            # Web compatibility
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-vsync", "cfr",
            "-r", str(self.config.frame_rate),
            
            # Copy audio from original clips if present
            "-c:a", self.config.audio_codec,
            "-b:a", self.config.audio_bitrate,
            
            "-y", output_path
        ]
    
    def _concatenate_videos(
        self, 
        concat_file_path: str, 
        audio_path: Optional[str] = None
    ) -> str:
        """
        Legacy method for backward compatibility.
        Delegates to the enhanced audio sync method.
        """
        return self._concatenate_videos_with_audio_sync(concat_file_path, audio_path, None)
    
    def create_simple_sequence_from_paths(
        self, 
        video_paths: List[str],
        output_path: Optional[str] = None
    ) -> ChoreographySequence:
        """
        Create a simple choreography sequence from a list of video paths.
        Useful for testing basic concatenation functionality.
        
        Args:
            video_paths: List of paths to video files
            output_path: Optional output path for the generated video
            
        Returns:
            ChoreographySequence ready for video generation
        """
        if output_path:
            self.config.output_path = output_path
        
        moves = []
        current_time = 0.0
        
        for i, video_path in enumerate(video_paths):
            if not os.path.exists(video_path):
                raise VideoGenerationError(f"Video file not found: {video_path}")
            
            # For now, assume each clip is 10 seconds (will be improved with actual duration detection)
            duration = 10.0
            
            move = SelectedMove(
                clip_id=f"clip_{i}",
                video_path=video_path,
                start_time=current_time,
                duration=duration,
                transition_type=TransitionType.CUT
            )
            
            moves.append(move)
            current_time += duration
        
        return ChoreographySequence(
            moves=moves,
            total_duration=current_time,
            difficulty_level="mixed"
        )
    
    def test_basic_concatenation(self, test_video_paths: List[str]) -> VideoGenerationResult:
        """
        Test basic video concatenation functionality.
        
        Args:
            test_video_paths: List of video file paths to concatenate
            
        Returns:
            VideoGenerationResult with test results
        """
        logger.info(f"Testing basic concatenation with {len(test_video_paths)} videos")
        
        # Create a simple sequence
        sequence = self.create_simple_sequence_from_paths(test_video_paths)
        
        # Generate the video
        result = self.generate_choreography_video(sequence)
        
        if result.success:
            logger.info(f"Basic concatenation test successful. Output: {result.output_path}")
        else:
            logger.error(f"Basic concatenation test failed: {result.error_message}")
        
        return result
    
    def create_beat_synchronized_sequence(
        self,
        video_paths: List[str],
        music_features: dict,
        target_duration: Optional[float] = None
    ) -> ChoreographySequence:
        """
        Create a choreography sequence synchronized to musical beats.
        
        Args:
            video_paths: List of video file paths
            music_features: Music analysis features including beat positions and tempo
            target_duration: Optional target duration for the sequence
            
        Returns:
            ChoreographySequence with beat-synchronized timing
        """
        beat_positions = music_features.get('beat_positions', [])
        tempo = music_features.get('tempo', 120)
        audio_duration = music_features.get('duration', 0)
        
        if not beat_positions:
            logger.warning("No beat positions available, falling back to simple sequence")
            return self.create_simple_sequence_from_paths(video_paths)
        
        logger.info(f"Creating beat-synchronized sequence with {len(beat_positions)} beats at {tempo} BPM")
        
        moves = []
        current_beat_index = 0
        
        # Calculate beats per move based on typical Bachata patterns
        # Most moves span 2-4 beats (4-8 counts in dance terminology)
        beats_per_move = 4  # Default to 4 beats per move
        
        for i, video_path in enumerate(video_paths):
            if not os.path.exists(video_path):
                raise VideoGenerationError(f"Video file not found: {video_path}")
            
            # Calculate start time from beat position
            if current_beat_index < len(beat_positions):
                start_time = beat_positions[current_beat_index]
            else:
                # If we run out of beats, continue with estimated timing
                beat_interval = 60.0 / tempo  # seconds per beat
                start_time = beat_positions[-1] + (current_beat_index - len(beat_positions) + 1) * beat_interval
            
            # Calculate duration based on beats
            end_beat_index = min(current_beat_index + beats_per_move, len(beat_positions) - 1)
            
            if end_beat_index > current_beat_index and end_beat_index < len(beat_positions):
                duration = beat_positions[end_beat_index] - start_time
            else:
                # Fallback to tempo-based duration
                duration = beats_per_move * (60.0 / tempo)
            
            # Ensure reasonable duration bounds
            duration = max(6.0, min(duration, 15.0))  # 6-15 seconds per move
            
            move = SelectedMove(
                clip_id=f"beat_sync_move_{i+1}",
                video_path=video_path,
                start_time=start_time,
                duration=duration,
                transition_type=TransitionType.CUT
            )
            
            moves.append(move)
            current_beat_index += beats_per_move
            
            logger.debug(f"Beat-sync move {i+1}: start={start_time:.2f}s, duration={duration:.2f}s")
            
            # Stop if we've reached the target duration or end of audio
            if target_duration and start_time + duration >= target_duration:
                break
            if audio_duration and start_time + duration >= audio_duration:
                break
        
        total_duration = moves[-1].start_time + moves[-1].duration if moves else 0.0
        
        sequence = ChoreographySequence(
            moves=moves,
            total_duration=total_duration,
            difficulty_level="mixed",
            audio_tempo=tempo,
            generation_parameters={
                "sync_type": "beat_synchronized",
                "tempo": tempo,
                "beats_per_move": beats_per_move,
                "total_beats_used": current_beat_index
            }
        )
        
        logger.info(f"Created beat-synchronized sequence: {len(moves)} moves, {total_duration:.2f}s total")
        return sequence