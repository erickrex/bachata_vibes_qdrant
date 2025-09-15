"""
Tests for the VideoGenerator service.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.video_generator import VideoGenerator, VideoGenerationError
from app.models.video_models import (
    ChoreographySequence, 
    SelectedMove, 
    VideoGenerationConfig,
    TransitionType
)


class TestVideoGenerator:
    """Test cases for VideoGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VideoGenerationConfig(
            output_path=os.path.join(self.temp_dir, "test_output.mp4"),
            temp_dir=self.temp_dir,
            cleanup_temp_files=False  # Keep files for inspection during tests
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_ffmpeg_availability_check_success(self, mock_run):
        """Test successful FFmpeg availability check."""
        mock_run.return_value = MagicMock(returncode=0)
        
        # Should not raise an exception
        generator = VideoGenerator(self.config)
        assert generator is not None
    
    @patch('subprocess.run')
    def test_ffmpeg_availability_check_failure(self, mock_run):
        """Test FFmpeg availability check failure."""
        mock_run.side_effect = FileNotFoundError("ffmpeg not found")
        
        with pytest.raises(VideoGenerationError, match="FFmpeg is not installed"):
            VideoGenerator(self.config)
    
    @patch('subprocess.run')
    def test_create_simple_sequence_from_paths(self, mock_run):
        """Test creating a simple sequence from video paths."""
        mock_run.return_value = MagicMock(returncode=0)
        
        generator = VideoGenerator(self.config)
        
        # Create some dummy video files for testing
        video_paths = []
        for i in range(3):
            video_path = os.path.join(self.temp_dir, f"test_video_{i}.mp4")
            Path(video_path).touch()  # Create empty file
            video_paths.append(video_path)
        
        sequence = generator.create_simple_sequence_from_paths(video_paths)
        
        assert len(sequence.moves) == 3
        assert sequence.total_duration == 30.0  # 3 clips * 10 seconds each
        assert sequence.difficulty_level == "mixed"
        
        for i, move in enumerate(sequence.moves):
            assert move.clip_id == f"clip_{i}"
            assert move.video_path == video_paths[i]
            assert move.duration == 10.0
            assert move.transition_type == TransitionType.CUT
    
    @patch('subprocess.run')
    def test_create_simple_sequence_missing_file(self, mock_run):
        """Test creating sequence with missing video file."""
        mock_run.return_value = MagicMock(returncode=0)
        
        generator = VideoGenerator(self.config)
        
        video_paths = ["/nonexistent/video.mp4"]
        
        with pytest.raises(VideoGenerationError, match="Video file not found"):
            generator.create_simple_sequence_from_paths(video_paths)
    
    @patch('subprocess.run')
    def test_validate_sequence_success(self, mock_run):
        """Test successful sequence validation."""
        mock_run.return_value = MagicMock(returncode=0)
        
        generator = VideoGenerator(self.config)
        
        # Create test video file
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        Path(video_path).touch()
        
        move = SelectedMove(
            clip_id="test_clip",
            video_path=video_path,
            start_time=0.0,
            duration=10.0
        )
        
        sequence = ChoreographySequence(
            moves=[move],
            total_duration=10.0,
            difficulty_level="beginner"
        )
        
        # Should not raise an exception
        generator._validate_sequence(sequence)
    
    @patch('subprocess.run')
    def test_validate_sequence_empty_moves(self, mock_run):
        """Test validation with empty moves list."""
        mock_run.return_value = MagicMock(returncode=0)
        
        generator = VideoGenerator(self.config)
        
        # Create sequence with empty moves but valid total_duration to pass Pydantic validation
        sequence = ChoreographySequence(
            moves=[],
            total_duration=1.0,  # Must be > 0 for Pydantic validation
            difficulty_level="beginner"
        )
        
        with pytest.raises(VideoGenerationError, match="Sequence must contain at least one move"):
            generator._validate_sequence(sequence)
    
    @patch('subprocess.run')
    def test_validate_sequence_missing_video_file(self, mock_run):
        """Test validation with missing video file."""
        mock_run.return_value = MagicMock(returncode=0)
        
        generator = VideoGenerator(self.config)
        
        move = SelectedMove(
            clip_id="test_clip",
            video_path="/nonexistent/video.mp4",
            start_time=0.0,
            duration=10.0
        )
        
        sequence = ChoreographySequence(
            moves=[move],
            total_duration=10.0,
            difficulty_level="beginner"
        )
        
        with pytest.raises(VideoGenerationError, match="Video file not found"):
            generator._validate_sequence(sequence)
    
    @patch('subprocess.run')
    def test_create_concat_file(self, mock_run):
        """Test creation of FFmpeg concatenation file."""
        mock_run.return_value = MagicMock(returncode=0)
        
        generator = VideoGenerator(self.config)
        
        # Create test video files
        video_paths = []
        for i in range(2):
            video_path = os.path.join(self.temp_dir, f"test_video_{i}.mp4")
            Path(video_path).touch()
            video_paths.append(video_path)
        
        moves = [
            SelectedMove(
                clip_id=f"clip_{i}",
                video_path=path,
                start_time=i * 10.0,
                duration=10.0
            )
            for i, path in enumerate(video_paths)
        ]
        
        sequence = ChoreographySequence(
            moves=moves,
            total_duration=20.0,
            difficulty_level="beginner"
        )
        
        temp_files = []
        concat_file_path = generator._create_concat_file(sequence, temp_files)
        
        assert os.path.exists(concat_file_path)
        assert concat_file_path in temp_files
        
        # Check file content
        with open(concat_file_path, 'r') as f:
            content = f.read()
            for video_path in video_paths:
                abs_path = os.path.abspath(video_path)
                assert f"file '{abs_path}'" in content
    
    @patch('subprocess.run')
    def test_get_output_info_success(self, mock_run):
        """Test getting output file information."""
        # Mock ffmpeg availability check
        mock_run.return_value = MagicMock(returncode=0)
        
        generator = VideoGenerator(self.config)
        
        # Create a test output file
        output_path = os.path.join(self.temp_dir, "test_output.mp4")
        with open(output_path, 'w') as f:
            f.write("dummy video content")
        
        # Mock ffprobe response with proper string format
        mock_ffprobe_response = MagicMock(returncode=0)
        mock_ffprobe_response.stdout = '{"format": {"duration": "15.5"}}'
        
        # We need to patch subprocess.run specifically for the ffprobe call
        with patch('app.services.video_generator.subprocess.run', side_effect=[
            mock_ffprobe_response     # ffprobe call
        ]):
            duration, file_size = generator._get_output_info(output_path)
        
        assert duration == 15.5
        assert file_size > 0
    
    @patch('subprocess.run')
    def test_cleanup_temp_files(self, mock_run):
        """Test cleanup of temporary files."""
        mock_run.return_value = MagicMock(returncode=0)
        
        generator = VideoGenerator(self.config)
        
        # Create temporary files
        temp_files = []
        for i in range(3):
            temp_file = os.path.join(self.temp_dir, f"temp_file_{i}.txt")
            with open(temp_file, 'w') as f:
                f.write("temporary content")
            temp_files.append(temp_file)
        
        # Verify files exist
        for temp_file in temp_files:
            assert os.path.exists(temp_file)
        
        # Clean up
        generator._cleanup_temp_files(temp_files)
        
        # Verify files are deleted
        for temp_file in temp_files:
            assert not os.path.exists(temp_file)


class TestVideoGeneratorIntegration:
    """Integration tests for VideoGenerator (require actual video files)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VideoGenerationConfig(
            output_path=os.path.join(self.temp_dir, "integration_test_output.mp4"),
            temp_dir=self.temp_dir,
            cleanup_temp_files=False
        )
        
        # Path to actual video files for integration testing
        self.video_data_dir = Path("data/Bachata_steps")
        
        # Sample music features for testing beat synchronization
        self.sample_music_features = {
            'tempo': 120.0,
            'beat_positions': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 
                             5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
            'duration': 30.0
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_integration_with_real_videos(self):
        """Integration test with real video files (if available)."""
        # Skip if video files are not available
        if not self.video_data_dir.exists():
            pytest.skip("Video data directory not available for integration testing")
        
        # Find some actual video files
        video_files = []
        for category_dir in self.video_data_dir.iterdir():
            if category_dir.is_dir():
                for video_file in category_dir.glob("*.mp4"):
                    video_files.append(str(video_file))
                    if len(video_files) >= 3:  # Use first 3 videos found
                        break
                if len(video_files) >= 3:
                    break
        
        if len(video_files) < 2:
            pytest.skip("Not enough video files available for integration testing")
        
        try:
            generator = VideoGenerator(self.config)
            
            # Test basic concatenation
            result = generator.test_basic_concatenation(video_files[:2])
            
            if result.success:
                assert os.path.exists(result.output_path)
                assert result.duration is not None
                assert result.file_size is not None
                assert result.clips_processed == 2
                print(f"Integration test successful: {result.output_path}")
            else:
                # If FFmpeg is not available, that's expected in some environments
                if "FFmpeg" in str(result.error_message):
                    pytest.skip(f"FFmpeg not available: {result.error_message}")
                else:
                    pytest.fail(f"Integration test failed: {result.error_message}")
                    
        except VideoGenerationError as e:
            if "FFmpeg" in str(e):
                pytest.skip(f"FFmpeg not available: {e}")
            else:
                raise
    
    def test_beat_synchronized_sequence_creation(self):
        """Test creation of beat-synchronized choreography sequence."""
        # Skip if video files are not available
        if not self.video_data_dir.exists():
            pytest.skip("Video data directory not available for testing")
        
        # Find some actual video files
        video_files = []
        for category_dir in self.video_data_dir.iterdir():
            if category_dir.is_dir():
                for video_file in category_dir.glob("*.mp4"):
                    video_files.append(str(video_file))
                    if len(video_files) >= 3:
                        break
                if len(video_files) >= 3:
                    break
        
        if len(video_files) < 2:
            pytest.skip("Not enough video files available for testing")
        
        try:
            generator = VideoGenerator(self.config)
            
            # Test beat-synchronized sequence creation
            sequence = generator.create_beat_synchronized_sequence(
                video_files[:2], 
                self.sample_music_features,
                target_duration=10.0
            )
            
            assert len(sequence.moves) == 2
            assert sequence.audio_tempo == 120.0
            assert sequence.generation_parameters['sync_type'] == 'beat_synchronized'
            
            # Check that moves are aligned to beat positions
            for move in sequence.moves:
                # Start times should be close to beat positions
                closest_beat_distance = min(
                    abs(move.start_time - beat) 
                    for beat in self.sample_music_features['beat_positions']
                )
                assert closest_beat_distance < 0.5  # Within 0.5 seconds of a beat
            
            print(f"Beat-synchronized sequence test successful with {len(sequence.moves)} moves")
            
        except VideoGenerationError as e:
            if "FFmpeg" in str(e):
                pytest.skip(f"FFmpeg not available: {e}")
            else:
                raise
    
    def test_audio_synchronization_with_music_features(self):
        """Test video generation with audio synchronization using music features."""
        # Skip if video files are not available
        if not self.video_data_dir.exists():
            pytest.skip("Video data directory not available for testing")
        
        # Check if audio files are available
        audio_dir = Path("data/songs")
        audio_files = list(audio_dir.glob("*.mp3")) if audio_dir.exists() else []
        
        if not audio_files:
            pytest.skip("No audio files available for audio sync testing")
        
        # Find video files
        video_files = []
        for category_dir in self.video_data_dir.iterdir():
            if category_dir.is_dir():
                for video_file in category_dir.glob("*.mp4"):
                    video_files.append(str(video_file))
                    if len(video_files) >= 2:
                        break
                if len(video_files) >= 2:
                    break
        
        if len(video_files) < 2:
            pytest.skip("Not enough video files available for testing")
        
        try:
            generator = VideoGenerator(self.config)
            
            # Create beat-synchronized sequence
            sequence = generator.create_beat_synchronized_sequence(
                video_files[:2], 
                self.sample_music_features,
                target_duration=10.0
            )
            
            # Generate video with audio synchronization
            audio_path = str(audio_files[0])
            result = generator.generate_choreography_video(
                sequence, 
                audio_path=audio_path,
                music_features=self.sample_music_features
            )
            
            if result.success:
                assert os.path.exists(result.output_path)
                assert result.duration is not None
                print(f"Audio synchronization test successful: {result.output_path}")
                print(f"Duration: {result.duration:.2f}s, Processing time: {result.processing_time:.2f}s")
                return True
            else:
                if "FFmpeg" in str(result.error_message):
                    pytest.skip(f"FFmpeg not available: {result.error_message}")
                else:
                    pytest.fail(f"Audio synchronization test failed: {result.error_message}")
                    
        except VideoGenerationError as e:
            if "FFmpeg" in str(e):
                pytest.skip(f"FFmpeg not available: {e}")
            else:
                raise