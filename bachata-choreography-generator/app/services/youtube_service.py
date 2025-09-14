"""
YouTube audio download service using yt-dlp.
Handles URL validation and audio extraction to MP3 format.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional
import yt_dlp
from pydantic import BaseModel
from tqdm import tqdm


class AudioDownloadResult(BaseModel):
    """Result of audio download operation."""
    success: bool
    file_path: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None


class YouTubeService:
    """Service for downloading and processing YouTube audio."""
    
    def __init__(self, output_dir: str = "data/temp"):
        """
        Initialize YouTube service.
        
        Args:
            output_dir: Directory to store downloaded audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_bar = None
        
        # yt-dlp configuration for audio extraction
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [self._progress_hook],
        }
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if the provided URL is a valid YouTube URL.
        
        Args:
            url: YouTube URL to validate
            
        Returns:
            True if valid YouTube URL, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
            
        # YouTube URL patterns
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+',
        ]
        
        return any(re.match(pattern, url.strip()) for pattern in youtube_patterns)
    
    def _progress_hook(self, d):
        """
        Progress hook for yt-dlp to show download progress with tqdm.
        
        Args:
            d: Dictionary containing download progress information
        """
        if d['status'] == 'downloading':
            if self.progress_bar is None:
                # Initialize progress bar
                total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                if total_bytes > 0:
                    self.progress_bar = tqdm(
                        total=total_bytes,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc="Downloading audio",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                    )
            
            if self.progress_bar is not None:
                downloaded = d.get('downloaded_bytes', 0)
                if hasattr(self.progress_bar, 'last_downloaded'):
                    # Update with the difference since last update
                    self.progress_bar.update(downloaded - self.progress_bar.last_downloaded)
                else:
                    # First update
                    self.progress_bar.update(downloaded)
                self.progress_bar.last_downloaded = downloaded
                
        elif d['status'] == 'finished':
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None
                print("✅ Download completed, converting to MP3...")
        
        elif d['status'] == 'error':
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID if found, None otherwise
        """
        patterns = [
            r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed/|v/)([0-9A-Za-z_-]{11})',
            r'youtu\.be/([0-9A-Za-z_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def download_audio(self, url: str) -> AudioDownloadResult:
        """
        Download audio from YouTube URL and convert to MP3.
        
        Args:
            url: YouTube URL to download audio from
            
        Returns:
            AudioDownloadResult with success status and file path or error message
        """
        # Validate URL first
        if not self.validate_url(url):
            return AudioDownloadResult(
                success=False,
                error_message="Invalid YouTube URL format"
            )
        
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    return AudioDownloadResult(
                        success=False,
                        error_message="Could not extract video information"
                    )
                
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                # Check duration limits (30 seconds to 8 minutes as per design)
                if duration < 30:
                    return AudioDownloadResult(
                        success=False,
                        error_message="Video too short (minimum 30 seconds required)"
                    )
                
                if duration > 480:  # 8 minutes
                    return AudioDownloadResult(
                        success=False,
                        error_message="Video too long (maximum 8 minutes allowed)"
                    )
                
                # Reset progress bar for new download
                self.progress_bar = None
                
                # Download and extract audio
                ydl.download([url])
                
                # Ensure progress bar is closed
                if self.progress_bar is not None:
                    self.progress_bar.close()
                    self.progress_bar = None
                
                # Find the downloaded file
                video_id = self.extract_video_id(url)
                downloaded_files = list(self.output_dir.glob("*.mp3"))
                
                # Find the most recently created file (should be our download)
                if downloaded_files:
                    latest_file = max(downloaded_files, key=os.path.getctime)
                    
                    return AudioDownloadResult(
                        success=True,
                        file_path=str(latest_file),
                        title=title,
                        duration=duration
                    )
                else:
                    return AudioDownloadResult(
                        success=False,
                        error_message="Audio file not found after download"
                    )
                    
        except yt_dlp.DownloadError as e:
            return AudioDownloadResult(
                success=False,
                error_message=f"Download failed: {str(e)}"
            )
        except Exception as e:
            return AudioDownloadResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def cleanup_temp_files(self) -> None:
        """Remove all temporary audio files."""
        try:
            for file_path in self.output_dir.glob("*.mp3"):
                file_path.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")