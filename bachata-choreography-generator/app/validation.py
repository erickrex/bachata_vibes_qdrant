"""
Input validation utilities for the Bachata Choreography Generator.
"""
import re
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs
import requests
from pydantic import BaseModel, validator, Field
from app.exceptions import ValidationError, get_user_friendly_message

class YouTubeURLValidator:
    """Validator for YouTube URLs with comprehensive checks."""
    
    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    ]
    
    @classmethod
    def validate_format(cls, url: str) -> bool:
        """Validate YouTube URL format."""
        if not url or not isinstance(url, str):
            return False
        
        url = url.strip()
        return any(re.match(pattern, url) for pattern in cls.YOUTUBE_PATTERNS)
    
    @classmethod
    def extract_video_id(cls, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        if not url:
            return None
        
        url = url.strip()
        
        for pattern in cls.YOUTUBE_PATTERNS:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    @classmethod
    def validate_accessibility(cls, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Validate that the YouTube video is accessible.
        Returns validation result with details.
        """
        result = {
            "valid": False,
            "error_type": None,
            "message": None,
            "details": {}
        }
        
        try:
            # Basic format validation
            if not cls.validate_format(url):
                result["error_type"] = "invalid_url_format"
                result["message"] = get_user_friendly_message("VALIDATION_ERROR", "invalid_url_format")
                return result
            
            # Extract video ID
            video_id = cls.extract_video_id(url)
            if not video_id:
                result["error_type"] = "invalid_url_format"
                result["message"] = get_user_friendly_message("VALIDATION_ERROR", "invalid_url_format")
                return result
            
            # Check if video exists (basic check)
            check_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            
            try:
                response = requests.get(check_url, timeout=timeout)
                if response.status_code == 200:
                    video_info = response.json()
                    result["valid"] = True
                    result["details"] = {
                        "video_id": video_id,
                        "title": video_info.get("title", "Unknown"),
                        "author": video_info.get("author_name", "Unknown")
                    }
                elif response.status_code == 404:
                    result["error_type"] = "video_unavailable"
                    result["message"] = get_user_friendly_message("YOUTUBE_DOWNLOAD_ERROR", "video_unavailable")
                else:
                    result["error_type"] = "network_error"
                    result["message"] = get_user_friendly_message("YOUTUBE_DOWNLOAD_ERROR", "network_error")
            
            except requests.RequestException:
                # If we can't check accessibility, assume it's valid (will be caught later)
                result["valid"] = True
                result["details"] = {"video_id": video_id}
        
        except Exception as e:
            result["error_type"] = "validation_error"
            result["message"] = f"URL validation failed: {str(e)}"
        
        return result

class ChoreographyRequestValidator(BaseModel):
    """Enhanced request validator with comprehensive validation."""
    
    youtube_url: str = Field(..., description="YouTube URL or local file path for the song")
    difficulty: str = Field(default="intermediate", description="Difficulty level")
    energy_level: Optional[str] = Field(default=None, description="Target energy level")
    quality_mode: str = Field(default="balanced", description="Quality mode")
    
    # Additional optional parameters
    max_duration: Optional[int] = Field(default=600, description="Maximum song duration in seconds")
    min_duration: Optional[int] = Field(default=30, description="Minimum song duration in seconds")
    
    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        """Validate YouTube URL or local file path."""
        if not v:
            raise ValidationError(
                message=get_user_friendly_message("VALIDATION_ERROR", "missing_required_field"),
                field="youtube_url",
                value=v
            )
        
        v = v.strip()
        
        # Check if it's a local file path
        if v.startswith('data/songs/') and v.endswith('.mp3'):
            # Validate local file exists
            file_path = Path(v)
            if not file_path.exists():
                raise ValidationError(
                    message=f"Local song file not found: {v}",
                    field="youtube_url",
                    value=v
                )
            return v
        
        # Otherwise, validate as YouTube URL
        if not YouTubeURLValidator.validate_format(v):
            raise ValidationError(
                message=get_user_friendly_message("VALIDATION_ERROR", "invalid_url_format"),
                field="youtube_url",
                value=v
            )
        
        return v
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        """Validate difficulty level."""
        valid_difficulties = ['beginner', 'intermediate', 'advanced']
        if v not in valid_difficulties:
            raise ValidationError(
                message=get_user_friendly_message("VALIDATION_ERROR", "invalid_difficulty"),
                field="difficulty",
                value=v,
                details={"valid_options": valid_difficulties}
            )
        return v
    
    @validator('energy_level')
    def validate_energy_level(cls, v):
        """Validate energy level."""
        if v is None:
            return v
        
        valid_energy_levels = ['low', 'medium', 'high']
        if v not in valid_energy_levels:
            raise ValidationError(
                message="Energy level must be 'low', 'medium', 'high', or null for auto-detection",
                field="energy_level",
                value=v,
                details={"valid_options": valid_energy_levels}
            )
        return v
    
    @validator('quality_mode')
    def validate_quality_mode(cls, v):
        """Validate quality mode."""
        valid_modes = ['fast', 'balanced', 'high_quality']
        if v not in valid_modes:
            raise ValidationError(
                message=get_user_friendly_message("VALIDATION_ERROR", "invalid_quality"),
                field="quality_mode",
                value=v,
                details={"valid_options": valid_modes}
            )
        return v
    
    @validator('max_duration')
    def validate_max_duration(cls, v):
        """Validate maximum duration."""
        if v is not None and (v < 30 or v > 1200):  # 30 seconds to 20 minutes
            raise ValidationError(
                message="Maximum duration must be between 30 and 1200 seconds",
                field="max_duration",
                value=v
            )
        return v
    
    @validator('min_duration')
    def validate_min_duration(cls, v):
        """Validate minimum duration."""
        if v is not None and (v < 10 or v > 300):  # 10 seconds to 5 minutes
            raise ValidationError(
                message="Minimum duration must be between 10 and 300 seconds",
                field="min_duration",
                value=v
            )
        return v

class SystemResourceValidator:
    """Validator for system resources and requirements."""
    
    @staticmethod
    def check_disk_space(required_mb: int = 500) -> Dict[str, Any]:
        """Check available disk space."""
        result = {"sufficient": False, "available_mb": 0, "required_mb": required_mb}
        
        try:
            # Check disk space in the output directory
            output_dir = Path("data/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            stat = os.statvfs(str(output_dir))
            available_bytes = stat.f_bavail * stat.f_frsize
            available_mb = available_bytes / (1024 * 1024)
            
            result["available_mb"] = int(available_mb)
            result["sufficient"] = available_mb >= required_mb
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    @staticmethod
    def check_required_directories() -> Dict[str, Any]:
        """Check that required directories exist and are writable."""
        required_dirs = [
            "data/temp",
            "data/output", 
            "data/cache",
            "app/static",
            "app/templates"
        ]
        
        result = {"valid": True, "issues": []}
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            
            try:
                # Create directory if it doesn't exist
                path.mkdir(parents=True, exist_ok=True)
                
                # Test write access
                test_file = path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                
            except Exception as e:
                result["valid"] = False
                result["issues"].append({
                    "directory": str(path),
                    "error": str(e)
                })
        
        return result
    
    @staticmethod
    def validate_move_clips_availability() -> Dict[str, Any]:
        """Validate that move clips are available for analysis."""
        result = {"valid": False, "clips_found": 0, "missing_clips": []}
        
        try:
            # Check for annotation file
            annotation_file = Path("data/bachata_annotations.json")
            if not annotation_file.exists():
                result["error"] = "Move annotations file not found"
                return result
            
            # Check for move clips directory
            clips_dir = Path("data/Bachata_steps")
            if not clips_dir.exists():
                result["error"] = "Move clips directory not found"
                return result
            
            # Count available clips
            clip_count = 0
            for category_dir in clips_dir.iterdir():
                if category_dir.is_dir():
                    for clip_file in category_dir.glob("*.mp4"):
                        if clip_file.is_file():
                            clip_count += 1
            
            result["clips_found"] = clip_count
            result["valid"] = clip_count >= 10  # Minimum required clips
            
            if not result["valid"]:
                result["error"] = f"Insufficient move clips found: {clip_count} (minimum 10 required)"
        
        except Exception as e:
            result["error"] = str(e)
        
        return result

def validate_system_requirements() -> Dict[str, Any]:
    """Comprehensive system requirements validation."""
    validation_result = {
        "valid": True,
        "issues": [],
        "warnings": []
    }
    
    # Check disk space
    disk_check = SystemResourceValidator.check_disk_space()
    if not disk_check["sufficient"]:
        validation_result["valid"] = False
        validation_result["issues"].append({
            "type": "disk_space",
            "message": f"Insufficient disk space: {disk_check['available_mb']}MB available, {disk_check['required_mb']}MB required"
        })
    
    # Check directories
    dir_check = SystemResourceValidator.check_required_directories()
    if not dir_check["valid"]:
        validation_result["valid"] = False
        validation_result["issues"].extend([
            {"type": "directory", "message": f"Directory issue: {issue['directory']} - {issue['error']}"}
            for issue in dir_check["issues"]
        ])
    
    # Check move clips
    clips_check = SystemResourceValidator.validate_move_clips_availability()
    if not clips_check["valid"]:
        validation_result["valid"] = False
        validation_result["issues"].append({
            "type": "move_clips",
            "message": clips_check.get("error", "Move clips validation failed")
        })
    elif clips_check["clips_found"] < 20:
        validation_result["warnings"].append({
            "type": "move_clips",
            "message": f"Limited move clips available: {clips_check['clips_found']} (recommended: 20+)"
        })
    
    return validation_result

async def validate_youtube_url_async(url: str) -> Dict[str, Any]:
    """Asynchronous URL validation for YouTube URLs or local file paths."""
    if not url:
        return {
            "valid": False,
            "error_type": "missing_url",
            "message": "URL is required"
        }
    
    url = url.strip()
    
    # Check if it's a local file path
    if url.startswith('data/songs/') and url.endswith('.mp3'):
        file_path = Path(url)
        if file_path.exists():
            return {
                "valid": True,
                "message": "Local song file found",
                "details": {
                    "type": "local_file",
                    "path": str(file_path),
                    "size": file_path.stat().st_size
                }
            }
        else:
            return {
                "valid": False,
                "error_type": "file_not_found",
                "message": f"Local song file not found: {url}"
            }
    
    # Otherwise, validate as YouTube URL
    return YouTubeURLValidator.validate_accessibility(url)