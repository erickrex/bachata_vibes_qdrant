"""
Custom exceptions and error handling for the Bachata Choreography Generator.
"""
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class ChoreographyGenerationError(Exception):
    """Base exception for choreography generation errors."""
    
    def __init__(self, message: str, error_code: str = "GENERATION_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class YouTubeDownloadError(ChoreographyGenerationError):
    """Exception raised when YouTube download fails."""
    
    def __init__(self, message: str, url: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "YOUTUBE_DOWNLOAD_ERROR", details)
        self.url = url

class MusicAnalysisError(ChoreographyGenerationError):
    """Exception raised when music analysis fails."""
    
    def __init__(self, message: str, audio_path: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MUSIC_ANALYSIS_ERROR", details)
        self.audio_path = audio_path

class MoveAnalysisError(ChoreographyGenerationError):
    """Exception raised when move analysis fails."""
    
    def __init__(self, message: str, video_path: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MOVE_ANALYSIS_ERROR", details)
        self.video_path = video_path

class VideoGenerationError(ChoreographyGenerationError):
    """Exception raised when video generation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VIDEO_GENERATION_ERROR", details)

class ValidationError(ChoreographyGenerationError):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str, field: str, value: Any, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value

class ResourceError(ChoreographyGenerationError):
    """Exception raised when system resources are insufficient."""
    
    def __init__(self, message: str, resource_type: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RESOURCE_ERROR", details)
        self.resource_type = resource_type

class ServiceUnavailableError(ChoreographyGenerationError):
    """Exception raised when a required service is unavailable."""
    
    def __init__(self, message: str, service_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SERVICE_UNAVAILABLE_ERROR", details)
        self.service_name = service_name

def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create a standardized error response."""
    content = {
        "error": {
            "code": error_code,
            "message": message,
            "details": details or {}
        }
    }
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )

async def choreography_exception_handler(request: Request, exc: ChoreographyGenerationError) -> JSONResponse:
    """Handle choreography generation exceptions."""
    logger.error(f"Choreography generation error: {exc.message}", exc_info=True)
    
    # Map exception types to HTTP status codes
    status_code_map = {
        "YOUTUBE_DOWNLOAD_ERROR": 400,
        "MUSIC_ANALYSIS_ERROR": 422,
        "MOVE_ANALYSIS_ERROR": 422,
        "VIDEO_GENERATION_ERROR": 500,
        "VALIDATION_ERROR": 400,
        "RESOURCE_ERROR": 503,
        "SERVICE_UNAVAILABLE_ERROR": 503,
        "GENERATION_ERROR": 500
    }
    
    status_code = status_code_map.get(exc.error_code, 500)
    
    return create_error_response(
        status_code=status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details
    )

async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic validation exceptions."""
    logger.error(f"Validation error: {str(exc)}", exc_info=True)
    
    return create_error_response(
        status_code=422,
        error_code="VALIDATION_ERROR",
        message="Invalid input data",
        details={"validation_error": str(exc)}
    )

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    
    return create_error_response(
        status_code=exc.status_code,
        error_code="HTTP_ERROR",
        message=exc.detail,
        details={"status_code": exc.status_code}
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return create_error_response(
        status_code=500,
        error_code="INTERNAL_ERROR",
        message="An unexpected error occurred. Please try again later.",
        details={"error_type": type(exc).__name__}
    )

# User-friendly error messages
ERROR_MESSAGES = {
    "YOUTUBE_DOWNLOAD_ERROR": {
        "invalid_url": "The YouTube URL you provided is not valid. Please check the URL and try again.",
        "video_unavailable": "This YouTube video is not available or has been removed.",
        "restricted_content": "This video is restricted and cannot be downloaded. Please try a different video.",
        "network_error": "Unable to download from YouTube. Please check your internet connection and try again.",
        "format_error": "The video format is not supported. Please try a different video."
    },
    "MUSIC_ANALYSIS_ERROR": {
        "file_not_found": "The audio file could not be found. Please try downloading again.",
        "unsupported_format": "The audio format is not supported. Please try a different video.",
        "corrupted_file": "The audio file appears to be corrupted. Please try downloading again.",
        "too_short": "The audio is too short (minimum 30 seconds required).",
        "too_long": "The audio is too long (maximum 10 minutes supported).",
        "analysis_failed": "Unable to analyze the music. The audio may be corrupted or in an unsupported format."
    },
    "MOVE_ANALYSIS_ERROR": {
        "video_not_found": "Move video file not found. Please contact support.",
        "pose_detection_failed": "Unable to detect poses in the move video. The video quality may be too low.",
        "insufficient_frames": "The move video has insufficient frames for analysis.",
        "analysis_timeout": "Move analysis timed out. Please try again."
    },
    "VIDEO_GENERATION_ERROR": {
        "ffmpeg_error": "Video generation failed due to encoding issues. Please try again.",
        "insufficient_moves": "Not enough suitable moves found for this song. Please try a different song.",
        "sync_error": "Unable to synchronize moves with the music. Please try again.",
        "output_error": "Unable to save the generated video. Please check disk space and try again."
    },
    "VALIDATION_ERROR": {
        "invalid_difficulty": "Difficulty must be 'beginner', 'intermediate', or 'advanced'.",
        "invalid_quality": "Quality mode must be 'fast', 'balanced', or 'high_quality'.",
        "invalid_url_format": "Please provide a valid YouTube URL.",
        "missing_required_field": "Required field is missing."
    },
    "RESOURCE_ERROR": {
        "insufficient_memory": "Insufficient memory to process this request. Please try again later.",
        "disk_space_full": "Insufficient disk space. Please contact support.",
        "cpu_overload": "System is currently overloaded. Please try again in a few minutes."
    },
    "SERVICE_UNAVAILABLE_ERROR": {
        "youtube_service": "YouTube service is currently unavailable. Please try again later.",
        "analysis_service": "Analysis service is currently unavailable. Please try again later.",
        "generation_service": "Video generation service is currently unavailable. Please try again later."
    }
}

def get_user_friendly_message(error_code: str, error_type: str = "default") -> str:
    """Get a user-friendly error message."""
    messages = ERROR_MESSAGES.get(error_code, {})
    return messages.get(error_type, f"An error occurred: {error_code}")