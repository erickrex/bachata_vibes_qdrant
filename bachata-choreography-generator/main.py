"""
Bachata Choreography Generator - Main FastAPI Application
"""
import asyncio
import logging
import os
import uuid
import psutil
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError as PydanticValidationError
import uvicorn

from app.services.choreography_pipeline import ChoreoGenerationPipeline, PipelineConfig, PipelineResult
from app.services.youtube_service import YouTubeService
from app.exceptions import (
    ChoreographyGenerationError, YouTubeDownloadError, MusicAnalysisError,
    VideoGenerationError, ValidationError, ResourceError, ServiceUnavailableError,
    choreography_exception_handler, validation_exception_handler, 
    http_exception_handler, general_exception_handler, create_error_response
)
from app.validation import (
    ChoreographyRequestValidator, SystemResourceValidator, 
    validate_system_requirements, validate_youtube_url_async
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bachata Choreography Generator",
    description="AI-powered Bachata choreography generator that creates dance sequences from YouTube music",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(ChoreographyGenerationError, choreography_exception_handler)
app.add_exception_handler(PydanticValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Global pipeline instance
pipeline = None
youtube_service = YouTubeService()

# Task storage for progress tracking
active_tasks: Dict[str, Dict] = {}

# Use the enhanced validator from validation module
ChoreographyRequest = ChoreographyRequestValidator

class ChoreographyResponse(BaseModel):
    """Response model for choreography generation."""
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    """Model for task status response."""
    task_id: str
    status: str
    progress: int
    stage: str
    message: str
    result: Optional[Dict] = None
    error: Optional[str] = None

def get_pipeline() -> ChoreoGenerationPipeline:
    """Get or create the global pipeline instance."""
    global pipeline
    if pipeline is None:
        config = PipelineConfig(
            quality_mode="balanced",
            enable_caching=True,
            max_workers=4,
            cleanup_after_generation=True
        )
        pipeline = ChoreoGenerationPipeline(config)
        logger.info("Choreography pipeline initialized")
    return pipeline

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup with comprehensive validation."""
    logger.info("Starting Bachata Choreography Generator API")
    
    try:
        # Validate system requirements
        system_check = validate_system_requirements()
        
        if not system_check["valid"]:
            logger.error("System requirements validation failed:")
            for issue in system_check["issues"]:
                logger.error(f"  - {issue['type']}: {issue['message']}")
            raise ServiceUnavailableError(
                message="System requirements not met",
                service_name="system",
                details=system_check
            )
        
        # Log warnings
        for warning in system_check.get("warnings", []):
            logger.warning(f"System warning - {warning['type']}: {warning['message']}")
        
        # Ensure required directories exist
        directories = [
            "app/static",
            "app/templates", 
            "data/temp",
            "data/output",
            "data/cache"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        get_pipeline()
        
        logger.info("API startup complete - all systems ready")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API")
    global pipeline
    if pipeline and hasattr(pipeline, '_executor') and pipeline._executor:
        pipeline._executor.shutdown(wait=True)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main application page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Basic system info
        health_status = {
            "status": "healthy",
            "version": "0.1.0",
            "timestamp": str(asyncio.get_event_loop().time()),
            "pipeline_initialized": pipeline is not None
        }
        
        # System resources
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            health_status["system"] = {
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        except Exception as e:
            health_status["system"] = {"error": str(e)}
        
        # Service status
        health_status["services"] = {
            "youtube_service": youtube_service is not None,
            "active_tasks": len(active_tasks),
            "pipeline_cache_enabled": pipeline.config.enable_caching if pipeline else False
        }
        
        # Quick system validation
        system_check = validate_system_requirements()
        health_status["system_validation"] = {
            "valid": system_check["valid"],
            "issues_count": len(system_check["issues"]),
            "warnings_count": len(system_check.get("warnings", []))
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "version": "0.1.0",
            "error": str(e)
        }

@app.post("/api/choreography", response_model=ChoreographyResponse)
async def create_choreography(
    request: ChoreographyRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new choreography from a YouTube URL.
    
    This endpoint starts the choreography generation process in the background
    and returns a task ID for tracking progress.
    """
    try:
        # Pre-flight system checks
        system_check = validate_system_requirements()
        if not system_check["valid"]:
            raise ResourceError(
                message="System requirements not met for choreography generation",
                resource_type="system",
                details=system_check
            )
        
        # Enhanced YouTube URL validation
        url_validation = await validate_youtube_url_async(request.youtube_url)
        if not url_validation["valid"]:
            raise YouTubeDownloadError(
                message=url_validation.get("message", "Invalid YouTube URL"),
                url=request.youtube_url,
                details=url_validation
            )
        
        # Check for too many concurrent tasks
        active_count = len([t for t in active_tasks.values() if t["status"] in ["started", "running"]])
        if active_count >= 3:  # Limit concurrent generations
            raise ResourceError(
                message="Too many concurrent generations. Please try again in a few minutes.",
                resource_type="concurrency",
                details={"active_tasks": active_count, "limit": 3}
            )
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status with enhanced info
        active_tasks[task_id] = {
            "status": "started",
            "progress": 0,
            "stage": "initializing",
            "message": "Starting choreography generation...",
            "result": None,
            "error": None,
            "created_at": asyncio.get_event_loop().time(),
            "request_params": {
                "difficulty": request.difficulty,
                "quality_mode": request.quality_mode,
                "energy_level": request.energy_level
            },
            "video_info": url_validation.get("details", {})
        }
        
        # Start background task with enhanced error handling
        background_tasks.add_task(
            generate_choreography_task_safe,
            task_id,
            request.youtube_url,
            request.difficulty,
            request.energy_level,
            request.quality_mode
        )
        
        logger.info(f"Started choreography generation task {task_id} for URL: {request.youtube_url}")
        logger.info(f"Video info: {url_validation.get('details', {})}")
        
        return ChoreographyResponse(
            task_id=task_id,
            status="started",
            message="Choreography generation started. Use the task ID to track progress."
        )
        
    except (YouTubeDownloadError, ResourceError, ValidationError) as e:
        # These are expected errors that should be returned to the user
        logger.warning(f"Choreography generation rejected: {e.message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting choreography generation: {e}", exc_info=True)
        raise ServiceUnavailableError(
            message="Unable to start choreography generation due to an internal error",
            service_name="choreography_api",
            details={"error": str(e)}
        )

@app.get("/api/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a choreography generation task."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = active_tasks[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task_data["status"],
        progress=task_data["progress"],
        stage=task_data["stage"],
        message=task_data["message"],
        result=task_data["result"],
        error=task_data["error"]
    )

@app.get("/api/video/{filename}")
async def serve_video(filename: str):
    """
    Serve generated choreography videos with proper headers for browser playback.
    
    This endpoint serves video files with appropriate headers for streaming
    and browser compatibility.
    """
    try:
        # Input validation
        if not filename or not isinstance(filename, str):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Security: sanitize filename (prevent path traversal)
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            logger.warning(f"Potential path traversal attempt: {filename}")
            raise HTTPException(status_code=400, detail="Invalid filename format")
        
        # Only allow specific file extensions
        allowed_extensions = {'.mp4', '.webm', '.mov'}
        file_ext = Path(safe_filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Security: only allow serving from output directory
        video_path = Path("data/output") / safe_filename
        
        # Validate file exists and is in the correct directory
        if not video_path.exists() or not video_path.is_file():
            logger.info(f"Video file not found: {video_path}")
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Double-check path security (prevent path traversal)
        try:
            video_path_resolved = video_path.resolve()
            output_dir_resolved = Path("data/output").resolve()
            
            if not str(video_path_resolved).startswith(str(output_dir_resolved)):
                logger.warning(f"Path traversal attempt blocked: {video_path}")
                raise HTTPException(status_code=403, detail="Access denied")
        except Exception as e:
            logger.error(f"Path resolution error for {video_path}: {e}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get file info
        try:
            file_stat = video_path.stat()
            file_size = file_stat.st_size
            
            # Check if file is too large (prevent serving corrupted/incomplete files)
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                logger.warning(f"File too large: {video_path} ({file_size} bytes)")
                raise HTTPException(status_code=413, detail="File too large")
            
            if file_size == 0:
                logger.warning(f"Empty file: {video_path}")
                raise HTTPException(status_code=404, detail="Video file is empty")
                
        except OSError as e:
            logger.error(f"Error accessing file {video_path}: {e}")
            raise HTTPException(status_code=500, detail="Error accessing video file")
        
        # Determine media type
        media_type_map = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mov': 'video/quicktime'
        }
        media_type = media_type_map.get(file_ext, 'video/mp4')
        
        # Return video file with proper headers
        return FileResponse(
            path=str(video_path),
            media_type=media_type,
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600",
                "X-Content-Type-Options": "nosniff"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error serving video {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/videos")
async def list_videos():
    """List all available generated choreography videos."""
    try:
        output_dir = Path("data/output")
        videos = []
        
        if output_dir.exists():
            for video_file in output_dir.glob("*.mp4"):
                stat = video_file.stat()
                videos.append({
                    "filename": video_file.name,
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "url": f"/api/video/{video_file.name}"
                })
        
        return {"videos": videos}
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail="Error listing videos")

@app.get("/api/songs")
async def list_songs():
    """List all available local songs."""
    try:
        songs_dir = Path("data/songs")
        songs = []
        
        if songs_dir.exists():
            for song_file in songs_dir.glob("*.mp3"):
                stat = song_file.stat()
                # Create a friendly display name from filename
                display_name = song_file.stem.replace('_', ' ').title()
                songs.append({
                    "filename": song_file.name,
                    "display_name": display_name,
                    "path": str(song_file),
                    "size": stat.st_size,
                    "created": stat.st_ctime
                })
        
        # Sort by display name
        songs.sort(key=lambda x: x['display_name'])
        
        return {"songs": songs}
        
    except Exception as e:
        logger.error(f"Error listing songs: {e}")
        raise HTTPException(status_code=500, detail="Error listing songs")

async def generate_choreography_task_safe(
    task_id: str,
    youtube_url: str,
    difficulty: str,
    energy_level: Optional[str],
    quality_mode: str
):
    """
    Safe wrapper for choreography generation with comprehensive error handling.
    """
    try:
        await generate_choreography_task(
            task_id, youtube_url, difficulty, energy_level, quality_mode
        )
    except Exception as e:
        logger.error(f"Critical error in task {task_id}: {e}", exc_info=True)
        
        # Ensure task status is updated even on critical failure
        if task_id in active_tasks:
            active_tasks[task_id].update({
                "status": "failed",
                "progress": 0,
                "stage": "failed",
                "message": "Critical system error occurred",
                "error": "Internal system error - please try again later"
            })

async def generate_choreography_task(
    task_id: str,
    youtube_url: str,
    difficulty: str,
    energy_level: Optional[str],
    quality_mode: str
):
    """
    Background task for generating choreography with enhanced error handling.
    
    This function runs the complete choreography generation pipeline
    and updates the task status throughout the process.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Update task status
        active_tasks[task_id].update({
            "status": "running",
            "progress": 5,
            "stage": "downloading",
            "message": "Downloading audio from YouTube..."
        })
        
        # Get pipeline with specified quality mode
        config = PipelineConfig(
            quality_mode=quality_mode,
            enable_caching=True,
            max_workers=4,
            cleanup_after_generation=True
        )
        task_pipeline = ChoreoGenerationPipeline(config)
        
        # Progress tracking with more granular updates
        progress_stages = [
            (10, "downloading", "Downloading audio from YouTube..."),
            (25, "analyzing", "Analyzing musical structure and tempo..."),
            (40, "selecting", "Analyzing dance moves and selecting sequences..."),
            (70, "generating", "Generating choreography video..."),
            (90, "finalizing", "Finalizing video and metadata...")
        ]
        
        # Simulate progress updates during generation
        for progress, stage, message in progress_stages:
            active_tasks[task_id].update({
                "progress": progress,
                "stage": stage,
                "message": message
            })
            
            # Small delay to allow progress updates to be visible
            await asyncio.sleep(0.1)
        
        # Generate choreography
        result = await task_pipeline.generate_choreography(
            audio_input=youtube_url,
            difficulty=difficulty,
            energy_level=energy_level
        )
        
        if result.success:
            # Update with success
            active_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "stage": "completed",
                "message": "Choreography generation completed successfully!",
                "completed_at": asyncio.get_event_loop().time(),
                "total_time": asyncio.get_event_loop().time() - start_time,
                "result": {
                    "video_path": result.output_path,
                    "metadata_path": result.metadata_path,
                    "processing_time": result.processing_time,
                    "sequence_duration": result.sequence_duration,
                    "moves_analyzed": result.moves_analyzed,
                    "recommendations_generated": result.recommendations_generated,
                    "cache_hits": result.cache_hits,
                    "cache_misses": result.cache_misses,
                    "video_filename": Path(result.output_path).name if result.output_path else None
                }
            })
            
            logger.info(f"Task {task_id} completed successfully in {result.processing_time:.2f}s")
            
        else:
            # Categorize the error for better user feedback
            error_message = result.error_message or "Unknown error occurred"
            user_message = _get_user_friendly_error_message(error_message)
            
            # Update with failure
            active_tasks[task_id].update({
                "status": "failed",
                "progress": 0,
                "stage": "failed",
                "message": user_message,
                "error": error_message,
                "failed_at": asyncio.get_event_loop().time(),
                "total_time": asyncio.get_event_loop().time() - start_time
            })
            
            logger.error(f"Task {task_id} failed: {error_message}")
    
    except YouTubeDownloadError as e:
        active_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "stage": "failed",
            "message": e.message,
            "error": f"YouTube download failed: {e.message}",
            "failed_at": asyncio.get_event_loop().time()
        })
        logger.error(f"Task {task_id} - YouTube download error: {e.message}")
    
    except MusicAnalysisError as e:
        active_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "stage": "failed",
            "message": e.message,
            "error": f"Music analysis failed: {e.message}",
            "failed_at": asyncio.get_event_loop().time()
        })
        logger.error(f"Task {task_id} - Music analysis error: {e.message}")
    
    except VideoGenerationError as e:
        active_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "stage": "failed",
            "message": e.message,
            "error": f"Video generation failed: {e.message}",
            "failed_at": asyncio.get_event_loop().time()
        })
        logger.error(f"Task {task_id} - Video generation error: {e.message}")
    
    except ResourceError as e:
        active_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "stage": "failed",
            "message": e.message,
            "error": f"Resource error: {e.message}",
            "failed_at": asyncio.get_event_loop().time()
        })
        logger.error(f"Task {task_id} - Resource error: {e.message}")
    
    except Exception as e:
        # Generic error handling
        error_message = str(e)
        user_message = "An unexpected error occurred during generation. Please try again."
        
        active_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "stage": "failed", 
            "message": user_message,
            "error": error_message,
            "failed_at": asyncio.get_event_loop().time()
        })
        
        logger.error(f"Task {task_id} failed with unexpected exception: {e}", exc_info=True)

def _get_user_friendly_error_message(error_message: str) -> str:
    """Convert technical error messages to user-friendly messages."""
    error_lower = error_message.lower()
    
    if "youtube" in error_lower and ("download" in error_lower or "fetch" in error_lower):
        return "Unable to download audio from YouTube. Please check the URL and try again."
    elif "music analysis" in error_lower or "librosa" in error_lower:
        return "Unable to analyze the music. The audio may be corrupted or in an unsupported format."
    elif "move analysis" in error_lower or "mediapipe" in error_lower:
        return "Unable to analyze dance moves. Please contact support if this persists."
    elif "video generation" in error_lower or "ffmpeg" in error_lower:
        return "Unable to generate the choreography video. Please try again."
    elif "memory" in error_lower or "disk" in error_lower:
        return "Insufficient system resources. Please try again later."
    elif "timeout" in error_lower:
        return "Generation took too long and was cancelled. Please try with a shorter song."
    else:
        return "An unexpected error occurred. Please try again or contact support."

# Additional utility endpoints with enhanced error handling
@app.delete("/api/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task (cleanup only, cannot stop running generation)."""
    try:
        if not task_id or not isinstance(task_id, str):
            raise HTTPException(status_code=400, detail="Invalid task ID")
        
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_status = active_tasks[task_id]["status"]
        
        # Remove from active tasks
        del active_tasks[task_id]
        
        logger.info(f"Task {task_id} removed from tracking (was {task_status})")
        
        return {
            "message": "Task removed from tracking",
            "task_id": task_id,
            "previous_status": task_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Error cancelling task")

@app.get("/api/tasks")
async def list_tasks():
    """List all active tasks with filtering options."""
    try:
        # Clean up old completed/failed tasks (older than 1 hour)
        current_time = asyncio.get_event_loop().time()
        tasks_to_remove = []
        
        for task_id, task_data in active_tasks.items():
            task_age = current_time - task_data.get("created_at", current_time)
            if task_age > 3600 and task_data["status"] in ["completed", "failed"]:  # 1 hour
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del active_tasks[task_id]
            logger.debug(f"Cleaned up old task: {task_id}")
        
        # Return task summary
        task_summary = {
            "total_tasks": len(active_tasks),
            "running_tasks": len([t for t in active_tasks.values() if t["status"] == "running"]),
            "completed_tasks": len([t for t in active_tasks.values() if t["status"] == "completed"]),
            "failed_tasks": len([t for t in active_tasks.values() if t["status"] == "failed"]),
            "tasks": active_tasks
        }
        
        return task_summary
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving tasks")

@app.get("/api/system/status")
async def system_status():
    """Get detailed system status and diagnostics."""
    try:
        status = {
            "timestamp": asyncio.get_event_loop().time(),
            "system_validation": validate_system_requirements(),
            "active_tasks": len(active_tasks),
            "pipeline_status": {
                "initialized": pipeline is not None,
                "cache_enabled": pipeline.config.enable_caching if pipeline else False,
                "quality_mode": pipeline.config.quality_mode if pipeline else None
            }
        }
        
        # Add system resources if available
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            status["resources"] = {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": disk.percent
                }
            }
        except Exception as e:
            status["resources"] = {"error": str(e)}
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system status")

@app.post("/api/validate/youtube")
async def validate_youtube_url(request: dict):
    """Validate a YouTube URL without starting generation."""
    try:
        url = request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        validation_result = await validate_youtube_url_async(url)
        
        return {
            "valid": validation_result["valid"],
            "message": validation_result.get("message"),
            "details": validation_result.get("details", {}),
            "error_type": validation_result.get("error_type")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating YouTube URL: {e}")
        raise HTTPException(status_code=500, detail="Error validating URL")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
