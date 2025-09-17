"""
Bachata Choreography Generator - Main FastAPI Application
"""
import asyncio
import json
import logging
import os
import time
import uuid
import psutil
from pathlib import Path
from typing import Dict, Optional, Any, List

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError as PydanticValidationError
import uvicorn

from app.services.choreography_pipeline import ChoreoGenerationPipeline, PipelineConfig, PipelineResult
from app.services.youtube_service import YouTubeService
from app.services.qdrant_service import create_qdrant_service, QdrantConfig
from app.services.superlinked_recommendation_engine import SuperlinkedRecommendationEngine
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

# Global service instances
pipeline = None
youtube_service = YouTubeService()
qdrant_service = None
superlinked_engine = None

# Task storage for progress tracking
active_tasks: Dict[str, Dict] = {}

# Qdrant connection monitoring
qdrant_connection_status = {
    "connected": False,
    "last_check": 0,
    "reconnect_attempts": 0,
    "max_reconnect_attempts": 5,
    "reconnect_interval": 30  # seconds
}

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
            cleanup_after_generation=True,
            enable_qdrant=True,
            auto_populate_qdrant=True
        )
        pipeline = ChoreoGenerationPipeline(config)
        logger.info("Choreography pipeline initialized with SuperlinkedRecommendationEngine")
    return pipeline

def get_superlinked_engine() -> SuperlinkedRecommendationEngine:
    """Get or create the global SuperlinkedRecommendationEngine instance."""
    global superlinked_engine
    if superlinked_engine is None:
        # Use environment-based configuration for Qdrant Cloud
        qdrant_config = QdrantConfig.from_env()
        superlinked_engine = SuperlinkedRecommendationEngine(
            data_dir="data",
            qdrant_config=qdrant_config
        )
        logger.info("SuperlinkedRecommendationEngine initialized with Qdrant Cloud integration")
    return superlinked_engine

def get_qdrant_service():
    """Get or create the global Qdrant service instance."""
    global qdrant_service
    if qdrant_service is None:
        # Use environment-based configuration for cloud deployment
        config = QdrantConfig.from_env()
        qdrant_service = create_qdrant_service(config)
        
        if config.url:
            logger.info(f"Qdrant service initialized (Cloud): {config.url}")
        else:
            logger.info(f"Qdrant service initialized (Local): {config.host}:{config.port}")
    return qdrant_service

async def check_qdrant_connection():
    """Check and maintain Qdrant connection with automatic reconnection."""
    global qdrant_service, qdrant_connection_status
    
    current_time = time.time()
    
    # Skip if recently checked
    if current_time - qdrant_connection_status["last_check"] < qdrant_connection_status["reconnect_interval"]:
        return qdrant_connection_status["connected"]
    
    qdrant_connection_status["last_check"] = current_time
    
    try:
        if qdrant_service is None:
            qdrant_service = get_qdrant_service()
        
        # Perform health check
        health_status = qdrant_service.health_check()
        
        if health_status.get("qdrant_available", False):
            if not qdrant_connection_status["connected"]:
                logger.info("Qdrant connection restored")
            qdrant_connection_status["connected"] = True
            qdrant_connection_status["reconnect_attempts"] = 0
            return True
        else:
            raise Exception(health_status.get("error_message", "Qdrant health check failed"))
            
    except Exception as e:
        if qdrant_connection_status["connected"]:
            logger.warning(f"Qdrant connection lost: {e}")
        
        qdrant_connection_status["connected"] = False
        qdrant_connection_status["reconnect_attempts"] += 1
        
        # Attempt reconnection if under limit
        if qdrant_connection_status["reconnect_attempts"] <= qdrant_connection_status["max_reconnect_attempts"]:
            logger.info(f"Attempting Qdrant reconnection ({qdrant_connection_status['reconnect_attempts']}/{qdrant_connection_status['max_reconnect_attempts']})")
            try:
                qdrant_service = get_qdrant_service()
                health_status = qdrant_service.health_check()
                if health_status.get("qdrant_available", False):
                    qdrant_connection_status["connected"] = True
                    qdrant_connection_status["reconnect_attempts"] = 0
                    logger.info("Qdrant reconnection successful")
                    return True
            except Exception as reconnect_error:
                logger.warning(f"Qdrant reconnection failed: {reconnect_error}")
        
        return False

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
            # Note: qdrant_storage removed - using Qdrant Cloud deployment
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant service
        try:
            qdrant_connection_status["startup_time"] = time.time()
            get_qdrant_service()
            await check_qdrant_connection()
            if qdrant_connection_status["connected"]:
                logger.info("Qdrant service initialized and connected")
            else:
                logger.warning("Qdrant service initialized but not connected - will use fallback")
        except Exception as e:
            logger.warning(f"Qdrant initialization failed: {e} - will use fallback")
        
        # Initialize pipeline
        get_pipeline()
        
        # Initialize SuperlinkedRecommendationEngine
        try:
            get_superlinked_engine()
            logger.info("SuperlinkedRecommendationEngine initialized successfully")
        except Exception as e:
            logger.warning(f"SuperlinkedRecommendationEngine initialization failed: {e} - will use fallback")
        
        logger.info("API startup complete - all systems ready")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API")
    global pipeline, qdrant_service, superlinked_engine
    
    # Shutdown pipeline
    if pipeline and hasattr(pipeline, '_executor') and pipeline._executor:
        pipeline._executor.shutdown(wait=True)
    
    # Cleanup Qdrant service
    if qdrant_service and hasattr(qdrant_service, 'client') and qdrant_service.client:
        try:
            # Qdrant client doesn't need explicit cleanup, but we can log the shutdown
            logger.info("Qdrant service shutdown")
        except Exception as e:
            logger.warning(f"Error during Qdrant shutdown: {e}")
    
    # Cleanup SuperlinkedRecommendationEngine
    if superlinked_engine:
        try:
            # SuperlinkedRecommendationEngine doesn't need explicit cleanup, but we can log the shutdown
            logger.info("SuperlinkedRecommendationEngine shutdown")
        except Exception as e:
            logger.warning(f"Error during SuperlinkedRecommendationEngine shutdown: {e}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main application page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check Qdrant connection
        await check_qdrant_connection()
        
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
            "pipeline_cache_enabled": pipeline.config.enable_caching if pipeline else False,
            "qdrant_service": qdrant_service is not None,
            "superlinked_engine": superlinked_engine is not None
        }
        
        # Qdrant status with collection statistics
        if qdrant_service:
            try:
                qdrant_health = qdrant_service.health_check()
                collection_info = qdrant_service.get_collection_info()
                qdrant_stats = qdrant_service.get_statistics()
                
                health_status["qdrant"] = {
                    "connected": qdrant_connection_status["connected"],
                    "available": qdrant_health.get("qdrant_available", False),
                    "collection_exists": qdrant_health.get("collection_exists", False),
                    "can_search": qdrant_health.get("can_search", False),
                    "can_store": qdrant_health.get("can_store", False),
                    "reconnect_attempts": qdrant_connection_status["reconnect_attempts"],
                    "last_check": qdrant_connection_status["last_check"],
                    "collection_stats": {
                        "points_count": collection_info.get("points_count", 0),
                        "vector_size": collection_info.get("vector_size", 0),
                        "distance_metric": collection_info.get("distance", "unknown"),
                        "status": collection_info.get("status", "unknown"),
                        "indexed_vectors": collection_info.get("indexed_vectors_count", 0)
                    },
                    "performance_stats": {
                        "search_requests": qdrant_stats.search_requests,
                        "avg_search_time_ms": round(qdrant_stats.avg_search_time_ms, 2),
                        "cache_hits": qdrant_stats.cache_hits,
                        "collection_size_mb": round(qdrant_stats.collection_size_mb, 2)
                    },
                    "error_message": qdrant_health.get("error_message")
                }
            except Exception as e:
                health_status["qdrant"] = {
                    "connected": False,
                    "error": str(e)
                }
        else:
            health_status["qdrant"] = {
                "connected": False,
                "error": "Qdrant service not initialized"
            }
        
        # SuperlinkedRecommendationEngine status
        if superlinked_engine:
            try:
                perf_stats = superlinked_engine.get_performance_stats()
                health_status["superlinked"] = {
                    "initialized": True,
                    "qdrant_available": superlinked_engine.is_qdrant_available,
                    "embedding_dimension": superlinked_engine.embedding_service.total_dimension,
                    "performance_stats": {
                        "unified_searches": perf_stats.get("unified_searches", 0),
                        "avg_search_time_ms": round(perf_stats.get("avg_search_time_ms", 0.0), 2),
                        "natural_language_queries": perf_stats.get("natural_language_queries", 0),
                        "cache_hits": perf_stats.get("cache_hits", 0),
                        "qdrant_searches": perf_stats.get("qdrant_searches", 0),
                        "fallback_searches": perf_stats.get("fallback_searches", 0)
                    }
                }
            except Exception as e:
                health_status["superlinked"] = {
                    "initialized": False,
                    "error": str(e)
                }
        else:
            health_status["superlinked"] = {
                "initialized": False,
                "error": "SuperlinkedRecommendationEngine not initialized"
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
            "video_info": url_validation.get("details", {}),
            "qdrant_metrics": {
                "enabled": qdrant_connection_status["connected"],
                "embeddings_stored": 0,
                "embeddings_retrieved": 0,
                "search_time_ms": 0.0
            },
            "superlinked_metrics": {
                "unified_searches": 0,
                "avg_search_time_ms": 0.0,
                "natural_language_queries": 0,
                "qdrant_searches": 0,
                "fallback_searches": 0,
                "cache_hits": 0
            }
        }
        
        # Start background task with enhanced error handling
        background_tasks.add_task(
            generate_choreography_task_safe,
            task_id,
            request.youtube_url,
            request.difficulty,
            request.energy_level,
            request.quality_mode,
            request.role_focus,
            request.move_types,
            request.tempo_range
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
    """Get the status of a choreography generation task with AI insights."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = active_tasks[task_id]
    
    # Enhance task status with real-time AI insights
    enhanced_result = task_data["result"]
    if enhanced_result and task_data["status"] == "completed":
        # Add real-time performance metrics
        if superlinked_engine:
            perf_stats = superlinked_engine.get_performance_stats()
            enhanced_result["ai_performance"] = {
                "search_method": "qdrant_cloud" if superlinked_engine.is_qdrant_available else "fallback_memory",
                "unified_searches": perf_stats.get("unified_searches", 0),
                "avg_search_time_ms": round(perf_stats.get("avg_search_time_ms", 0.0), 2),
                "cache_hits": perf_stats.get("cache_hits", 0),
                "embedding_dimension": getattr(superlinked_engine.embedding_service, 'total_dimension', 470)
            }
    
    return TaskStatus(
        task_id=task_id,
        status=task_data["status"],
        progress=task_data["progress"],
        stage=task_data["stage"],
        message=task_data["message"],
        result=enhanced_result,
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

@app.get("/api/metadata/{filename}")
async def serve_metadata(filename: str):
    """Serve choreography metadata files with enhanced AI insights."""
    try:
        # Security: sanitize filename
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            logger.warning(f"Potential path traversal attempt: {filename}")
            raise HTTPException(status_code=400, detail="Invalid filename format")
        
        # Only allow JSON files
        if not safe_filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files are allowed")
        
        # Security: only allow serving from metadata directory
        metadata_path = Path("data/choreography_metadata") / safe_filename
        
        # Validate file exists and is in the correct directory
        if not metadata_path.exists() or not metadata_path.is_file():
            logger.info(f"Metadata file not found: {metadata_path}")
            raise HTTPException(status_code=404, detail="Metadata not found")
        
        # Double-check path security
        try:
            metadata_path_resolved = metadata_path.resolve()
            metadata_dir_resolved = Path("data/choreography_metadata").resolve()
            
            if not str(metadata_path_resolved).startswith(str(metadata_dir_resolved)):
                logger.warning(f"Path traversal attempt blocked: {metadata_path}")
                raise HTTPException(status_code=403, detail="Access denied")
        except Exception as e:
            logger.error(f"Path resolution error for {metadata_path}: {e}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Load and enhance metadata with AI insights
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Add AI insights if not already present
        if 'ai_insights' not in metadata:
            metadata['ai_insights'] = await _generate_ai_insights_for_metadata(metadata)
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error serving metadata {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

async def _generate_ai_insights_for_metadata(metadata: dict) -> dict:
    """Generate AI insights for existing metadata."""
    try:
        # Get SuperlinkedRecommendationEngine for insights
        engine = get_superlinked_engine()
        
        # Extract move information
        moves_used = metadata.get('moves_used', [])
        music_analysis = metadata.get('music_analysis', {})
        
        # Analyze move diversity and patterns
        move_categories = {}
        move_transitions = []
        
        for i, move in enumerate(moves_used):
            # Extract move category from video path
            video_path = move.get('video_path', '')
            if '/Bachata_steps/' in video_path:
                category = video_path.split('/Bachata_steps/')[1].split('/')[0]
                move_categories[category] = move_categories.get(category, 0) + 1
                
                # Track transitions
                if i > 0:
                    prev_move = moves_used[i-1]
                    prev_category = prev_move.get('video_path', '').split('/Bachata_steps/')[1].split('/')[0] if '/Bachata_steps/' in prev_move.get('video_path', '') else 'unknown'
                    move_transitions.append(f"{prev_category} → {category}")
        
        # Calculate diversity metrics
        total_moves = len(moves_used)
        unique_categories = len(move_categories)
        diversity_score = unique_categories / max(total_moves, 1) if total_moves > 0 else 0
        
        # Analyze tempo matching
        audio_tempo = music_analysis.get('tempo', 0)
        tempo_variance = abs(audio_tempo - 125) / 125 if audio_tempo > 0 else 0  # 125 BPM is typical bachata
        
        # Generate insights
        ai_insights = {
            "embedding_analysis": {
                "total_dimension": 470,  # SuperlinkedEmbeddingService total dimension
                "embedding_spaces": {
                    "semantic_text": {"dimension": 384, "weight": 0.3, "description": "Move descriptions and semantic understanding"},
                    "tempo_matching": {"dimension": 8, "weight": 0.25, "description": "BPM alignment and rhythm compatibility"},
                    "difficulty_progression": {"dimension": 8, "weight": 0.15, "description": "Skill level and complexity matching"},
                    "energy_compatibility": {"dimension": 16, "weight": 0.15, "description": "Energy level and intensity matching"},
                    "role_dynamics": {"dimension": 16, "weight": 0.1, "description": "Lead/follow role compatibility"},
                    "transition_flow": {"dimension": 64, "weight": 0.05, "description": "Move-to-move transition patterns"}
                }
            },
            "search_performance": {
                "method": "qdrant_cloud" if engine.is_qdrant_available else "fallback_memory",
                "vector_database": "Qdrant Cloud" if engine.is_qdrant_available else "In-Memory",
                "embedding_dimension": 470,
                "search_algorithm": "Cosine Similarity with Multi-Space Fusion"
            },
            "choreography_analysis": {
                "move_diversity": {
                    "unique_categories": unique_categories,
                    "total_moves": total_moves,
                    "diversity_score": round(diversity_score, 3),
                    "category_distribution": move_categories,
                    "dominant_category": max(move_categories.items(), key=lambda x: x[1])[0] if move_categories else "none"
                },
                "tempo_analysis": {
                    "audio_tempo_bpm": audio_tempo,
                    "tempo_variance": round(tempo_variance, 3),
                    "tempo_matching_quality": "excellent" if tempo_variance < 0.1 else "good" if tempo_variance < 0.2 else "moderate",
                    "rhythm_strength": music_analysis.get('rhythm_strength', 0),
                    "syncopation_level": music_analysis.get('syncopation', 0)
                },
                "transition_patterns": {
                    "total_transitions": len(move_transitions),
                    "unique_transitions": len(set(move_transitions)),
                    "most_common_transitions": list(set(move_transitions))[:5],
                    "flow_quality": "smooth" if len(set(move_transitions)) > len(move_transitions) * 0.7 else "repetitive"
                }
            },
            "ai_recommendations": {
                "similarity_matching": "Vector similarity search with 470-dimensional embeddings",
                "personalization": "Multi-factor embedding spaces for nuanced matching",
                "diversity_algorithm": "Category-based selection with randomization",
                "quality_indicators": [
                    f"Tempo matching: {round((1 - tempo_variance) * 100, 1)}%",
                    f"Move diversity: {round(diversity_score * 100, 1)}%",
                    f"Rhythm alignment: {round(music_analysis.get('rhythm_strength', 0) * 100, 1)}%"
                ]
            }
        }
        
        return ai_insights
        
    except Exception as e:
        logger.warning(f"Error generating AI insights: {e}")
        return {
            "embedding_analysis": {"error": "Unable to generate insights"},
            "search_performance": {"method": "unknown"},
            "choreography_analysis": {"error": "Analysis unavailable"},
            "ai_recommendations": {"error": "Recommendations unavailable"}
        }

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

@app.get("/api/ai-insights/{video_filename}")
async def get_ai_insights(video_filename: str):
    """Get detailed AI insights for a specific choreography video."""
    try:
        # Extract base name for metadata lookup
        base_name = video_filename.replace('_balanced_choreography.mp4', '')
        metadata_filename = f"{base_name}_pipeline_metadata.json"
        metadata_path = Path("data/choreography_metadata") / metadata_filename
        
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Metadata not found for this video")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Generate comprehensive AI insights
        ai_insights = await _generate_comprehensive_ai_insights(metadata)
        
        return {
            "video_filename": video_filename,
            "ai_insights": ai_insights,
            "metadata_source": metadata_filename,
            "generated_at": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating AI insights for {video_filename}: {e}")
        raise HTTPException(status_code=500, detail="Error generating AI insights")

async def _generate_comprehensive_ai_insights(metadata: dict) -> dict:
    """Generate comprehensive AI insights with detailed analysis."""
    try:
        # Get engine for real-time stats
        engine = get_superlinked_engine()
        perf_stats = engine.get_performance_stats() if engine else {}
        
        # Extract data
        moves_used = metadata.get('moves_used', [])
        music_analysis = metadata.get('music_analysis', {})
        
        # Advanced move analysis
        move_analysis = _analyze_move_patterns(moves_used)
        tempo_analysis = _analyze_tempo_matching(music_analysis, moves_used)
        diversity_analysis = _analyze_choreography_diversity(moves_used)
        
        return {
            "embedding_analysis": {
                "total_dimension": 470,
                "active_spaces": {
                    "semantic_understanding": {"dim": 384, "weight": 30, "description": "Natural language move descriptions"},
                    "tempo_synchronization": {"dim": 8, "weight": 25, "description": "BPM and rhythm alignment"},
                    "difficulty_progression": {"dim": 8, "weight": 15, "description": "Skill level matching"},
                    "energy_compatibility": {"dim": 16, "weight": 15, "description": "Intensity and mood matching"},
                    "role_dynamics": {"dim": 16, "weight": 10, "description": "Lead/follow interactions"},
                    "transition_flow": {"dim": 64, "weight": 5, "description": "Move sequence fluidity"}
                },
                "search_performance": {
                    "method": "qdrant_cloud" if engine and engine.is_qdrant_available else "in_memory",
                    "avg_search_time_ms": perf_stats.get("avg_search_time_ms", 0),
                    "total_searches": perf_stats.get("unified_searches", 0),
                    "cache_efficiency": f"{perf_stats.get('cache_hits', 0)}/{perf_stats.get('unified_searches', 1)}"
                }
            },
            "move_intelligence": move_analysis,
            "tempo_intelligence": tempo_analysis,
            "diversity_intelligence": diversity_analysis,
            "quality_scores": {
                "overall_match": _calculate_overall_match_score(move_analysis, tempo_analysis, diversity_analysis),
                "tempo_alignment": tempo_analysis.get("alignment_score", 0),
                "move_diversity": diversity_analysis.get("diversity_score", 0),
                "flow_quality": move_analysis.get("flow_score", 0)
            },
            "ai_recommendations": _generate_ai_recommendations(move_analysis, tempo_analysis, diversity_analysis)
        }
        
    except Exception as e:
        logger.warning(f"Error in comprehensive AI insights: {e}")
        return {"error": "Unable to generate comprehensive insights"}

def _analyze_move_patterns(moves_used: list) -> dict:
    """Analyze move patterns and transitions."""
    if not moves_used:
        return {"error": "No moves to analyze"}
    
    # Extract move categories and transitions
    categories = []
    transitions = []
    
    for i, move in enumerate(moves_used):
        video_path = move.get('video_path', '')
        if '/Bachata_steps/' in video_path:
            category = video_path.split('/Bachata_steps/')[1].split('/')[0]
            categories.append(category)
            
            if i > 0:
                prev_category = categories[i-1] if i-1 < len(categories) else None
                if prev_category:
                    transitions.append(f"{prev_category}→{category}")
    
    # Analyze patterns
    from collections import Counter
    category_counts = Counter(categories)
    transition_counts = Counter(transitions)
    
    # Calculate flow score based on transition variety
    unique_transitions = len(set(transitions))
    total_transitions = len(transitions)
    flow_score = unique_transitions / max(total_transitions, 1) if total_transitions > 0 else 0
    
    return {
        "total_moves": len(moves_used),
        "unique_categories": len(set(categories)),
        "category_distribution": dict(category_counts),
        "transition_patterns": dict(transition_counts.most_common(5)),
        "flow_score": round(flow_score, 3),
        "repetition_analysis": {
            "most_used_category": category_counts.most_common(1)[0] if category_counts else ("none", 0),
            "category_balance": "balanced" if len(set(categories)) > len(categories) * 0.4 else "repetitive"
        }
    }

def _analyze_tempo_matching(music_analysis: dict, moves_used: list) -> dict:
    """Analyze how well moves match the music tempo."""
    audio_tempo = music_analysis.get('tempo', 125)
    rhythm_strength = music_analysis.get('rhythm_strength', 0.5)
    syncopation = music_analysis.get('syncopation', 0.5)
    
    # Ideal bachata tempo range
    ideal_min, ideal_max = 110, 140
    tempo_variance = 0
    
    if audio_tempo < ideal_min:
        tempo_variance = (ideal_min - audio_tempo) / ideal_min
    elif audio_tempo > ideal_max:
        tempo_variance = (audio_tempo - ideal_max) / ideal_max
    
    alignment_score = max(0, 1 - tempo_variance)
    
    return {
        "audio_tempo_bpm": audio_tempo,
        "ideal_range": [ideal_min, ideal_max],
        "tempo_variance": round(tempo_variance, 3),
        "alignment_score": round(alignment_score, 3),
        "rhythm_strength": round(rhythm_strength, 3),
        "syncopation_level": round(syncopation, 3),
        "tempo_quality": "excellent" if alignment_score > 0.9 else "good" if alignment_score > 0.7 else "moderate",
        "recommendations": _get_tempo_recommendations(audio_tempo, rhythm_strength, syncopation)
    }

def _analyze_choreography_diversity(moves_used: list) -> dict:
    """Analyze the diversity and variety in the choreography."""
    if not moves_used:
        return {"diversity_score": 0, "analysis": "No moves to analyze"}
    
    # Extract categories
    categories = []
    for move in moves_used:
        video_path = move.get('video_path', '')
        if '/Bachata_steps/' in video_path:
            category = video_path.split('/Bachata_steps/')[1].split('/')[0]
            categories.append(category)
    
    if not categories:
        return {"diversity_score": 0, "analysis": "No valid move categories found"}
    
    # Calculate diversity metrics
    unique_categories = len(set(categories))
    total_moves = len(categories)
    diversity_score = unique_categories / total_moves
    
    # Analyze distribution
    from collections import Counter
    category_counts = Counter(categories)
    max_count = max(category_counts.values())
    min_count = min(category_counts.values())
    distribution_balance = 1 - ((max_count - min_count) / total_moves)
    
    return {
        "diversity_score": round(diversity_score, 3),
        "distribution_balance": round(distribution_balance, 3),
        "unique_categories": unique_categories,
        "total_moves": total_moves,
        "category_analysis": {
            "most_frequent": category_counts.most_common(1)[0],
            "least_frequent": category_counts.most_common()[-1],
            "balance_quality": "excellent" if distribution_balance > 0.8 else "good" if distribution_balance > 0.6 else "unbalanced"
        },
        "diversity_quality": "high" if diversity_score > 0.6 else "moderate" if diversity_score > 0.4 else "low"
    }

def _calculate_overall_match_score(move_analysis: dict, tempo_analysis: dict, diversity_analysis: dict) -> float:
    """Calculate an overall AI match score."""
    try:
        flow_score = move_analysis.get("flow_score", 0) * 0.3
        tempo_score = tempo_analysis.get("alignment_score", 0) * 0.4
        diversity_score = diversity_analysis.get("diversity_score", 0) * 0.3
        
        overall = flow_score + tempo_score + diversity_score
        return round(overall, 3)
    except:
        return 0.0

def _generate_ai_recommendations(move_analysis: dict, tempo_analysis: dict, diversity_analysis: dict) -> list:
    """Generate AI-powered recommendations for improvement."""
    recommendations = []
    
    # Tempo recommendations
    if tempo_analysis.get("alignment_score", 0) < 0.8:
        recommendations.append({
            "type": "tempo",
            "priority": "high",
            "message": f"Consider adjusting move selection for {tempo_analysis.get('audio_tempo_bpm', 0):.0f} BPM tempo",
            "suggestion": "Focus on moves that match the music's rhythm more closely"
        })
    
    # Diversity recommendations
    if diversity_analysis.get("diversity_score", 0) < 0.5:
        recommendations.append({
            "type": "diversity",
            "priority": "medium",
            "message": "Choreography could benefit from more move variety",
            "suggestion": "Try incorporating different move categories for better visual interest"
        })
    
    # Flow recommendations
    if move_analysis.get("flow_score", 0) < 0.6:
        recommendations.append({
            "type": "flow",
            "priority": "medium",
            "message": "Consider improving transition variety between moves",
            "suggestion": "Mix different move types to create smoother flow patterns"
        })
    
    return recommendations

def _get_tempo_recommendations(tempo: float, rhythm_strength: float, syncopation: float) -> list:
    """Get tempo-specific recommendations."""
    recommendations = []
    
    if tempo < 110:
        recommendations.append("Consider slower, more dramatic moves for this tempo")
    elif tempo > 140:
        recommendations.append("Focus on quick, energetic moves for this fast tempo")
    else:
        recommendations.append("Perfect tempo range for classic bachata moves")
    
    if rhythm_strength > 0.8:
        recommendations.append("Strong rhythm - emphasize beat-matching moves")
    
    if syncopation > 0.7:
        recommendations.append("High syncopation - incorporate complex timing variations")
    
    return recommendations

@app.get("/api/moves/stats")
async def get_move_statistics():
    """Get statistics about available moves for UI controls."""
    try:
        # Load annotation data
        annotation_path = Path("data/bachata_annotations.json")
        if not annotation_path.exists():
            raise HTTPException(status_code=404, detail="Annotation data not found")
        
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        clips = data.get('clips', [])
        
        # Calculate statistics
        from collections import Counter
        
        difficulty_counts = Counter(clip['difficulty'] for clip in clips)
        energy_counts = Counter(clip['energy_level'] for clip in clips)
        role_counts = Counter(clip['lead_follow_roles'] for clip in clips)
        move_counts = Counter(clip['move_label'] for clip in clips)
        
        tempos = [clip['estimated_tempo'] for clip in clips]
        
        return {
            "total_clips": len(clips),
            "difficulty": dict(difficulty_counts),
            "energy_level": dict(energy_counts),
            "role_focus": dict(role_counts),
            "move_types": dict(move_counts),
            "tempo": {
                "min": min(tempos),
                "max": max(tempos),
                "avg": round(sum(tempos) / len(tempos), 1)
            },
            "move_categories": data.get('move_categories', [])
        }
        
    except Exception as e:
        logger.error(f"Error getting move statistics: {e}")
        raise HTTPException(status_code=500, detail="Error getting move statistics")

@app.post("/api/moves/filter")
async def filter_moves(filters: Dict[str, Any]):
    """Filter moves based on criteria using SuperlinkedRecommendationEngine."""
    try:
        # Get SuperlinkedRecommendationEngine instance
        engine = get_superlinked_engine()
        
        # Create a dummy music features object for filtering (tempo-based)
        from app.services.music_analyzer import MusicFeatures
        import numpy as np
        
        # Use average tempo if tempo_range is provided, otherwise use default
        tempo_range = filters.get('tempo_range', [102, 150])
        target_tempo = (tempo_range[0] + tempo_range[1]) / 2 if tempo_range else 125.0
        
        # Create minimal music features for filtering
        dummy_music_features = MusicFeatures(
            tempo=target_tempo,
            beat_positions=np.array([]),
            duration=180.0,  # Default 3 minutes
            mfcc_features=np.zeros((13, 100)),
            chroma_features=np.zeros((12, 100)),
            spectral_centroid=np.zeros((1, 100)),
            zero_crossing_rate=np.zeros((1, 100)),
            rms_energy=np.zeros((1, 100)),
            harmonic_component=np.zeros((100,)),
            percussive_component=np.zeros((100,)),
            energy_profile=np.zeros((100,)),
            tempo_confidence=0.8,
            sections=[],
            rhythm_pattern_strength=0.7,
            syncopation_level=0.3,
            audio_embedding=np.zeros((128,))
        )
        
        # Use SuperlinkedRecommendationEngine for categorical filtering
        recommendations = engine.search_categorical_filtered_moves(
            music_features=dummy_music_features,
            energy_level=filters.get('energy_level', 'medium'),
            role_focus=filters.get('role_focus', 'both'),
            difficulty_level=filters.get('difficulty'),
            top_k=20  # Get more results for filtering
        )
        
        # Convert recommendations to the expected format
        filtered_clips = []
        for rec in recommendations:
            candidate = rec.move_candidate
            
            # Additional filtering for move types if specified
            if filters.get('move_types') and candidate.move_label not in filters['move_types']:
                continue
            
            # Additional tempo range filtering if specified
            if tempo_range:
                min_tempo, max_tempo = tempo_range
                if not (min_tempo <= candidate.tempo <= max_tempo):
                    continue
            
            filtered_clips.append({
                "clip_id": candidate.move_id,
                "move_label": candidate.move_label,
                "difficulty": candidate.difficulty_score,
                "energy_level": candidate.energy_level,
                "role_focus": candidate.role_focus,
                "tempo": candidate.tempo,
                "notes": candidate.notes,
                "similarity_score": rec.similarity_score
            })
        
        # Limit to first 10 for preview
        filtered_clips = filtered_clips[:10]
        
        return {
            "total_matches": len(filtered_clips),
            "clips": filtered_clips,
            "engine_used": "SuperlinkedRecommendationEngine"
        }
        
    except Exception as e:
        logger.error(f"Error filtering moves with SuperlinkedRecommendationEngine: {e}")
        # Fallback to original annotation-based filtering
        try:
            annotation_path = Path("data/bachata_annotations.json")
            if not annotation_path.exists():
                raise HTTPException(status_code=404, detail="Annotation data not found")
            
            with open(annotation_path, 'r') as f:
                data = json.load(f)
            
            clips = data.get('clips', [])
            filtered_clips = []
            
            for clip in clips:
                # Apply filters
                if filters.get('difficulty') and clip['difficulty'] != filters['difficulty']:
                    continue
                if filters.get('energy_level') and clip['energy_level'] != filters['energy_level']:
                    continue
                if filters.get('role_focus') and clip['lead_follow_roles'] != filters['role_focus']:
                    continue
                if filters.get('move_types') and clip['move_label'] not in filters['move_types']:
                    continue
                
                # Tempo range filter
                tempo_range = filters.get('tempo_range')
                if tempo_range:
                    min_tempo, max_tempo = tempo_range
                    if not (min_tempo <= clip['estimated_tempo'] <= max_tempo):
                        continue
                
                filtered_clips.append({
                    "clip_id": clip['clip_id'],
                    "move_label": clip['move_label'],
                    "difficulty": clip['difficulty'],
                    "energy_level": clip['energy_level'],
                    "role_focus": clip['lead_follow_roles'],
                    "tempo": clip['estimated_tempo'],
                    "notes": clip.get('notes', '')
                })
            
            return {
                "total_matches": len(filtered_clips),
                "clips": filtered_clips[:10],
                "engine_used": "fallback_annotation_based"
            }
            
        except Exception as fallback_error:
            logger.error(f"Fallback filtering also failed: {fallback_error}")
            raise HTTPException(status_code=500, detail="Error filtering moves")

@app.post("/api/moves/preview")
async def preview_moves(filters: Dict[str, Any]):
    """Get a preview of 3-5 sample moves using SuperlinkedRecommendationEngine with diversity."""
    try:
        # Get SuperlinkedRecommendationEngine instance
        engine = get_superlinked_engine()
        
        # Create a dummy music features object for preview
        from app.services.music_analyzer import MusicFeatures
        import numpy as np
        
        # Use average tempo if tempo_range is provided, otherwise use default
        tempo_range = filters.get('tempo_range', [102, 150])
        target_tempo = (tempo_range[0] + tempo_range[1]) / 2 if tempo_range else 125.0
        
        # Create minimal music features for preview
        dummy_music_features = MusicFeatures(
            tempo=target_tempo,
            beat_positions=np.array([]),
            duration=180.0,  # Default 3 minutes
            mfcc_features=np.zeros((13, 100)),
            chroma_features=np.zeros((12, 100)),
            spectral_centroid=np.zeros((1, 100)),
            zero_crossing_rate=np.zeros((1, 100)),
            rms_energy=np.zeros((1, 100)),
            harmonic_component=np.zeros((100,)),
            percussive_component=np.zeros((100,)),
            energy_profile=np.zeros((100,)),
            tempo_confidence=0.8,
            sections=[],
            rhythm_pattern_strength=0.7,
            syncopation_level=0.3,
            audio_embedding=np.zeros((128,))
        )
        
        # Use SuperlinkedRecommendationEngine with high diversity for preview
        recommendations = engine.recommend_moves(
            music_features=dummy_music_features,
            target_difficulty=filters.get('difficulty', 'intermediate'),
            target_energy=filters.get('energy_level', 'medium'),
            role_focus=filters.get('role_focus', 'both'),
            description=f"preview moves for {filters.get('energy_level', 'medium')} energy",
            top_k=15,  # Get more results for diversity selection
            diversity_factor=0.8,  # High diversity for preview
            randomization_seed=None  # Random seed for variety
        )
        
        # Convert to preview format and apply additional filters
        preview_clips = []
        for rec in recommendations:
            candidate = rec.move_candidate
            
            # Additional filtering for move types if specified
            if filters.get('move_types') and candidate.move_label not in filters['move_types']:
                continue
            
            # Additional tempo range filtering if specified
            if tempo_range:
                min_tempo, max_tempo = tempo_range
                if not (min_tempo <= candidate.tempo <= max_tempo):
                    continue
            
            preview_clips.append({
                "clip_id": candidate.move_id,
                "move_label": candidate.move_label,
                "difficulty": candidate.difficulty_score,
                "energy_level": candidate.energy_level,
                "role_focus": candidate.role_focus,
                "tempo": candidate.tempo,
                "notes": candidate.notes,
                "similarity_score": rec.similarity_score,
                "explanation": rec.explanation
            })
            
            # Limit to 5 for preview
            if len(preview_clips) >= 5:
                break
        
        # Get total available count using filter endpoint
        try:
            filter_result = await filter_moves(filters)
            total_available = filter_result['total_matches']
        except Exception:
            total_available = len(preview_clips)
        
        return {
            "preview_clips": preview_clips,
            "total_available": total_available,
            "engine_used": "SuperlinkedRecommendationEngine",
            "diversity_applied": True
        }
        
    except Exception as e:
        logger.error(f"Error previewing moves with SuperlinkedRecommendationEngine: {e}")
        # Fallback to filter-based preview
        try:
            filter_result = await filter_moves(filters)
            clips = filter_result['clips']
            
            # Select diverse sample moves (max 5)
            import random
            sample_size = min(5, len(clips))
            
            if sample_size > 0:
                # Try to get diverse moves by type if possible
                move_types = list(set(clip['move_label'] for clip in clips))
                sample_clips = []
                
                # Get one from each type first
                for move_type in move_types[:sample_size]:
                    type_clips = [c for c in clips if c['move_label'] == move_type]
                    if type_clips:
                        sample_clips.append(random.choice(type_clips))
                
                # Fill remaining slots randomly
                while len(sample_clips) < sample_size and len(sample_clips) < len(clips):
                    remaining_clips = [c for c in clips if c not in sample_clips]
                    if remaining_clips:
                        sample_clips.append(random.choice(remaining_clips))
                
                return {
                    "preview_clips": sample_clips,
                    "total_available": filter_result['total_matches'],
                    "engine_used": "fallback_filter_based"
                }
            else:
                return {
                    "preview_clips": [],
                    "total_available": 0,
                    "engine_used": "fallback_filter_based"
                }
                
        except Exception as fallback_error:
            logger.error(f"Fallback preview also failed: {fallback_error}")
            raise HTTPException(status_code=500, detail="Error previewing moves")

@app.get("/api/query-templates")
async def get_query_templates():
    """Get predefined query templates validated with SuperlinkedRecommendationEngine."""
    try:
        # Get SuperlinkedRecommendationEngine instance for validation
        engine = get_superlinked_engine()
        
        # Define base templates with SuperlinkedRecommendationEngine-optimized parameters
        base_templates = [
            {
                "name": "Show me beginner moves",
                "description": "Easy moves perfect for learning",
                "filters": {
                    "difficulty": "beginner"
                },
                "superlinked_query": "basic beginner moves for learning"
            },
            {
                "name": "High energy advanced moves",
                "description": "Dynamic moves for experienced dancers",
                "filters": {
                    "difficulty": "advanced",
                    "energy_level": "high"
                },
                "superlinked_query": "energetic advanced moves for experienced dancers"
            },
            {
                "name": "Slow romantic moves",
                "description": "Gentle moves for slower songs",
                "filters": {
                    "energy_level": "low",
                    "tempo_range": [102, 115]
                },
                "superlinked_query": "gentle romantic moves for slow songs"
            },
            {
                "name": "Lead-focused moves",
                "description": "Moves that highlight the leader",
                "filters": {
                    "role_focus": "lead_focus"
                },
                "superlinked_query": "moves that showcase the lead dancer"
            },
            {
                "name": "Follow styling moves",
                "description": "Moves that showcase the follower",
                "filters": {
                    "role_focus": "follow_focus"
                },
                "superlinked_query": "styling moves that highlight the follow"
            },
            {
                "name": "Fast tempo moves",
                "description": "Moves for high-energy songs",
                "filters": {
                    "tempo_range": [135, 150]
                },
                "superlinked_query": "fast moves for high tempo songs"
            },
            {
                "name": "Intermediate balanced moves",
                "description": "Well-rounded moves for intermediate dancers",
                "filters": {
                    "difficulty": "intermediate",
                    "energy_level": "medium"
                },
                "superlinked_query": "balanced intermediate moves for practice"
            },
            {
                "name": "Turn-based moves",
                "description": "Moves featuring turns and spins",
                "filters": {
                    "move_types": ["lady_right_turn", "lady_left_turn"]
                },
                "superlinked_query": "turn moves with spins and rotations"
            }
        ]
        
        # Validate templates using SuperlinkedRecommendationEngine
        validated_templates = []
        
        for template in base_templates:
            try:
                # Test the template by running a quick filter
                test_result = await filter_moves(template["filters"])
                available_moves = test_result.get("total_matches", 0)
                
                # Add validation info to template
                template_with_validation = template.copy()
                template_with_validation.update({
                    "available_moves": available_moves,
                    "validated": available_moves > 0,
                    "engine_validated": True
                })
                
                validated_templates.append(template_with_validation)
                
            except Exception as e:
                logger.warning(f"Template validation failed for '{template['name']}': {e}")
                # Include template but mark as not validated
                template_with_validation = template.copy()
                template_with_validation.update({
                    "available_moves": 0,
                    "validated": False,
                    "engine_validated": False,
                    "validation_error": str(e)
                })
                validated_templates.append(template_with_validation)
        
        return {
            "templates": validated_templates,
            "engine_used": "SuperlinkedRecommendationEngine",
            "validation_performed": True
        }
        
    except Exception as e:
        logger.error(f"Error getting query templates with SuperlinkedRecommendationEngine: {e}")
        # Fallback to basic templates
        fallback_templates = [
            {
                "name": "Show me beginner moves",
                "description": "Easy moves perfect for learning",
                "filters": {
                    "difficulty": "beginner"
                },
                "validated": False,
                "engine_validated": False
            },
            {
                "name": "High energy advanced moves",
                "description": "Dynamic moves for experienced dancers",
                "filters": {
                    "difficulty": "advanced",
                    "energy_level": "high"
                },
                "validated": False,
                "engine_validated": False
            },
            {
                "name": "Slow romantic moves",
                "description": "Gentle moves for slower songs",
                "filters": {
                    "energy_level": "low",
                    "tempo_range": [102, 115]
                },
                "validated": False,
                "engine_validated": False
            }
        ]
        
        return {
            "templates": fallback_templates,
            "engine_used": "fallback",
            "validation_performed": False,
            "error": str(e)
        }

async def generate_choreography_task_safe(
    task_id: str,
    youtube_url: str,
    difficulty: str,
    energy_level: Optional[str],
    quality_mode: str,
    role_focus: Optional[str] = None,
    move_types: Optional[List[str]] = None,
    tempo_range: Optional[List[int]] = None
):
    """
    Safe wrapper for choreography generation with comprehensive error handling.
    """
    try:
        await generate_choreography_task(
            task_id, youtube_url, difficulty, energy_level, quality_mode,
            role_focus, move_types, tempo_range
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
    quality_mode: str,
    role_focus: Optional[str] = None,
    move_types: Optional[List[str]] = None,
    tempo_range: Optional[List[int]] = None
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
        
        # Get pipeline with specified quality mode and Qdrant Cloud configuration
        config = PipelineConfig(
            quality_mode=quality_mode,
            enable_caching=True,
            max_workers=4,
            cleanup_after_generation=True,
            enable_qdrant=True,
            auto_populate_qdrant=True
        )
        task_pipeline = ChoreoGenerationPipeline(config)
        
        # Progress tracking with SuperlinkedRecommendationEngine and Qdrant Cloud integration
        progress_stages = [
            (10, "downloading", "Downloading audio from YouTube..."),
            (25, "analyzing", "Analyzing musical structure and tempo..."),
            (35, "embedding", "Generating unified Superlinked embeddings..."),
            (50, "searching", "Searching Qdrant Cloud for compatible moves..."),
            (65, "selecting", "Selecting diverse moves with unified vector similarity..."),
            (80, "generating", "Generating choreography video..."),
            (95, "finalizing", "Finalizing video and metadata...")
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
            energy_level=energy_level,
            role_focus=role_focus,
            move_types=move_types,
            tempo_range=tempo_range
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
                    "video_filename": Path(result.output_path).name if result.output_path else None,
                    "qdrant_enabled": result.qdrant_enabled,
                    "qdrant_embeddings_stored": result.qdrant_embeddings_stored,
                    "qdrant_embeddings_retrieved": result.qdrant_embeddings_retrieved,
                    "qdrant_search_time": result.qdrant_search_time
                }
            })
            
            # Update Qdrant and Superlinked metrics in task
            if hasattr(result, 'qdrant_enabled') and result.qdrant_enabled:
                active_tasks[task_id]["qdrant_metrics"].update({
                    "embeddings_stored": result.qdrant_embeddings_stored,
                    "embeddings_retrieved": result.qdrant_embeddings_retrieved,
                    "search_time_ms": result.qdrant_search_time * 1000  # Convert to ms
                })
            
            # Add SuperlinkedRecommendationEngine metrics
            try:
                engine = get_superlinked_engine()
                perf_stats = engine.get_performance_stats()
                active_tasks[task_id]["superlinked_metrics"] = {
                    "unified_searches": perf_stats.get("unified_searches", 0),
                    "avg_search_time_ms": perf_stats.get("avg_search_time_ms", 0.0),
                    "natural_language_queries": perf_stats.get("natural_language_queries", 0),
                    "qdrant_searches": perf_stats.get("qdrant_searches", 0),
                    "fallback_searches": perf_stats.get("fallback_searches", 0),
                    "cache_hits": perf_stats.get("cache_hits", 0)
                }
            except Exception as e:
                logger.warning(f"Failed to get SuperlinkedRecommendationEngine metrics: {e}")
                active_tasks[task_id]["superlinked_metrics"] = {"error": str(e)}
            
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

# Qdrant Admin Endpoints

@app.get("/api/admin/qdrant/status")
async def get_qdrant_status():
    """Get detailed Qdrant status and statistics."""
    try:
        await check_qdrant_connection()
        
        if not qdrant_service:
            return {
                "error": "Qdrant service not initialized",
                "connected": False
            }
        
        # Get comprehensive status
        health_status = qdrant_service.health_check()
        collection_info = qdrant_service.get_collection_info()
        stats = qdrant_service.get_statistics()
        
        return {
            "connection_status": {
                "connected": qdrant_connection_status["connected"],
                "last_check": qdrant_connection_status["last_check"],
                "reconnect_attempts": qdrant_connection_status["reconnect_attempts"],
                "max_reconnect_attempts": qdrant_connection_status["max_reconnect_attempts"]
            },
            "health": health_status,
            "collection": collection_info,
            "statistics": {
                "total_points": stats.total_points,
                "search_requests": stats.search_requests,
                "avg_search_time_ms": round(stats.avg_search_time_ms, 2),
                "cache_hits": stats.cache_hits,
                "collection_size_mb": round(stats.collection_size_mb, 2)
            },
            "config": {
                "host": qdrant_service.config.host,
                "port": qdrant_service.config.port,
                "collection_name": qdrant_service.config.collection_name,
                "vector_size": qdrant_service.config.vector_size,
                "distance_metric": qdrant_service.config.distance_metric
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting Qdrant status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving Qdrant status: {str(e)}")

@app.post("/api/admin/qdrant/reconnect")
async def force_qdrant_reconnect():
    """Force Qdrant reconnection."""
    try:
        global qdrant_service, qdrant_connection_status
        
        # Reset connection status
        qdrant_connection_status["reconnect_attempts"] = 0
        qdrant_connection_status["last_check"] = 0
        
        # Force reconnection
        qdrant_service = get_qdrant_service()
        connection_result = await check_qdrant_connection()
        
        return {
            "success": connection_result,
            "message": "Reconnection successful" if connection_result else "Reconnection failed",
            "connection_status": qdrant_connection_status
        }
        
    except Exception as e:
        logger.error(f"Error forcing Qdrant reconnection: {e}")
        raise HTTPException(status_code=500, detail=f"Error reconnecting to Qdrant: {str(e)}")

@app.get("/api/admin/qdrant/collection/info")
async def get_collection_info():
    """Get detailed collection information."""
    try:
        await check_qdrant_connection()
        
        if not qdrant_service or not qdrant_connection_status["connected"]:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        collection_info = qdrant_service.get_collection_info()
        
        return {
            "collection_info": collection_info,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving collection info: {str(e)}")

@app.post("/api/admin/qdrant/collection/clear")
async def clear_qdrant_collection():
    """Clear all points from the Qdrant collection."""
    try:
        await check_qdrant_connection()
        
        if not qdrant_service or not qdrant_connection_status["connected"]:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        # Get current count before clearing
        collection_info = qdrant_service.get_collection_info()
        points_before = collection_info.get("points_count", 0)
        
        # Clear collection
        success = qdrant_service.clear_collection()
        
        if success:
            return {
                "success": True,
                "message": f"Collection cleared successfully. Removed {points_before} points.",
                "points_removed": points_before
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear collection")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing collection: {str(e)}")

@app.get("/api/admin/qdrant/moves/{move_id}")
async def get_move_from_qdrant(move_id: str):
    """Get a specific move from Qdrant by ID."""
    try:
        await check_qdrant_connection()
        
        if not qdrant_service or not qdrant_connection_status["connected"]:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        result = qdrant_service.get_move_by_id(move_id)
        
        if result:
            return {
                "found": True,
                "move": {
                    "move_id": result.move_id,
                    "score": result.score,
                    "metadata": result.metadata
                }
            }
        else:
            raise HTTPException(status_code=404, detail=f"Move {move_id} not found in Qdrant")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting move from Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving move: {str(e)}")

@app.delete("/api/admin/qdrant/moves/{move_id}")
async def delete_move_from_qdrant(move_id: str):
    """Delete a specific move from Qdrant by ID."""
    try:
        await check_qdrant_connection()
        
        if not qdrant_service or not qdrant_connection_status["connected"]:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        success = qdrant_service.delete_move(move_id)
        
        if success:
            return {
                "success": True,
                "message": f"Move {move_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Move {move_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting move from Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting move: {str(e)}")

@app.post("/api/admin/qdrant/search/test")
async def test_qdrant_search(request: dict):
    """Test Qdrant search functionality with a random query."""
    try:
        await check_qdrant_connection()
        
        if not qdrant_service or not qdrant_connection_status["connected"]:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        # Get search parameters
        limit = request.get("limit", 5)
        tempo_range = request.get("tempo_range")
        difficulty = request.get("difficulty")
        energy_level = request.get("energy_level")
        
        # Create a random query vector
        import numpy as np
        query_vector = np.random.random(qdrant_service.config.vector_size)
        
        # Perform search
        start_time = time.time()
        results = qdrant_service.search_similar_moves(
            query_embedding=query_vector,
            limit=limit,
            tempo_range=tuple(tempo_range) if tempo_range and len(tempo_range) == 2 else None,
            difficulty=difficulty,
            energy_level=energy_level
        )
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "success": True,
            "search_time_ms": round(search_time, 2),
            "results_count": len(results),
            "results": [
                {
                    "move_id": r.move_id,
                    "score": round(r.score, 4),
                    "move_label": r.metadata.get("move_label"),
                    "difficulty": r.metadata.get("difficulty"),
                    "energy_level": r.metadata.get("energy_level"),
                    "estimated_tempo": r.metadata.get("estimated_tempo")
                }
                for r in results
            ],
            "query_params": {
                "limit": limit,
                "tempo_range": tempo_range,
                "difficulty": difficulty,
                "energy_level": energy_level
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing Qdrant search: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing search: {str(e)}")

@app.get("/api/admin/qdrant/metrics")
async def get_qdrant_metrics():
    """Get comprehensive Qdrant performance metrics."""
    try:
        await check_qdrant_connection()
        
        if not qdrant_service:
            return {"error": "Qdrant service not initialized"}
        
        stats = qdrant_service.get_statistics()
        collection_info = qdrant_service.get_collection_info()
        
        # Calculate additional metrics
        current_time = time.time()
        uptime_hours = (current_time - qdrant_connection_status.get("startup_time", current_time)) / 3600
        
        return {
            "performance": {
                "total_search_requests": stats.search_requests,
                "average_search_time_ms": round(stats.avg_search_time_ms, 2),
                "cache_hits": stats.cache_hits,
                "cache_hit_rate": round(stats.cache_hits / max(stats.search_requests, 1) * 100, 2)
            },
            "storage": {
                "total_points": stats.total_points,
                "collection_size_mb": round(stats.collection_size_mb, 2),
                "indexed_vectors": collection_info.get("indexed_vectors_count", 0),
                "vector_size": collection_info.get("vector_size", 0)
            },
            "connection": {
                "connected": qdrant_connection_status["connected"],
                "reconnect_attempts": qdrant_connection_status["reconnect_attempts"],
                "uptime_hours": round(uptime_hours, 2),
                "last_health_check": qdrant_connection_status["last_check"]
            },
            "timestamp": current_time
        }
        
    except Exception as e:
        logger.error(f"Error getting Qdrant metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
