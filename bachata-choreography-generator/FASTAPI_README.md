# Bachata Choreography Generator - FastAPI Web Application

This document describes the FastAPI web application for the Bachata Choreography Generator, which provides a complete web interface for generating dance choreographies from YouTube videos.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- All dependencies installed via `uv` (see main README)
- Move clips and annotations data in place

### Starting the Server

```bash
# Option 1: Using the startup script (recommended)
python start_server.py

# Option 2: Direct uvicorn command
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Python module
python main.py
```

The server will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📋 API Endpoints

### Core Endpoints

#### `GET /`
- **Description**: Main web interface
- **Returns**: HTML page with choreography generation form
- **Features**: Real-time progress tracking, video playback

#### `POST /api/choreography`
- **Description**: Start choreography generation
- **Request Body**:
  ```json
  {
    "youtube_url": "https://www.youtube.com/watch?v=...",
    "difficulty": "intermediate",
    "energy_level": null,
    "quality_mode": "balanced"
  }
  ```
- **Response**:
  ```json
  {
    "task_id": "uuid-string",
    "status": "started",
    "message": "Choreography generation started..."
  }
  ```

#### `GET /api/task/{task_id}`
- **Description**: Get task status and progress
- **Response**:
  ```json
  {
    "task_id": "uuid-string",
    "status": "running",
    "progress": 45,
    "stage": "analyzing",
    "message": "Analyzing musical structure...",
    "result": null,
    "error": null
  }
  ```

#### `GET /api/video/{filename}`
- **Description**: Serve generated choreography videos
- **Features**: Proper streaming headers, security validation
- **Returns**: MP4 video file

### Utility Endpoints

#### `GET /health`
- **Description**: Comprehensive health check
- **Returns**: System status, resource usage, service availability

#### `GET /api/tasks`
- **Description**: List all active tasks
- **Features**: Automatic cleanup of old tasks

#### `DELETE /api/task/{task_id}`
- **Description**: Remove task from tracking

#### `GET /api/system/status`
- **Description**: Detailed system diagnostics

#### `POST /api/validate/youtube`
- **Description**: Validate YouTube URL without starting generation

## 🎛️ Configuration Options

### Request Parameters

#### `difficulty`
- **Options**: `"beginner"`, `"intermediate"`, `"advanced"`
- **Default**: `"intermediate"`
- **Description**: Target difficulty level for selected moves

#### `quality_mode`
- **Options**: `"fast"`, `"balanced"`, `"high_quality"`
- **Default**: `"balanced"`
- **Description**: Processing quality vs speed trade-off
  - `fast`: 8 moves, 720p, faster processing
  - `balanced`: 12 moves, 720p, good quality
  - `high_quality`: 20 moves, 1080p, best quality

#### `energy_level`
- **Options**: `"low"`, `"medium"`, `"high"`, `null`
- **Default**: `null` (auto-detect)
- **Description**: Target energy level for choreography

## 🛡️ Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "User-friendly error message",
    "details": {
      "additional": "context"
    }
  }
}
```

### Error Categories

#### `YOUTUBE_DOWNLOAD_ERROR`
- Invalid URL format
- Video unavailable or restricted
- Network connectivity issues

#### `MUSIC_ANALYSIS_ERROR`
- Unsupported audio format
- File corruption
- Audio too short/long

#### `VIDEO_GENERATION_ERROR`
- FFmpeg encoding issues
- Insufficient suitable moves
- Synchronization problems

#### `VALIDATION_ERROR`
- Invalid input parameters
- Missing required fields

#### `RESOURCE_ERROR`
- Insufficient disk space
- Memory limitations
- System overload

## 🔒 Security Features

### Input Validation
- YouTube URL format validation
- Parameter type and range checking
- File path sanitization

### File Serving Security
- Path traversal prevention
- File type restrictions
- Size limitations
- Directory access controls

### Rate Limiting
- Maximum 3 concurrent generations
- Automatic task cleanup
- Resource monitoring

## 📊 Monitoring & Diagnostics

### Health Monitoring
The `/health` endpoint provides:
- System resource usage (memory, disk)
- Service availability status
- Active task counts
- Pipeline configuration

### Logging
- Structured logging with timestamps
- Error tracking with stack traces
- Performance metrics
- User action auditing

### Task Management
- Automatic cleanup of old tasks (1 hour)
- Progress tracking with detailed stages
- Error categorization and user-friendly messages

## 🎨 Web Interface Features

### User Experience
- Responsive design for mobile/desktop
- Real-time progress updates
- Drag-and-drop URL input
- Video player with controls

### Progress Tracking
- Visual progress bar
- Stage-specific messages
- Estimated completion time
- Error recovery options

### Video Playback
- HTML5 video player
- Streaming support
- Quality adaptation
- Download options

## 🔧 Development

### Testing
```bash
# Run application tests
python test_app.py

# Test specific components
python -c "from main import app; print('App loaded successfully')"
```

### Adding New Endpoints
1. Add route handler to `main.py`
2. Include appropriate error handling
3. Add validation if needed
4. Update this documentation

### Custom Error Types
1. Define in `app/exceptions.py`
2. Add user-friendly messages
3. Include in exception handlers
4. Test error scenarios

## 📁 File Structure

```
bachata-choreography-generator/
├── main.py                 # FastAPI application
├── app/
│   ├── exceptions.py       # Custom exceptions
│   ├── validation.py       # Input validation
│   ├── services/          # Business logic services
│   ├── templates/         # Jinja2 HTML templates
│   └── static/           # CSS, JS, images
├── data/
│   ├── temp/             # Temporary files
│   ├── output/           # Generated videos
│   └── cache/            # Service cache
├── start_server.py        # Server startup script
└── test_app.py           # Application tests
```

## 🚨 Troubleshooting

### Common Issues

#### "System requirements not met"
- Check disk space (500MB+ required)
- Verify move clips are available
- Ensure all directories are writable

#### "YouTube download failed"
- Verify URL format
- Check internet connectivity
- Try a different video

#### "Video generation failed"
- Check FFmpeg installation
- Verify sufficient disk space
- Review logs for specific errors

#### "Service unavailable"
- Check system resources
- Restart the application
- Review error logs

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
python main.py
```

### Performance Optimization
- Enable caching for repeated requests
- Adjust quality mode based on requirements
- Monitor system resources
- Clean up old files regularly

## 📈 Scaling Considerations

### Current Limitations
- Single-server deployment
- File-based storage
- In-memory task tracking

### Future Enhancements
- Database integration
- Distributed task queue
- CDN for video delivery
- Load balancing support

## 🤝 Contributing

When adding new features:
1. Follow existing error handling patterns
2. Add comprehensive validation
3. Include user-friendly error messages
4. Update documentation
5. Add tests for new functionality

## 📞 Support

For issues or questions:
1. Check the logs for detailed error information
2. Verify system requirements
3. Test with the health check endpoint
4. Review this documentation

The FastAPI application provides a robust, user-friendly interface for the Bachata Choreography Generator with comprehensive error handling, security features, and monitoring capabilities.