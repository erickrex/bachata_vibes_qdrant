# Bachata Choreography Generator

AI-powered Bachata choreography generator that creates dance sequences from YouTube music.

## Features

- Analyzes YouTube music videos for tempo and musical structure
- Selects appropriate dance moves from a curated collection of 40 Bachata clips
- Generates synchronized choreography videos with smooth transitions
- Web-based interface with real-time progress tracking

## Setup

1. Install UV (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone and setup the project:
   ```bash
   cd bachata-choreography-generator
   uv sync
   ```

3. Run the development server:
   ```bash
   uv run uvicorn main:app --reload
   ```

4. Open http://localhost:8000 in your browser

## Project Structure

```
bachata-choreography-generator/
├── app/
│   ├── services/          # Core business logic
│   ├── models/           # Data models and schemas
│   ├── static/           # CSS, JS files
│   └── templates/        # HTML templates
├── data/
│   ├── move_clips/       # Bachata move video clips
│   ├── temp/            # Temporary processing files
│   └── generated/       # Generated choreography videos
├── tests/               # Test suite
└── main.py             # FastAPI application entry point
```

## Dependencies

- FastAPI - Web framework
- Librosa - Audio analysis
- MediaPipe - Pose detection and movement analysis
- yt-dlp - YouTube audio download
- OpenCV - Video processing
- Qdrant - Vector database for similarity search