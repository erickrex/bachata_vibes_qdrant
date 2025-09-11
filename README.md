# Bachata Choreography Generator

**Product Vision**: Democratize bachata choreography creation by enabling anyone to generate professional-quality dance videos from any bachata song using AI-powered movement analysis and video assembly.

## 🎯 Product Overview

Transform any bachata song into a complete choreographed dance video. Upload your favorite bachata track, and our AI analyzes the music's tempo, energy, and rhythm to intelligently select and sequence dance moves from our curated library of professional clips. The result: a full-length, synchronized dance video ready to share.

## ✨ Core Features

### 🎵 **Intelligent Music Analysis**
- Automatic tempo, beat, and energy detection
- Musical phrase recognition for natural choreography flow
- Support for all bachata sub-genres (Traditional, Sensual, Modern)

### 🕺 **AI-Powered Choreography Generation** 
- 34+ professional dance clips across 11+ move categories
- Smart move selection based on song characteristics
- Diversity algorithm ensures varied, engaging sequences
- Difficulty progression from beginner to advanced moves

### 🎬 **Professional Video Assembly**
- Automatic frame rate and resolution normalization
- Seamless clip transitions without frozen sections
- Full-song duration support (3+ minutes)
- High-quality output (1920x1080, 30fps)

### 📊 **Comprehensive Metadata Export**
- Detailed timeline with start/end times for each move
- Complete choreography breakdown and statistics
- JSON export for integration with other tools

## 🏗️ Technical Architecture

### **Backend (FastAPI)**
- **Audio Processing**: Librosa-based music analysis and feature extraction
- **Pose Detection**: MediaPipe integration for movement analysis  
- **Video Assembly**: FFmpeg-powered professional video generation
- **API Layer**: RESTful endpoints with real-time WebSocket updates

### **Frontend (HTMX + Alpine.js)**
- **Zero Build Process**: Pure HTML templates with server-side rendering
- **Interactive UI**: Dynamic forms and real-time updates via HTMX
- **State Management**: Alpine.js for client-side reactivity
- **Modern Design**: Tailwind CSS with responsive layouts


### Utils

### 1. **Upload Your Song**
```bash
# Add bachata songs to the system
uv run python song_downloader/get_songs.py  # Download from YouTube
# Or manually add MP3/WAV files to data/test_songs/
```

### 2. **Generate Choreography**
```bash
# Create a complete choreography video
uv run python MVP/complete_pipeline.py data/test_songs/your_song.mp3 --output my_choreography.mp4

# For full song duration (auto-detected)
uv run python MVP/full_song_choreography.py data/test_songs/your_song.mp3
```


## 🎨 Tech Stack

### **Core Technologies**
- **FastAPI**: High-performance Python web framework
- **HTMX**: Modern HTML-driven interactivity  
- **Alpine.js**: Lightweight JavaScript framework
- **Tailwind CSS**: Utility-first CSS framework

### **AI & Media Processing**
- **Librosa**: Advanced audio analysis and feature extraction
- **MediaPipe**: Real-time pose detection and movement analysis
- **FFmpeg**: Professional video processing and assembly
- **NumPy/Pandas**: Scientific computing and data analysis

### **Development & Deployment**
- **uv**: Ultra-fast Python package management
- **Docker**: Containerized development and deployment
- **Jinja2**: Server-side HTML templating


## 📄 License

This project is for educational and demonstration purposes. Please respect music copyrights and dance creator rights when using this system. Developed for the AWS Kiro Hackathon :) 

## 🎯 Success Criteria

### **MVP Achieved ✅**
- [x] Functional AI choreography generation from any bachata song
- [x] Professional video output with seamless transitions  
- [x] Comprehensive metadata and timeline export
- [x] Support for full-song duration (3+ minutes)
- [x] 89.5% processing success rate with diverse move selection

### **Production Ready Targets**
- [ ] 95%+ processing success rate
- [ ] Sub-60 second generation time for 3-minute videos
- [ ] 50+ unique move types across multiple bachata styles
- [ ] Web interface with real-time progress tracking
- [ ] Cloud deployment with scalable video processing

---

**Status**: ✅ **MVP Complete** - Functional AI-powered bachata choreography generator ready for production enhancement and user testing.