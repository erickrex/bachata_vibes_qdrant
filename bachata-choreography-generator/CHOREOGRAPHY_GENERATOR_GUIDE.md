# 🎵 Bachata Choreography Generator Guide

The universal Bachata choreography generator that creates dance videos from existing songs or YouTube URLs with customizable duration and quality settings.

## 🚀 Quick Start

### Activate Environment
```bash
# Use UV to run with the virtual environment
uv run python create_bachata_choreography.py [options]
```

### Basic Usage Examples

```bash
# List available songs
uv run python create_bachata_choreography.py --list-songs

# Generate 30-second choreography for Chayanne (fast quality)
uv run python create_bachata_choreography.py "Chayanne" --duration 30s --quality fast

# Generate 1-minute choreography for Romeo Santos (default settings)
uv run python create_bachata_choreography.py "Romeo Santos"

# Generate full-length choreography with high quality
uv run python create_bachata_choreography.py "Aventura" --duration full --quality high

# Download from YouTube and create choreography
uv run python create_bachata_choreography.py "https://www.youtube.com/watch?v=abc123" --duration 1min
```

## 📋 Command Line Options

### Required Arguments
- `song`: Song name (partial match) or YouTube URL

### Optional Arguments
- `--duration, -d`: Video duration
  - `30s`: 30-second video
  - `1min`: 1-minute video (default)
  - `full`: Full song length
  
- `--quality, -q`: Processing quality
  - `fast`: Faster processing, lower quality (default)
  - `high`: Slower processing, higher quality
  
- `--list-songs, -l`: List available songs and exit

## 🎵 Song Input Options

### 1. Existing Songs (Partial Match)
The script will search for songs in `data/songs/` directory:

```bash
# These all work for "Chayanne - Bailando Bachata (Letra⧸Lyrics).mp3"
uv run python create_bachata_choreography.py "Chayanne"
uv run python create_bachata_choreography.py "Bailando"
uv run python create_bachata_choreography.py "Chayanne - Bailando"
```

### 2. YouTube URLs
Download and process songs directly from YouTube:

```bash
# Full YouTube URLs
uv run python create_bachata_choreography.py "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python create_bachata_choreography.py "https://youtu.be/VIDEO_ID"

# Shortened URLs (will add https:// automatically)
uv run python create_bachata_choreography.py "youtube.com/watch?v=VIDEO_ID"
uv run python create_bachata_choreography.py "youtu.be/VIDEO_ID"
```

## ⚙️ Quality Settings

### Fast Mode (Default)
- **Processing Time**: ~1-2 minutes
- **Video Resolution**: 1280x720 (HD)
- **Video Bitrate**: 4M
- **Audio Bitrate**: 128k
- **Move Analysis**: 8 moves, 10 FPS sampling
- **Best For**: Quick previews, testing, demos

### High Quality Mode
- **Processing Time**: ~5-10 minutes
- **Video Resolution**: 1920x1080 (Full HD)
- **Video Bitrate**: 8M
- **Audio Bitrate**: 320k
- **Move Analysis**: 15 moves, 30 FPS sampling
- **Best For**: Final videos, presentations, sharing

## 📊 Duration Options

### 30 Seconds (`30s`)
- **Use Case**: Quick previews, social media clips
- **Processing Time**: Fastest
- **Moves**: 5 selected moves
- **Output Size**: ~3-5 MB

### 1 Minute (`1min`) - Default
- **Use Case**: Standard choreography demos
- **Processing Time**: Moderate
- **Moves**: 5 selected moves
- **Output Size**: ~5-8 MB

### Full Song (`full`)
- **Use Case**: Complete choreography videos
- **Processing Time**: Longest
- **Moves**: 6 selected moves
- **Output Size**: Varies by song length

## 📁 Output Files

Generated videos are saved to `data/output/` with descriptive filenames:

```
{SongName}_{Duration}_{Quality}_choreography.mp4

Examples:
- Chayanne_Bailando_Bachata_30s_fast_choreography.mp4
- Romeo_Santos_Suegra_1m_fast_choreography.mp4
- Aventura_Obsesion_full_high_choreography.mp4
```

## 🎯 Available Songs

Current collection includes 13 songs:

1. Aventura - Obsesion
2. Bubalu (Bachata Version) - DJ Tronky & J-Style
3. Chayanne - Bailando Bachata
4. Desnúdate (feat. Jose De Rico)
5. Este Secreto
6. Gusttavo Lima - Veneno feat. Prince Royce
7. Jiory - Eso Es Amor - Bachata - Single
8. Lil Nas X ft. Billy Ray Cyrus - Old Town Road (Bachata Remix)
9. Luis Miguel del Amargue - Besito a Besito letra
10. Me Emborracharè (Bachata Version)
11. Ozuna X Doja Cat X Sia - Del Mar (Bachata Remix)
12. Prince Royce - Te Me Vas
13. Romeo Santos - Suegra

## 🔧 Technical Details

### Processing Pipeline
1. **Audio Acquisition**: Load existing song or download from YouTube
2. **Music Analysis**: Extract tempo, beats, sections, and musical features
3. **Move Analysis**: Analyze dance moves using MediaPipe pose detection
4. **Recommendation Engine**: Score and select best moves for the song
5. **Sequence Creation**: Create beat-synchronized choreography sequence
6. **Video Generation**: Generate final video with FFmpeg
7. **Quality Validation**: Verify output file quality

### Performance Metrics
- **Fast Mode**: ~1-2 minutes total processing time
- **High Mode**: ~5-10 minutes total processing time
- **Move Analysis**: ~15-30 seconds per move (fast mode)
- **Video Generation**: ~3-5 seconds for final rendering

### System Requirements
- **Python**: 3.12+
- **Dependencies**: All included in pyproject.toml
- **FFmpeg**: Required for video processing
- **Storage**: ~50-100 MB per generated video
- **RAM**: ~2-4 GB during processing

## 🎨 Choreography Features

### Move Selection
- **Diversity**: Selects moves from different categories
- **Tempo Matching**: Matches moves to song tempo (±15 BPM tolerance)
- **Energy Alignment**: Matches move energy to song energy
- **Difficulty**: Targets intermediate level by default

### Beat Synchronization
- **Beat Detection**: Automatic beat detection from audio
- **Move Timing**: Aligns moves to musical beats
- **Transition Smoothness**: Smooth transitions between moves
- **Musical Structure**: Respects song sections (intro, verse, chorus, etc.)

### Video Quality
- **Resolution**: HD (1280x720) or Full HD (1920x1080)
- **Frame Rate**: 30 FPS
- **Audio Sync**: Perfect audio-video synchronization
- **Compression**: Optimized for web playback

## 🚨 Troubleshooting

### Common Issues

1. **"Song not found"**
   - Use `--list-songs` to see available songs
   - Try partial matches (e.g., "Chayanne" instead of full title)

2. **YouTube download fails**
   - Check internet connection
   - Verify the YouTube URL is valid and accessible
   - Some videos may be region-restricted

3. **Video generation fails**
   - Ensure FFmpeg is installed and in PATH
   - Check available disk space
   - Try with `--quality fast` for faster processing

4. **Long processing times**
   - Use `--quality fast` for quicker results
   - Use `--duration 30s` for shorter videos
   - Close other applications to free up resources

### Performance Tips

1. **For Quick Testing**: Use `--duration 30s --quality fast`
2. **For Final Videos**: Use `--duration 1min --quality high`
3. **For Full Songs**: Use `--duration full --quality fast` first, then high if satisfied

## 🎉 Success Examples

```bash
# Quick 30-second preview of Chayanne
uv run python create_bachata_choreography.py "Chayanne" -d 30s -q fast

# High-quality 1-minute Romeo Santos video
uv run python create_bachata_choreography.py "Romeo Santos" -d 1min -q high

# Full Aventura song with fast processing
uv run python create_bachata_choreography.py "Aventura" -d full -q fast

# Download and process new song from YouTube
uv run python create_bachata_choreography.py "https://youtu.be/VIDEO_ID" -d 1min
```

## 📈 Next Steps

After generating your choreography video:

1. **Review Output**: Check the generated video in `data/output/`
2. **Adjust Settings**: Try different duration/quality combinations
3. **Add New Songs**: Download more songs via YouTube URLs
4. **Share Results**: Videos are optimized for web sharing
5. **Iterate**: Experiment with different songs and settings

---

🎵 **Happy Dancing!** Your AI-generated Bachata choreography awaits! 💃🕺