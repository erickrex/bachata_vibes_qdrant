# Optimized Bachata Choreography Generation Pipeline

## Quick Start

### 1. Setup Environment

**For UV users (recommended):**
```bash
# Navigate to project directory
cd bachata-choreography-generator

# Run UV-specific setup
uv run python setup_uv_environment.py
```

**For pip users:**
```bash
# Navigate to project directory
cd bachata-choreography-generator

# Run standard setup
python setup_optimized_pipeline.py
```

**Both setups will:**
- Check system requirements (Python 3.8+, FFmpeg, Docker)
- Install required Python packages
- Create necessary directories
- Validate installation
- Create test scripts

### 2. Add Test Data

```bash
# Add audio files
cp your_bachata_songs.mp3 data/songs/

# Verify data structure
ls data/songs/     # Should contain .mp3 files
ls data/moves/     # Should contain dance move .mp4 files
ls data/           # Should contain bachata_annotations.json
```

### 3. Run Basic Test

**For UV users:**
```bash
# Test pipeline functionality
uv run python test_pipeline_simple.py

# List available songs
uv run python optimized_choreography_generator.py --list-songs

# Generate your first choreography (fast mode)
uv run python optimized_choreography_generator.py data/songs/your_song.mp3 --quality fast --duration 30s
```

**For pip users:**
```bash
# Test pipeline functionality
python test_pipeline.py

# List available songs
python optimized_choreography_generator.py --list-songs

# Generate your first choreography (fast mode)
python optimized_choreography_generator.py data/songs/your_song.mp3 --quality fast --duration 30s
```

## Complete Testing Instructions

### Phase 1: Basic Functionality

```bash
# Test all quality modes
python optimized_choreography_generator.py data/songs/test_song.mp3 --quality fast --duration 30s
python optimized_choreography_generator.py data/songs/test_song.mp3 --quality balanced --duration 1min
python optimized_choreography_generator.py data/songs/test_song.mp3 --quality high_quality --duration 1min

# Verify outputs
ls -la data/output/                    # Generated videos
ls -la data/choreography_metadata/     # Metadata files
ffprobe data/output/your_video.mp4     # Video properties
```

### Phase 2: Performance Testing

```bash
# Test caching (second run should be faster)
time python optimized_choreography_generator.py data/songs/test_song.mp3 --quality balanced
time python optimized_choreography_generator.py data/songs/test_song.mp3 --quality balanced

# Test parallel processing
python optimized_choreography_generator.py data/songs/song1.mp3 data/songs/song2.mp3 --batch

# Run comprehensive performance test
python optimized_choreography_generator.py --test-all --quality balanced
```

### Phase 3: Vector Database (Optional)

```bash
# Setup Qdrant vector database
python setup_qdrant.py start          # Start Qdrant container
python setup_qdrant.py status         # Verify running
python setup_qdrant.py migrate        # Migrate existing data
python setup_qdrant.py test           # Test performance

# Test with Qdrant enabled
python optimized_choreography_generator.py data/songs/test_song.mp3 --quality balanced
```

### Phase 4: Error Recovery

```bash
# Test error handling
python optimized_choreography_generator.py --test-recovery

# Test with invalid inputs
python optimized_choreography_generator.py nonexistent_file.mp3
python optimized_choreography_generator.py "https://invalid-youtube-url.com"
```

### Phase 5: YouTube Integration

```bash
# Test YouTube download and processing
python optimized_choreography_generator.py "https://www.youtube.com/watch?v=VALID_VIDEO_ID" --quality fast --duration 30s

# Verify downloaded file
ls -la data/temp/    # Should contain downloaded .mp3
```

### Phase 6: Batch Processing

```bash
# Process multiple songs concurrently
python optimized_choreography_generator.py \
    data/songs/song1.mp3 \
    data/songs/song2.mp3 \
    data/songs/song3.mp3 \
    --batch --quality fast --duration 30s

# Monitor system resources during batch processing
htop  # In another terminal
```

### Phase 7: Data Management

```bash
# Test data persistence and search
python3 -c "
from app.services.data_persistence import DataPersistenceManager
manager = DataPersistenceManager()

# Search choreographies
results = manager.search_choreographies(
    tempo_range=(100, 140),
    difficulty='intermediate',
    limit=5
)
print(f'Found {len(results)} choreographies')
for r in results:
    print(f'- {r.choreography_id}: {r.tempo} BPM, {r.difficulty}')
"

# Test export functionality
python3 -c "
from app.services.data_persistence import DataPersistenceManager
manager = DataPersistenceManager()
summary = manager.export_analysis_data('data/export_test.json')
print(f'Export summary: {summary}')
"
```

### Phase 8: Performance Benchmarking

```bash
# Benchmark different quality modes
echo "=== Fast Mode Benchmark ==="
time python optimized_choreography_generator.py data/songs/test_song.mp3 --quality fast --duration 1min

echo "=== Balanced Mode Benchmark ==="
time python optimized_choreography_generator.py data/songs/test_song.mp3 --quality balanced --duration 1min

echo "=== High Quality Mode Benchmark ==="
time python optimized_choreography_generator.py data/songs/test_song.mp3 --quality high_quality --duration 1min
```

### Phase 9: System Validation

```bash
# Comprehensive system test
python optimized_choreography_generator.py --test-all --quality balanced

# Long-running stability test
for i in {1..5}; do
    echo "Running stability test iteration $i"
    python optimized_choreography_generator.py data/songs/test_song.mp3 --quality balanced
    sleep 10
done

# Resource monitoring
watch -n 1 'echo "=== Memory ==="; free -h; echo "=== Disk ==="; df -h; echo "=== Processes ==="; ps aux | grep python | head -5'
```

## Expected Results

### Performance Benchmarks
- **Fast Mode**: 30-60 seconds for 1-minute choreography
- **Balanced Mode**: 60-120 seconds for 1-minute choreography
- **High Quality Mode**: 120-300 seconds for 1-minute choreography
- **Cache Hit**: 50-80% reduction in processing time
- **Qdrant Search**: <10ms similarity search latency

### Success Indicators
- ✅ All quality modes complete without errors
- ✅ Cache hit rates improve on subsequent runs
- ✅ Parallel processing utilizes multiple CPU cores
- ✅ Qdrant integration reduces similarity search time
- ✅ Error recovery handles invalid inputs gracefully
- ✅ Output videos are playable and contain expected content
- ✅ Metadata files contain complete choreography information
- ✅ Performance metrics show expected improvements

### Output Files
```
data/
├── output/                           # Generated choreography videos
│   ├── song_name_1min_balanced_choreography.mp4
│   └── ...
├── choreography_metadata/            # Detailed metadata
│   ├── choreography_id_metadata.json
│   └── ...
├── cache/                           # Analysis cache
│   ├── music_analysis/
│   └── move_embedding/
└── test_results/                    # Test reports
    ├── test_report_timestamp.json
    └── ...
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall packages
   pip install --upgrade librosa numpy opencv-python mediapipe
   ```

2. **FFmpeg Not Found**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Verify installation
   ffmpeg -version
   ```

3. **Memory Issues**
   ```bash
   # Use fast mode with reduced workers
   python optimized_choreography_generator.py song.mp3 --quality fast
   
   # Clear cache
   rm -rf data/cache/*
   ```

4. **Qdrant Connection Issues**
   ```bash
   # Check Qdrant status
   python setup_qdrant.py status
   
   # Restart Qdrant
   python setup_qdrant.py restart
   
   # Check Docker
   docker ps | grep qdrant
   ```

5. **Slow Performance**
   ```bash
   # Enable Qdrant for faster similarity search
   python setup_qdrant.py start
   python setup_qdrant.py migrate
   
   # Use fast mode
   python optimized_choreography_generator.py song.mp3 --quality fast
   ```

### Performance Tuning

- **For Speed**: Use `--quality fast` with Qdrant enabled
- **For Quality**: Use `--quality high_quality` with full caching
- **For Memory**: Reduce workers in config, use fast mode
- **For Accuracy**: Increase confidence thresholds and FPS

### Log Analysis

```bash
# Check logs for errors
tail -f logs/choreography_generator.log

# Monitor system resources
htop
iotop
nvidia-smi  # If using GPU
```

## Advanced Usage

### Custom Configuration

```python
from app.services.choreography_pipeline import PipelineConfig

# Custom pipeline configuration
config = PipelineConfig(
    quality_mode="custom",
    target_fps=15,
    min_detection_confidence=0.5,
    max_workers=6,
    enable_caching=True,
    max_cache_size_mb=1000
)
```

### Programmatic Usage

```python
import asyncio
from app.services.choreography_pipeline import ChoreoGenerationPipeline

async def generate_choreography():
    pipeline = ChoreoGenerationPipeline()
    
    result = await pipeline.generate_choreography(
        audio_input="path/to/song.mp3",
        duration="1min",
        difficulty="intermediate"
    )
    
    if result.success:
        print(f"Generated: {result.output_path}")
        print(f"Processing time: {result.processing_time:.2f}s")
    else:
        print(f"Failed: {result.error_message}")

# Run
asyncio.run(generate_choreography())
```

## Documentation

- **OPTIMIZED_PIPELINE_DOCUMENTATION.md** - Complete feature documentation
- **TESTING_INSTRUCTIONS.md** - Comprehensive testing procedures
- **setup_qdrant.py** - Vector database setup utility
- **setup_optimized_pipeline.py** - Environment setup script

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the comprehensive testing instructions
3. Examine log files for detailed error messages
4. Verify system requirements and dependencies

## Performance Monitoring

The pipeline provides detailed metrics:

```bash
# View performance metrics
python3 -c "
from app.services.optimized_recommendation_engine import OptimizedRecommendationEngine
engine = OptimizedRecommendationEngine()
metrics = engine.get_performance_metrics()
print('Performance Metrics:')
for key, value in metrics.items():
    print(f'  {key}: {value}')
"
```

This optimized pipeline delivers production-ready performance with comprehensive testing, monitoring, and documentation.