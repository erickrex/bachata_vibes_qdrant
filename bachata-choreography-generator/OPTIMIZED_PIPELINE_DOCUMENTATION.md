# Optimized Bachata Choreography Generation Pipeline

## Overview

The optimized pipeline provides a production-ready, high-performance choreography generation system with advanced caching, parallel processing, and vector database integration. This documentation covers the complete system architecture, features, and testing procedures.

## Architecture

### Core Components

1. **ChoreoGenerationPipeline** (`app/services/choreography_pipeline.py`)
   - Main orchestration layer with service coordination
   - Thread-safe caching system with TTL management
   - Parallel processing for move analysis
   - Smart service initialization (lazy loading)
   - Memory optimization and automatic cleanup

2. **OptimizedRecommendationEngine** (`app/services/optimized_recommendation_engine.py`)
   - Pre-computed similarity matrices for instant lookups
   - Embedding cache with LRU eviction
   - Batch processing capabilities
   - 50% reduction in feature fusion computation time
   - Performance metrics tracking

3. **DataPersistenceManager** (`app/services/data_persistence.py`)
   - JSON-based music analysis caching
   - Move embedding persistence with pickle
   - Choreography metadata storage with search
   - Automatic cleanup utilities
   - Data export/import functionality

4. **QdrantEmbeddingService** (`app/services/qdrant_service.py`)
   - Vector database integration for similarity search
   - 512-dimensional multimodal embeddings
   - Metadata filtering (tempo, difficulty, energy)
   - Sub-millisecond search performance
   - Docker-based deployment

### Quality Modes

- **Fast Mode**: Optimized for speed with reduced accuracy
  - 10 FPS analysis, 8 max moves, basic confidence thresholds
  - 720p output, 4M video bitrate, 128k audio bitrate
  - Parallel processing enabled, aggressive caching

- **Balanced Mode**: Optimal balance of quality and performance
  - 20 FPS analysis, 12 max moves, moderate confidence thresholds
  - 720p output, 6M video bitrate, 192k audio bitrate
  - Full feature set enabled

- **High Quality Mode**: Maximum quality with longer processing time
  - 30 FPS analysis, 20 max moves, high confidence thresholds
  - 1080p output, 8M video bitrate, 320k audio bitrate
  - Detailed analysis, no cleanup for quality inspection

## Features

### Performance Optimizations

- **Service Caching**: Multi-level caching (memory + disk) with automatic expiration
- **Parallel Processing**: Concurrent move analysis using ThreadPoolExecutor
- **Pre-computed Matrices**: Instant similarity lookups for known move pairs
- **Vector Database**: Qdrant integration for sub-millisecond similarity search
- **Smart Loading**: Lazy initialization of services based on request type
- **Memory Management**: Automatic cleanup and resource optimization

### Data Management

- **Persistent Caching**: Music analysis and move embeddings cached to disk
- **Metadata Storage**: Searchable choreography metadata with tags
- **Export/Import**: Share analysis results between systems
- **Cleanup Utilities**: Automatic management of temporary files and cache size
- **Search Capabilities**: Query by tempo, difficulty, duration, energy level

### Testing & Validation

- **Comprehensive Test Suite**: Multi-quality mode testing framework
- **Error Recovery**: Graceful handling of invalid inputs and failures
- **Performance Benchmarking**: Detailed timing and resource usage metrics
- **Pipeline Validation**: Stage-by-stage success/failure reporting
- **Batch Processing**: Concurrent processing of multiple songs

## Installation & Setup

### Prerequisites

```bash
# Required system dependencies
sudo apt-get update
sudo apt-get install ffmpeg python3-pip docker.io

# Python dependencies
pip install -r requirements.txt

# Optional: Qdrant client for vector database
pip install qdrant-client
```

### Directory Structure

```
bachata-choreography-generator/
├── app/
│   ├── services/
│   │   ├── choreography_pipeline.py      # Main pipeline orchestration
│   │   ├── optimized_recommendation_engine.py  # Performance-optimized recommendations
│   │   ├── data_persistence.py           # Caching and persistence layer
│   │   └── qdrant_service.py             # Vector database integration
├── data/
│   ├── songs/                            # Input audio files (.mp3)
│   ├── moves/                            # Dance move video clips
│   ├── cache/                            # Analysis result cache
│   ├── output/                           # Generated choreography videos
│   └── choreography_metadata/            # Metadata and search index
├── optimized_choreography_generator.py   # Main optimized script
├── setup_qdrant.py                      # Qdrant deployment utility
└── tests/                               # Test suite
```

## Usage

### Basic Usage

```bash
# Generate choreography with balanced quality
python optimized_choreography_generator.py path/to/song.mp3

# Specify quality mode and duration
python optimized_choreography_generator.py song.mp3 --quality high_quality --duration 1min

# Use YouTube URL as input
python optimized_choreography_generator.py "https://youtube.com/watch?v=VIDEO_ID" --quality fast

# Generate with specific difficulty
python optimized_choreography_generator.py song.mp3 --difficulty advanced --duration full
```

### Advanced Features

```bash
# Batch processing multiple songs
python optimized_choreography_generator.py song1.mp3 song2.mp3 song3.mp3 --batch

# Run comprehensive test suite
python optimized_choreography_generator.py --test-all --quality balanced

# Test error recovery mechanisms
python optimized_choreography_generator.py --test-recovery

# List available test songs
python optimized_choreography_generator.py --list-songs
```

### Vector Database Setup

```bash
# Start Qdrant vector database
python setup_qdrant.py start

# Migrate existing data to Qdrant
python setup_qdrant.py migrate

# Test Qdrant performance
python setup_qdrant.py test

# Check Qdrant status
python setup_qdrant.py status

# Stop Qdrant
python setup_qdrant.py stop
```

## Configuration

### Pipeline Configuration

```python
from app.services.choreography_pipeline import PipelineConfig

config = PipelineConfig(
    quality_mode="balanced",           # fast, balanced, high_quality
    target_fps=20,                     # Analysis frame rate
    min_detection_confidence=0.4,      # Pose detection threshold
    max_workers=4,                     # Parallel processing threads
    enable_caching=True,               # Enable result caching
    enable_parallel_move_analysis=True, # Parallel move processing
    max_cache_size_mb=500,             # Maximum cache size
    cleanup_after_generation=True      # Auto cleanup temp files
)
```

### Optimization Configuration

```python
from app.services.optimized_recommendation_engine import OptimizationConfig

config = OptimizationConfig(
    enable_embedding_cache=True,        # Cache move embeddings
    enable_similarity_cache=True,       # Cache similarity computations
    enable_precomputed_matrices=True,   # Use pre-computed similarity matrices
    batch_size=32,                     # Batch processing size
    max_workers=4,                     # Parallel processing threads
    fast_mode=False,                   # Speed vs accuracy trade-off
    similarity_threshold=0.1           # Minimum similarity threshold
)
```

### Qdrant Configuration

```python
from app.services.qdrant_service import QdrantConfig

config = QdrantConfig(
    host="localhost",                  # Qdrant server host
    port=6333,                        # Qdrant server port
    collection_name="bachata_moves",   # Collection name
    vector_size=512,                  # Embedding dimensions
    distance_metric="Cosine"          # Similarity metric
)
```

## Performance Metrics

The optimized pipeline provides comprehensive performance tracking:

### Pipeline Metrics
- Total processing time per stage
- Cache hit/miss ratios
- Memory usage tracking
- Parallel processing efficiency
- Error recovery statistics

### Recommendation Engine Metrics
- Average response time
- Cache performance statistics
- Similarity computation counts
- Pre-computed matrix utilization
- Batch processing throughput

### Vector Database Metrics
- Search latency (sub-millisecond)
- Collection size and growth
- Index performance
- Query throughput
- Storage efficiency

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Reduce cache size and worker count
   python optimized_choreography_generator.py --quality fast
   ```

2. **Slow Performance**
   ```bash
   # Enable Qdrant for faster similarity search
   python setup_qdrant.py start
   python setup_qdrant.py migrate
   ```

3. **Cache Issues**
   ```bash
   # Clear cache manually
   rm -rf data/cache/*
   ```

4. **Qdrant Connection Issues**
   ```bash
   # Check Qdrant status
   python setup_qdrant.py status
   
   # Restart Qdrant
   python setup_qdrant.py restart
   ```

### Performance Tuning

1. **For Speed**: Use fast mode with Qdrant enabled
2. **For Quality**: Use high_quality mode with full caching
3. **For Memory**: Reduce max_workers and cache_size_mb
4. **For Accuracy**: Increase min_detection_confidence and target_fps

## API Reference

### ChoreoGenerationPipeline

```python
pipeline = ChoreoGenerationPipeline(config)

# Generate choreography
result = await pipeline.generate_choreography(
    audio_input="song.mp3",
    duration="1min",
    difficulty="intermediate",
    energy_level="medium"  # Optional: auto-detected if None
)
```

### OptimizedRecommendationEngine

```python
engine = OptimizedRecommendationEngine(config)

# Optimized recommendations
scores = engine.recommend_moves_optimized(
    request=recommendation_request,
    move_candidates=candidates,
    top_k=10
)

# Batch processing
batch_results = engine.batch_recommend_moves(batch_request)
```

### QdrantEmbeddingService

```python
service = QdrantEmbeddingService(config)

# Store embeddings
point_id = service.store_move_embedding(move_candidate)

# Search similar moves
results = service.search_similar_moves(
    query_embedding=embedding,
    limit=10,
    tempo_range=(100, 140),
    difficulty="intermediate"
)
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd bachata-choreography-generator

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 app/
black app/
```

### Adding New Features

1. Follow the modular service architecture
2. Implement comprehensive error handling
3. Add performance metrics tracking
4. Include unit tests and integration tests
5. Update documentation

### Performance Guidelines

1. Use caching for expensive operations
2. Implement parallel processing where beneficial
3. Add performance metrics for monitoring
4. Consider memory usage and cleanup
5. Test with various quality modes

## License

This project is licensed under the MIT License. See LICENSE file for details.