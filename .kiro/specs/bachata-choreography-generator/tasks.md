# Implementation Plan

- [ ] 1. Set up UV project and dependencies









  - Initialize UV project with pyproject.toml configuration
  - Add core dependencies (Librosa, MediaPipe, yt-dlp, opencv-python, numpy)
  - Create basic project structure with directories for services, data, and move clips
  - Set up virtual environment and verify all dependencies install correctly
  - _Requirements: 1.1, 1.2_

- [x] 2. Implement YouTube audio download service





  - Create simple YouTubeService class with yt-dlp integration
  - Add basic URL validation and audio extraction to MP3 format
  - Test with sample Bachata YouTube URLs to verify audio quality
  - _Requirements: 1.1, 1.2_

- [x] 3. Build music analysis service with Librosa





  - [x] 3.1 Implement core audio feature extraction


    - Create MusicAnalyzer class with Librosa integration
    - Extract tempo, beat positions, and spectral features (MFCC, chroma, spectral centroid)
    - Test with downloaded Bachata songs to validate tempo detection accuracy
    - _Requirements: 2.1, 2.2, 5.1_



  - [x] 3.2 Generate music embeddings and structure analysis





    - Create embedding generation from extracted audio features
    - Implement musical structure segmentation and energy analysis
    - Create simple feature vector that captures Bachata rhythm characteristics
    - _Requirements: 2.1, 4.3_

- [x] 4. Create move analysis service with MediaPipe


  - [x] 4.1 Set up video annotation framework and data collection






    - ✅ Created comprehensive Pydantic-based annotation schema with all required fields (clip_id, video_path, move_label, energy_level, estimated_tempo, difficulty, lead_follow_roles, notes)
    - ✅ Built annotation validation service with quality checks for video files and data completeness
    - ✅ Implemented CSV import/export interface for bulk annotation editing with templates and detailed instructions
    - ✅ Created directory organization system for categorizing move clips by type
    - ✅ Integrated with existing 38 annotated clips in data/Bachata_steps/ directory structure
    - _Requirements: 4.1, 4.2_

  - [x] 4.2 Implement comprehensive move categorization system


    - ✅ Implemented 12 core move categories based on existing data: basic_step, cross_body_lead, lady_right_turn, lady_left_turn, forward_backward, dip, body_roll, hammerlock, shadow_position, combination, arm_styling, double_cross_body_lead
    - ✅ Created automatic category derivation from move labels with mapping to organized directory structure
    - ✅ Implemented role-based annotations (lead_follow_roles: lead_focus, follow_focus, both) for all 38 clips
    - ✅ Added comprehensive energy level annotations (low/medium/high) and tempo compatibility (102-150 BPM range)
    - ✅ Built quality validation system ensuring annotation consistency and completeness
    - _Requirements: 4.1, 4.2, 7.1_

  - [x] 4.3 Build MediaPipe pose detection and feature extraction system





    - Create MoveAnalyzer class with MediaPipe Pose integration (33 landmarks)
    - Implement hand tracking for styling moves using MediaPipe Hands
    - Set up frame sampling at 30 frames per video for consistent analysis
    - Extract pose landmarks and calculate joint angles for movement dynamics
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 4.4 Implement movement dynamics analysis





    - Calculate movement velocity and acceleration patterns from pose sequences
    - Analyze spatial movement patterns (footwork area coverage, upper body movement range)
    - Extract rhythm compatibility scores by analyzing movement timing patterns
    - Identify transition points and calculate transition compatibility between moves
    - _Requirements: 4.2, 4.4_

  - [x] 4.5 Create feature fusion system for multi-modal embeddings





    - Design 384-dimensional pose feature vector from MediaPipe analysis
    - Implement movement complexity scoring based on joint angle variations and spatial coverage
    - Create tempo compatibility ranges for each move based on movement analysis
    - Generate difficulty scores using movement speed, complexity, and coordination requirements
    - _Requirements: 4.2, 4.3, 7.1, 7.2_

  - [x] 4.6 Build training data validation and quality assurance





    - Create automated quality checks for video annotations (missing fields, invalid values)
    - Implement pose detection confidence scoring to identify low-quality clips
    - Build annotation consistency checker to validate move labels against extracted features
    - Create training data statistics dashboard showing distribution of moves, difficulties, and tempos
    - _Requirements: 4.1, 4.2_

- [x] 5. Build comprehensive recommendation engine and model training





  - [x] 5.1 Create feature fusion system for multi-modal embeddings


    - Implement 512-dimensional combined feature vector (128D audio + 384D pose features)
    - Create audio feature extraction pipeline using MFCC, Chroma, and Tonnetz features from Librosa
    - Build pose feature aggregation from MediaPipe landmarks across video frames
    - Test embedding quality using similarity metrics between known compatible moves and music
    - _Requirements: 2.1, 4.3_

  - [x] 5.2 Implement multi-factor scoring recommendation system


    - Create weighted scoring algorithm: audio similarity (40%), tempo matching (30%), energy alignment (20%), difficulty compatibility (10%)
    - Build cosine similarity matching between music embeddings and move embeddings
    - Implement tempo compatibility scoring with BPM range matching (±10 BPM tolerance)
    - Add energy level alignment scoring (low/medium/high) between music and move characteristics
    - _Requirements: 2.1, 2.2, 7.1, 7.2_

  - [x] 5.3 Build diversity selection and choreography flow optimization


    - Implement diversity selection algorithm to avoid repetitive move sequences
    - Create transition compatibility matrix between all move pairs based on pose analysis
    - Build sequence optimization using dynamic programming for smooth choreography flow
    - Add musical structure awareness to align move complexity with song sections (intro/verse/chorus/bridge)
    - _Requirements: 2.2, 2.3, 5.2, 5.4_

  - [x] 5.4 Create model validation and performance testing framework


    - Build cross-validation system using held-out test songs and expert choreographer ratings
    - Implement A/B testing framework to compare different scoring weight combinations
    - Create evaluation metrics for choreography quality (flow, musicality, difficulty progression)
    - Build performance benchmarking for recommendation speed and accuracy
    - _Requirements: 2.1, 2.2, 5.1-5.4_

- [ ] 6. Create video generation service
  - [x] 6.1 Implement basic video clip stitching





    - Create VideoGenerator class with FFmpeg subprocess integration
    - Stitch selected move clips in chronological sequence
    - Test with simple concatenation first, then add transitions
    - _Requirements: 3.1, 3.3_

  - [x] 6.2 Add audio synchronization and beat alignment





    - Synchronize video clips with beat positions from music analysis
    - Overlay original song audio with generated video sequence
    - Optimize output format for web browser playback and test quality
    - _Requirements: 3.1, 3.2, 3.4_

- [x] 7. Build training data preparation and model optimization pipeline




  - [x] 7.1 Create comprehensive training dataset from annotated move clips



    - Process all 40 annotated move clips through MediaPipe pose detection pipeline
    - Generate pose feature vectors for each clip with movement dynamics analysis
    - Create ground truth similarity matrices between moves based on expert annotations
    - Build training dataset with positive/negative pairs for similarity learning
    - _Requirements: 4.1, 4.2, 4.3_



  - [x] 7.2 Implement hyperparameter optimization for recommendation weights

    - Create grid search framework for optimal scoring weights (audio/tempo/energy/difficulty)
    - Build validation pipeline using expert-rated choreography examples
    - Implement cross-validation with different Bachata song styles and tempos
    - Optimize embedding dimensions and similarity thresholds for best performance

    - _Requirements: 2.1, 2.2, 7.1, 7.2_

  - [x] 7.3 Build model performance monitoring and continuous improvement

    - Create logging system for recommendation decisions and user feedback
    - Implement A/B testing infrastructure for different algorithm versions
    - Build analytics dashboard for tracking choreography quality metrics
    - Create automated retraining pipeline when new move clips are added
    - _Requirements: 2.1, 2.2, 5.1-5.4_

- [x] 8. Create end-to-end validation script





  - Build simple command-line script that runs the complete pipeline
  - Test with sample YouTube URL and validate output video quality
  - Measure processing time and identify performance bottlenecks
  - _Requirements: 1.3, 1.4, 1.5_

- [x] 9. Integrate and optimize choreography generation pipeline





  - [x] 9.1 Create optimized service integration layer


    - Build ChoreoGenerationPipeline class that efficiently coordinates all existing services
    - Implement service caching to avoid redundant analysis (music features, move embeddings)
    - Add parallel processing for move analysis using asyncio/threading where possible
    - Create smart service initialization that only loads required components based on request type
    - Optimize memory usage by implementing lazy loading and cleanup of large objects
    - _Requirements: 1.3, 1.4, 1.5, 2.1, 4.3_

  - [x] 9.2 Build performance-optimized recommendation engine


    - Implement pre-computed similarity matrices for faster move matching
    - Create embedding cache system to store and reuse computed features
    - Add batch processing capabilities for multiple song analysis
    - Optimize feature fusion pipeline to reduce computation time by 50%
    - Implement smart move selection that balances quality and processing speed
    - _Requirements: 2.1, 2.2, 4.3, 5.1-5.4_

  - [x] 9.3 Create comprehensive pipeline test script


    - Build enhanced version of create_bachata_choreography.py with all optimizations
    - Add support for batch processing multiple songs with progress tracking
    - Implement quality vs speed modes (fast/balanced/high-quality)
    - Create comprehensive error handling and recovery mechanisms
    - Add detailed performance metrics and benchmarking capabilities
    - Include validation of all pipeline stages with clear success/failure reporting
    - _Requirements: 1.3, 1.4, 1.5, 8.1_

  - [x] 9.4 Implement data persistence and caching layer


    - Create lightweight JSON-based caching for music analysis results
    - Implement move embedding cache to avoid recomputing MediaPipe features
    - Add metadata persistence for generated choreographies with search capabilities
    - Create cleanup utilities for managing temporary files and cache size
    - Build data export/import functionality for sharing analysis results
    - _Requirements: 4.1, 4.2, 4.3, 6.6_

  - [x] 9.5 Integrate Qdrant vector database for faster similarity search


    - Set up local Qdrant instance using Docker for easy deployment
    - Create single collection for 512-dimensional multimodal embeddings with basic metadata
    - Implement QdrantEmbeddingService to store and search move embeddings efficiently
    - Add simple metadata filtering for tempo range and difficulty level
    - Migrate existing move embeddings to Qdrant with batch upload
    - Replace in-memory similarity search with Qdrant's optimized vector search
    - Test performance improvement over current cosine similarity approach
    - _Requirements: 4.3, 5.1, 5.2_

- [x] 10. Build FastAPI web application





  - [x] 10.1 Create FastAPI app with core endpoints


    - Set up FastAPI application structure
    - Create POST endpoint for choreography generation requests
    - Add video serving endpoint with proper headers for browser playback
    - _Requirements: 1.1, 1.3, 3.4, 6.1_

  - [x] 10.2 Add error handling and validation


    - Implement comprehensive error handling for all services
    - Add request validation and user-friendly error messages
    - Create fallback mechanisms for processing failures
    - _Requirements: 1.1, 1.2, 6.2, 6.3_

- [x] 11. Build core web interface with song selection and progress tracking





  - [x] 11.1 Create centered song selection interface


    - Design clean, centered main page template with dropdown listing available songs from data/songs/ directory
    - Add "New song" option in dropdown that reveals YouTube URL input textbox when selected
    - Create single "Create Choreography" button that uses existing /api/choreography endpoint
    - Add video player interface below form that uses /api/video/{filename} endpoint for playback
    - Remove all extra features (difficulty selection, style controls) - focus on core functionality only
    - _Requirements: 1.1, 3.4_


  - [x] 11.2 Implement real-time progress tracking using existing endpoints


    - Build progress bar that polls /api/task/{task_id} endpoint for status updates
    - Display progress percentage, stage, and status messages from task API response
    - Add loading states that disable form controls during processing
    - Implement dynamic UI that shows/hides YouTube URL input based on dropdown selection
    - Use existing FastAPI endpoints: /api/choreography (POST), /api/task/{task_id} (GET), /api/video/{filename} (GET)
    - _Requirements: 6.1, 6.2, 6.3_

- [ ] 12. Optimize and polish the application
  - [ ] 12.1 Add file cleanup and resource management
    - Implement automatic cleanup of temporary files
    - Add resource monitoring and memory management
    - Create proper shutdown procedures for all services
    - _Requirements: 1.5, 6.6_

  - [ ] 12.2 Create testing and validation framework
    - Write unit tests for core services including recommendation engine
    - Test complete end-to-end workflow with sample YouTube URLs
    - Validate choreography quality with different Bachata styles and tempos
    - Create integration tests for annotation pipeline and model training
    - _Requirements: 2.1, 2.2, 3.1, 3.2, 5.1-5.4_