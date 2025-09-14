# Implementation Plan

- [-] 1. Set up UV project and dependencies








  - Initialize UV project with pyproject.toml configuration
  - Add core dependencies (Librosa, MediaPipe, yt-dlp, opencv-python, numpy)
  - Create basic project structure with directories for services, data, and move clips
  - Set up virtual environment and verify all dependencies install correctly
  - _Requirements: 1.1, 1.2_

- [ ] 2. Implement YouTube audio download service
  - Create simple YouTubeService class with yt-dlp integration
  - Add basic URL validation and audio extraction to MP3 format
  - Test with sample Bachata YouTube URLs to verify audio quality
  - _Requirements: 1.1, 1.2_

- [ ] 3. Build music analysis service with Librosa
  - [ ] 3.1 Implement core audio feature extraction
    - Create MusicAnalyzer class with Librosa integration
    - Extract tempo, beat positions, and spectral features (MFCC, chroma, spectral centroid)
    - Test with downloaded Bachata songs to validate tempo detection accuracy
    - _Requirements: 2.1, 2.2, 5.1_

  - [ ] 3.2 Generate music embeddings and structure analysis
    - Create embedding generation from extracted audio features
    - Implement musical structure segmentation and energy analysis
    - Create simple feature vector that captures Bachata rhythm characteristics
    - _Requirements: 2.1, 4.3_

- [x] 4. Create move analysis service with MediaPipe


  - [ ] 4.1 Set up video annotation framework and data collection
    - Create annotation schema with required fields (clip_id, move_label, energy_level, estimated_tempo, difficulty)
    - Set up directory structure for organizing 40 move clips by category (basic_moves/, partner_work/, turns_spins/, styling/, advanced/)
    - Create annotation validation script to ensure video quality standards (full body visible, good lighting, 5-20 second duration)
    - Build annotation interface or CSV template for systematic labeling of all move clips
    - _Requirements: 4.1, 4.2_

  - [ ] 4.2 Implement comprehensive move categorization system
    - Define 12 core move categories: basic_step, forward_backward, side_step, cross_body_lead, hammerlock, close_embrace, lady_left_turn, lady_right_turn, copa, arm_styling, body_roll, hip_roll
    - Create move taxonomy with subcategories for advanced moves (dips, shadow_position, combination)
    - Implement role-based annotations (lead_follow_roles) for partner work identification
    - Add styling annotations for energy levels (low/medium/high) and tempo compatibility (90-140 BPM)
    - _Requirements: 4.1, 4.2, 7.1_

  - [ ] 4.3 Build MediaPipe pose detection and feature extraction system
    - Create MoveAnalyzer class with MediaPipe Pose integration (33 landmarks)
    - Implement hand tracking for styling moves using MediaPipe Hands
    - Set up frame sampling at 30 frames per video for consistent analysis
    - Extract pose landmarks and calculate joint angles for movement dynamics
    - _Requirements: 4.1, 4.2, 4.4_

  - [ ] 4.4 Implement movement dynamics analysis
    - Calculate movement velocity and acceleration patterns from pose sequences
    - Analyze spatial movement patterns (footwork area coverage, upper body movement range)
    - Extract rhythm compatibility scores by analyzing movement timing patterns
    - Identify transition points and calculate transition compatibility between moves
    - _Requirements: 4.2, 4.4_

  - [ ] 4.5 Create feature fusion system for multi-modal embeddings
    - Design 384-dimensional pose feature vector from MediaPipe analysis
    - Implement movement complexity scoring based on joint angle variations and spatial coverage
    - Create tempo compatibility ranges for each move based on movement analysis
    - Generate difficulty scores using movement speed, complexity, and coordination requirements
    - _Requirements: 4.2, 4.3, 7.1, 7.2_

  - [ ] 4.6 Build training data validation and quality assurance
    - Create automated quality checks for video annotations (missing fields, invalid values)
    - Implement pose detection confidence scoring to identify low-quality clips
    - Build annotation consistency checker to validate move labels against extracted features
    - Create training data statistics dashboard showing distribution of moves, difficulties, and tempos
    - _Requirements: 4.1, 4.2_

- [ ] 5. Build comprehensive recommendation engine and model training
  - [ ] 5.1 Create feature fusion system for multi-modal embeddings
    - Implement 512-dimensional combined feature vector (128D audio + 384D pose features)
    - Create audio feature extraction pipeline using MFCC, Chroma, and Tonnetz features from Librosa
    - Build pose feature aggregation from MediaPipe landmarks across video frames
    - Test embedding quality using similarity metrics between known compatible moves and music
    - _Requirements: 2.1, 4.3_

  - [ ] 5.2 Implement multi-factor scoring recommendation system
    - Create weighted scoring algorithm: audio similarity (40%), tempo matching (30%), energy alignment (20%), difficulty compatibility (10%)
    - Build cosine similarity matching between music embeddings and move embeddings
    - Implement tempo compatibility scoring with BPM range matching (±10 BPM tolerance)
    - Add energy level alignment scoring (low/medium/high) between music and move characteristics
    - _Requirements: 2.1, 2.2, 7.1, 7.2_

  - [ ] 5.3 Build diversity selection and choreography flow optimization
    - Implement diversity selection algorithm to avoid repetitive move sequences
    - Create transition compatibility matrix between all move pairs based on pose analysis
    - Build sequence optimization using dynamic programming for smooth choreography flow
    - Add musical structure awareness to align move complexity with song sections (intro/verse/chorus/bridge)
    - _Requirements: 2.2, 2.3, 5.2, 5.4_

  - [ ] 5.4 Create model validation and performance testing framework
    - Build cross-validation system using held-out test songs and expert choreographer ratings
    - Implement A/B testing framework to compare different scoring weight combinations
    - Create evaluation metrics for choreography quality (flow, musicality, difficulty progression)
    - Build performance benchmarking for recommendation speed and accuracy
    - _Requirements: 2.1, 2.2, 5.1-5.4_

- [ ] 6. Create video generation service
  - [ ] 6.1 Implement basic video clip stitching
    - Create VideoGenerator class with FFmpeg subprocess integration
    - Stitch selected move clips in chronological sequence
    - Test with simple concatenation first, then add transitions
    - _Requirements: 3.1, 3.3_

  - [ ] 6.2 Add audio synchronization and beat alignment
    - Synchronize video clips with beat positions from music analysis
    - Overlay original song audio with generated video sequence
    - Optimize output format for web browser playback and test quality
    - _Requirements: 3.1, 3.2, 3.4_

- [ ] 7. Build training data preparation and model optimization pipeline
  - [ ] 7.1 Create comprehensive training dataset from annotated move clips
    - Process all 40 annotated move clips through MediaPipe pose detection pipeline
    - Generate pose feature vectors for each clip with movement dynamics analysis
    - Create ground truth similarity matrices between moves based on expert annotations
    - Build training dataset with positive/negative pairs for similarity learning
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 7.2 Implement hyperparameter optimization for recommendation weights
    - Create grid search framework for optimal scoring weights (audio/tempo/energy/difficulty)
    - Build validation pipeline using expert-rated choreography examples
    - Implement cross-validation with different Bachata song styles and tempos
    - Optimize embedding dimensions and similarity thresholds for best performance
    - _Requirements: 2.1, 2.2, 7.1, 7.2_

  - [ ] 7.3 Build model performance monitoring and continuous improvement
    - Create logging system for recommendation decisions and user feedback
    - Implement A/B testing infrastructure for different algorithm versions
    - Build analytics dashboard for tracking choreography quality metrics
    - Create automated retraining pipeline when new move clips are added
    - _Requirements: 2.1, 2.2, 5.1-5.4_

- [ ] 8. Create end-to-end validation script
  - Build simple command-line script that runs the complete pipeline
  - Test with sample YouTube URL and validate output video quality
  - Measure processing time and identify performance bottlenecks
  - _Requirements: 1.3, 1.4, 1.5_

- [ ] 9. Create data models and database integration
  - [ ] 9.1 Implement Pydantic models for music and move features
    - Create MusicAnalysis, MoveClip, and ChoreographySequence data models
    - Add validation and serialization methods based on proven algorithm results
    - Include annotation schema models for training data management
    - _Requirements: 4.1, 4.2_

  - [ ] 9.2 Set up Qdrant vector database integration
    - Initialize local Qdrant client and collection setup
    - Create database schemas for music and move embeddings with metadata filtering
    - Migrate existing embeddings from algorithm validation to Qdrant
    - Implement efficient similarity search with filtering by difficulty and tempo
    - _Requirements: 4.3, 4.4_

- [ ] 10. Build FastAPI web application
  - [ ] 10.1 Create FastAPI app with core endpoints
    - Set up FastAPI application structure
    - Create POST endpoint for choreography generation requests
    - Add video serving endpoint with proper headers for browser playback
    - _Requirements: 1.1, 1.3, 3.4, 6.1_

  - [ ] 10.2 Add error handling and validation
    - Implement comprehensive error handling for all services
    - Add request validation and user-friendly error messages
    - Create fallback mechanisms for processing failures
    - _Requirements: 1.1, 1.2, 6.2, 6.3_

- [ ] 11. Build web interface with HTMX and Alpine.js
  - [ ] 11.1 Create base HTML templates with Jinja2
    - Design main page template with YouTube URL input form
    - Create video player interface for choreography playback
    - Add user preference controls for difficulty and style selection
    - _Requirements: 1.1, 3.4, 7.1, 7.3_

  - [ ] 11.2 Implement real-time progress tracking
    - Create Server-Sent Events endpoint for progress updates
    - Build Alpine.js components for reactive progress bar
    - Add HTMX integration for seamless form submission and updates
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

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