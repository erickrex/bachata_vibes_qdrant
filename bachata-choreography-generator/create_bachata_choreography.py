#!/usr/bin/env python3
"""
Universal Bachata Choreography Generator.
Creates choreography videos from existing songs or YouTube URLs with customizable duration.
"""

import sys
import time
import asyncio
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add app to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import all required services
from app.services.music_analyzer import MusicAnalyzer
from app.services.move_analyzer import MoveAnalyzer
from app.services.recommendation_engine import (
    RecommendationEngine, RecommendationRequest, MoveCandidate
)
from app.services.video_generator import VideoGenerator, VideoGenerationConfig
from app.services.annotation_interface import AnnotationInterface
from app.services.feature_fusion import FeatureFusion
from app.services.youtube_service import YouTubeService


class BachataChoreoGenerator:
    """Universal Bachata choreography generator with flexible input options."""
    
    def __init__(self, duration: str = "1min", quality: str = "fast"):
        """
        Initialize the choreography generator.
        
        Args:
            duration: Video duration ("30s", "1min", "full")
            quality: Processing quality ("fast", "high")
        """
        self.duration = duration
        self.quality = quality
        
        # Set processing parameters based on quality
        if quality == "fast":
            self.target_fps = 10
            self.max_moves = 8
            self.min_detection_confidence = 0.3
            self.resolution = "1280x720"
            self.video_bitrate = "4M"
            self.audio_bitrate = "128k"
        else:  # high quality
            self.target_fps = 30
            self.max_moves = 15
            self.min_detection_confidence = 0.5
            self.resolution = "1920x1080"
            self.video_bitrate = "8M"
            self.audio_bitrate = "320k"
        
        # Initialize services
        self.music_analyzer = MusicAnalyzer()
        self.move_analyzer = MoveAnalyzer(
            target_fps=self.target_fps,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_detection_confidence
        )
        self.recommendation_engine = RecommendationEngine()
        self.annotation_interface = AnnotationInterface(data_dir="data")
        self.feature_fusion = FeatureFusion()
        self.youtube_service = YouTubeService(output_dir="data/temp")
        
        # Ensure directories exist
        Path("data/output").mkdir(exist_ok=True)
        Path("data/temp").mkdir(exist_ok=True)
        
        # Cache for music analysis
        self._music_features_cache = None
        self._current_song_path = None
        
    def list_available_songs(self) -> List[str]:
        """List all available songs in the data/songs directory."""
        songs_dir = Path("data/songs")
        if not songs_dir.exists():
            return []
        
        songs = []
        for file_path in songs_dir.glob("*.mp3"):
            songs.append(file_path.stem)  # Filename without extension
        
        return sorted(songs)
    
    async def download_from_youtube(self, url: str) -> Optional[str]:
        """Download audio from YouTube URL."""
        print(f"📥 Downloading audio from YouTube...")
        print(f"🔗 URL: {url}")
        
        try:
            result = await self.youtube_service.download_audio(url)
            
            if result.success:
                print(f"✅ Download successful!")
                print(f"   📁 File: {Path(result.file_path).name}")
                print(f"   🎵 Title: {result.title}")
                print(f"   ⏱️  Duration: {result.duration:.1f}s")
                return result.file_path
            else:
                print(f"❌ Download failed: {result.error_message}")
                return None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def find_song_path(self, song_input: str) -> Optional[str]:
        """Find song path from input (name or partial match)."""
        songs_dir = Path("data/songs")
        if not songs_dir.exists():
            return None
        
        # Try exact match first
        exact_path = songs_dir / f"{song_input}.mp3"
        if exact_path.exists():
            return str(exact_path)
        
        # Try partial match (case insensitive)
        song_input_lower = song_input.lower()
        for file_path in songs_dir.glob("*.mp3"):
            if song_input_lower in file_path.stem.lower():
                return str(file_path)
        
        return None
    
    async def generate_choreography(self, song_input: str) -> Dict[str, Any]:
        """
        Generate choreography video from song input.
        
        Args:
            song_input: Either a song name or YouTube URL
        """
        print("🎵 BACHATA CHOREOGRAPHY GENERATOR")
        print("=" * 50)
        print(f"⚙️  Quality: {self.quality.upper()}")
        print(f"⏱️  Duration: {self.duration}")
        
        start_time = time.time()
        results = {}
        
        try:
            # Step 1: Get audio file
            audio_path = await self._get_audio_file(song_input)
            if not audio_path:
                return {"success": False, "error": "Could not get audio file"}
            
            results["audio_path"] = audio_path
            
            # Step 2: Analyze music
            music_features = await self._analyze_music(audio_path)
            if not music_features:
                return {"success": False, "error": "Music analysis failed"}
            
            results["music_analysis"] = music_features
            
            # Step 3: Analyze moves
            move_candidates = await self._analyze_moves()
            if not move_candidates:
                return {"success": False, "error": "Move analysis failed"}
            
            results["move_candidates"] = len(move_candidates)
            
            # Step 4: Generate recommendations
            recommendations = await self._generate_recommendations(
                music_features, move_candidates
            )
            if not recommendations:
                return {"success": False, "error": "Recommendation generation failed"}
            
            results["recommendations"] = len(recommendations)
            
            # Step 5: Create sequence
            sequence = await self._create_sequence(recommendations, music_features)
            if not sequence:
                return {"success": False, "error": "Sequence creation failed"}
            
            results["sequence"] = {
                "moves": len(sequence.moves),
                "duration": sequence.total_duration
            }
            
            # Step 6: Generate video
            video_result = await self._generate_video(sequence, music_features, audio_path)
            if not video_result:
                return {"success": False, "error": "Video generation failed"}
            
            results["video_output"] = video_result
            
            # Step 7: Validate output
            validation = await self._validate_output(video_result)
            results["validation"] = validation
            
            total_time = time.time() - start_time
            results["total_time"] = total_time
            results["success"] = True
            
            self._print_summary(results, audio_path)
            return results
            
        except Exception as e:
            logger.error(f"Choreography generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _get_audio_file(self, song_input: str) -> Optional[str]:
        """Get audio file from input (existing song or YouTube URL)."""
        print("\n🎧 Step 1: Getting Audio File")
        print("-" * 30)
        
        # Check if it's a YouTube URL
        if song_input.startswith(('http://', 'https://', 'www.', 'youtube.com', 'youtu.be')):
            if not song_input.startswith('http'):
                song_input = 'https://' + song_input
            return await self.download_from_youtube(song_input)
        
        # Try to find existing song
        song_path = self.find_song_path(song_input)
        if song_path:
            print(f"✅ Found existing song: {Path(song_path).name}")
            return song_path
        
        # List available songs if not found
        available_songs = self.list_available_songs()
        print(f"❌ Song '{song_input}' not found")
        print(f"📋 Available songs ({len(available_songs)}):")
        for i, song in enumerate(available_songs[:10], 1):
            print(f"   {i:2d}. {song}")
        if len(available_songs) > 10:
            print(f"   ... and {len(available_songs) - 10} more")
        
        return None
    
    async def _analyze_music(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Analyze music and cache results."""
        print("\n🎼 Step 2: Music Analysis")
        print("-" * 30)
        
        if self._music_features_cache and self._current_song_path == audio_path:
            print("✅ Using cached music analysis")
            return self._music_features_cache
        
        print(f"🎵 Analyzing: {Path(audio_path).name}")
        
        try:
            music_features = self.music_analyzer.analyze_audio(audio_path)
            
            print(f"✅ Analysis complete!")
            print(f"   🥁 Tempo: {music_features.tempo:.1f} BPM")
            print(f"   ⏱️  Duration: {music_features.duration:.1f}s")
            print(f"   🎵 Beats: {len(music_features.beat_positions)}")
            print(f"   📊 Sections: {len(music_features.sections)}")
            
            # Convert to dictionary and cache
            music_dict = {
                "tempo": music_features.tempo,
                "duration": music_features.duration,
                "beat_positions": music_features.beat_positions,
                "sections": [
                    {
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "section_type": s.section_type,
                        "energy_level": s.energy_level,
                        "recommended_move_types": s.recommended_move_types
                    }
                    for s in music_features.sections
                ],
                "rhythm_pattern_strength": music_features.rhythm_pattern_strength,
                "syncopation_level": music_features.syncopation_level,
                "audio_embedding": music_features.audio_embedding,
                "features_object": music_features
            }
            
            self._music_features_cache = music_dict
            self._current_song_path = audio_path
            return music_dict
            
        except Exception as e:
            print(f"❌ Music analysis failed: {e}")
            return None
    
    async def _analyze_moves(self) -> Optional[List[MoveCandidate]]:
        """Analyze dance moves."""
        print(f"\n💃 Step 3: Move Analysis ({self.quality.upper()} mode)")
        print("-" * 30)
        
        try:
            collection = self.annotation_interface.load_annotations("bachata_annotations.json")
            print(f"📋 Loaded {collection.total_clips} move clips")
            
            # Select diverse moves
            selected_clips = self._select_diverse_moves(collection.clips, self.max_moves)
            print(f"🔍 Analyzing {len(selected_clips)} moves...")
            
            move_candidates = []
            for i, clip in enumerate(selected_clips):
                try:
                    video_path = Path("data") / clip.video_path
                    if not video_path.exists():
                        continue
                    
                    print(f"   📊 {i+1}/{len(selected_clips)}: {clip.move_label}")
                    
                    analysis_result = self.move_analyzer.analyze_move_clip(str(video_path))
                    music_features_obj = self._music_features_cache["features_object"]
                    multimodal_embedding = self.feature_fusion.create_multimodal_embedding(
                        music_features_obj, analysis_result
                    )
                    
                    candidate = MoveCandidate(
                        move_id=clip.clip_id,
                        video_path=str(video_path),
                        move_label=clip.move_label,
                        analysis_result=analysis_result,
                        multimodal_embedding=multimodal_embedding,
                        energy_level=clip.energy_level,
                        difficulty=clip.difficulty,
                        estimated_tempo=clip.estimated_tempo,
                        lead_follow_roles=clip.lead_follow_roles
                    )
                    
                    move_candidates.append(candidate)
                    
                except Exception as e:
                    print(f"   ⚠️  Failed: {clip.clip_id}: {e}")
                    continue
            
            print(f"✅ Analyzed {len(move_candidates)} moves successfully")
            return move_candidates if move_candidates else None
            
        except Exception as e:
            print(f"❌ Move analysis failed: {e}")
            return None
    
    def _select_diverse_moves(self, clips: List, max_count: int) -> List:
        """Select diverse moves for analysis."""
        categories = {}
        for clip in clips:
            category = clip.move_label
            if category not in categories:
                categories[category] = []
            categories[category].append(clip)
        
        selected = []
        categories_list = list(categories.keys())
        
        # Take one from each category first
        for category in categories_list:
            if len(selected) < max_count and categories[category]:
                selected.append(categories[category][0])
        
        # Fill remaining slots
        category_idx = 0
        while len(selected) < max_count:
            category = categories_list[category_idx % len(categories_list)]
            remaining = [c for c in categories[category] if c not in selected]
            
            if remaining:
                selected.append(remaining[0])
            
            category_idx += 1
            if category_idx > len(categories_list) * 3:
                break
        
        return selected[:max_count]
    
    def _select_diverse_sequence_moves(self, recommendations: List, target_duration: float) -> List:
        """Select diverse moves for the sequence, avoiding repetition."""
        import random
        import hashlib
        
        # Create a seed based on the song name for consistent but different randomization per song
        song_seed = hashlib.md5(str(self._current_song_path).encode()).hexdigest()[:8]
        random.seed(int(song_seed, 16))
        
        # Group recommendations by move type
        move_groups = {}
        for rec in recommendations:
            move_type = rec.move_candidate.move_label
            if move_type not in move_groups:
                move_groups[move_type] = []
            move_groups[move_type].append(rec)
        
        # Calculate how many moves we need based on target duration
        # Assume each move is about 6-8 seconds on average
        avg_move_duration = 7.0
        num_moves_needed = max(4, int(target_duration / avg_move_duration))
        
        selected = []
        move_types_used = set()
        
        # Shuffle the move groups for different ordering per song
        move_group_items = list(move_groups.items())
        random.shuffle(move_group_items)
        
        # First pass: select one from each different move type with randomization
        for move_type, group in move_group_items:
            if len(selected) < num_moves_needed:
                # Add randomness to avoid always picking the same moves
                if len(group) > 1:
                    # Pick randomly from top candidates in each category
                    candidate_pool = group[:min(len(group), 3)]
                    selected.append(random.choice(candidate_pool))
                else:
                    selected.append(group[0])
                move_types_used.add(move_type)
        
        # Second pass: fill remaining slots with randomized selection
        remaining_recs = [rec for rec in recommendations if rec not in selected]
        random.shuffle(remaining_recs)  # Randomize order
        
        while len(selected) < num_moves_needed and remaining_recs:
            # Add some variety by occasionally picking less optimal moves
            if random.random() < 0.3:  # 30% chance to pick a random move for variety
                next_rec = random.choice(remaining_recs)
            else:
                # Pick the best remaining move
                next_rec = remaining_recs[0]
            
            selected.append(next_rec)
            remaining_recs.remove(next_rec)
        
        # Reset random seed to avoid affecting other operations
        random.seed()
        
        print(f"🎯 Selected {len(selected)} diverse moves for {target_duration:.0f}s target (seed: {song_seed})")
        return selected
    
    def _create_full_duration_sequence(
        self, 
        video_paths: List[str], 
        music_features: Dict[str, Any], 
        target_duration: float,
        selected_moves: List
    ) -> Any:
        """Create a sequence that respects ACTUAL song duration and natural clip timing."""
        from app.models.video_models import ChoreographySequence, SelectedMove, TransitionType
        
        # CRITICAL FIX: Use actual song duration, not target duration
        song_duration = music_features.get('duration', target_duration)
        effective_duration = min(target_duration, song_duration)
        
        beat_positions = music_features.get('beat_positions', [])
        tempo = music_features.get('tempo', 120)
        
        moves = []
        current_time = 0.0
        move_index = 0
        
        print(f"🔄 Creating sequence for {effective_duration:.1f}s (song: {song_duration:.1f}s, target: {target_duration:.1f}s)")
        
        # Keep adding moves until we reach the EFFECTIVE duration (not exceeding song length)
        while current_time < effective_duration:
            # Cycle through available moves with some randomization
            base_index = move_index % len(video_paths)
            video_path = video_paths[base_index]
            move_label = selected_moves[base_index].move_candidate.move_label
            
            # Get NATURAL clip duration instead of calculating artificial duration
            natural_duration = self._get_natural_clip_duration(video_path)
            
            # CRITICAL FIX: Ensure we don't exceed song duration
            remaining_time = effective_duration - current_time
            if natural_duration > remaining_time:
                # If natural clip is too long, we need to stop here
                print(f"   ⚠️  Stopping at {current_time:.1f}s - next clip ({natural_duration:.1f}s) would exceed song duration")
                break
            
            move = SelectedMove(
                clip_id=f"move_{move_index + 1}_{move_label}",
                video_path=video_path,
                start_time=current_time,
                duration=natural_duration,  # Use natural duration
                transition_type=TransitionType.CUT
            )
            
            moves.append(move)
            current_time += natural_duration
            move_index += 1
            
            print(f"   Move {move_index}: {move_label} ({natural_duration:.1f}s) -> {current_time:.1f}s")
            
            # Safety check to prevent infinite loops
            max_moves = 50  # Reasonable limit for any duration
            if move_index > max_moves:
                print(f"⚠️  Safety limit reached ({max_moves} moves), stopping sequence creation")
                break
        
        sequence = ChoreographySequence(
            moves=moves,
            total_duration=current_time,
            difficulty_level="mixed",
            audio_tempo=tempo,
            generation_parameters={
                "sync_type": "natural_duration_sequence",
                "tempo": tempo,
                "target_duration": target_duration,
                "song_duration": song_duration,
                "actual_duration": current_time,
                "moves_repeated": move_index > len(video_paths)
            }
        )
        
        return sequence
    
    def _get_natural_clip_duration(self, video_path: str) -> float:
        """Get the natural duration of a video clip using ffprobe."""
        import subprocess
        import json
        
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info.get("format", {}).get("duration", 8.0))
                return duration
            else:
                print(f"   ⚠️  Could not get duration for {video_path}, using default 8.0s")
                return 8.0
                
        except Exception as e:
            print(f"   ⚠️  Error getting duration for {video_path}: {e}, using default 8.0s")
            return 8.0
    
    async def _generate_recommendations(
        self, 
        music_features: Dict[str, Any], 
        move_candidates: List[MoveCandidate]
    ) -> Optional[List]:
        """Generate move recommendations."""
        print(f"\n🎯 Step 4: Generating Recommendations")
        print("-" * 30)
        
        try:
            music_features_obj = music_features["features_object"]
            reference_embedding = move_candidates[0].multimodal_embedding
            
            request = RecommendationRequest(
                music_features=music_features_obj,
                music_embedding=reference_embedding,
                target_difficulty="intermediate",
                target_energy=None,
                tempo_tolerance=15.0
            )
            
            recommendations = self.recommendation_engine.recommend_moves(
                request, move_candidates, top_k=len(move_candidates)
            )
            
            print(f"✅ Generated {len(recommendations)} recommendations")
            print("🏆 Top moves:")
            for i, rec in enumerate(recommendations[:5]):
                print(f"   {i+1}. {rec.move_candidate.move_label:<20} ({rec.overall_score:.3f})")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Recommendation generation failed: {e}")
            return None
    
    async def _create_sequence(
        self, 
        recommendations: List, 
        music_features: Dict[str, Any]
    ) -> Optional[Any]:
        """Create choreography sequence based on duration setting."""
        print(f"\n🎬 Step 5: Creating Sequence ({self.duration})")
        print("-" * 30)
        
        try:
            # Determine target duration
            full_duration = music_features["duration"]
            if self.duration == "30s":
                target_duration = 30.0
            elif self.duration == "1min":
                target_duration = 60.0
            else:  # "full"
                target_duration = full_duration
            
            target_duration = min(target_duration, full_duration)
            
            print(f"🎵 Target duration: {target_duration:.0f}s (of {full_duration:.0f}s total)")
            
            # Select diverse moves for sequence with better distribution
            selected_moves = self._select_diverse_sequence_moves(recommendations, target_duration)
            video_paths = [rec.move_candidate.video_path for rec in selected_moves]
            
            print(f"📝 Selected {len(selected_moves)} moves:")
            for i, rec in enumerate(selected_moves):
                print(f"   {i+1}. {rec.move_candidate.move_label}")
            
            # Configure video generator
            output_filename = self._generate_output_filename(music_features)
            self.video_generator = VideoGenerator(
                VideoGenerationConfig(
                    output_path=f"data/output/{output_filename}",
                    cleanup_temp_files=False,
                    add_audio_overlay=True,
                    normalize_audio=self.quality == "high",
                    resolution=self.resolution,
                    video_bitrate=self.video_bitrate,
                    audio_bitrate=self.audio_bitrate
                )
            )
            
            # Create sequence with improved duration handling
            sequence = self._create_full_duration_sequence(
                video_paths=video_paths,
                music_features=music_features,
                target_duration=target_duration,
                selected_moves=selected_moves
            )
            
            print(f"✅ Sequence created: {len(sequence.moves)} moves, {sequence.total_duration:.1f}s")
            return sequence
            
        except Exception as e:
            print(f"❌ Sequence creation failed: {e}")
            return None
    
    def _generate_output_filename(self, music_features: Dict[str, Any]) -> str:
        """Generate output filename based on song and settings."""
        # Try to extract song name from current path
        if self._current_song_path:
            song_name = Path(self._current_song_path).stem
            # Clean up the name
            song_name = song_name.replace(" - ", "_").replace(" ", "_").replace("(", "").replace(")", "")
            song_name = "".join(c for c in song_name if c.isalnum() or c in "_-")[:30]
        else:
            song_name = "bachata_song"
        
        # Add duration and quality info
        duration_suffix = self.duration.replace("min", "m")
        quality_suffix = "fast" if self.quality == "fast" else "hq"
        
        return f"{song_name}_{duration_suffix}_{quality_suffix}_choreography.mp4"
    
    async def _generate_video(
        self, 
        sequence: Any, 
        music_features: Dict[str, Any],
        audio_path: str
    ) -> Optional[Dict[str, Any]]:
        """Generate final video."""
        print(f"\n🎥 Step 6: Video Generation ({self.quality.upper()})")
        print("-" * 30)
        
        try:
            result = self.video_generator.generate_choreography_video(
                sequence=sequence,
                audio_path=audio_path,
                music_features=music_features
            )
            
            if result.success:
                print(f"✅ Video generated successfully!")
                print(f"   📁 Output: {Path(result.output_path).name}")
                print(f"   ⏱️  Duration: {result.duration:.1f}s")
                print(f"   💾 Size: {result.file_size/1024/1024:.1f} MB")
                print(f"   ⚡ Processing: {result.processing_time:.1f}s")
                
                # Export sequence metadata to JSON
                metadata_path = self._export_sequence_metadata(
                    sequence, result, music_features, audio_path
                )
                
                return {
                    "output_path": result.output_path,
                    "duration": result.duration,
                    "file_size": result.file_size,
                    "processing_time": result.processing_time,
                    "metadata_path": metadata_path,
                    "sequence": sequence
                }
            else:
                print(f"❌ Video generation failed: {result.error_message}")
                return None
                
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            return None
    
    async def _validate_output(self, video_result: Dict[str, Any]) -> Dict[str, Any]:
        """Quick output validation."""
        output_path = video_result["output_path"]
        
        validation = {"file_exists": False, "file_size_mb": 0, "quality_score": 0.0}
        
        if Path(output_path).exists():
            validation["file_exists"] = True
            file_size = Path(output_path).stat().st_size
            validation["file_size_mb"] = file_size / (1024 * 1024)
            
            quality_checks = [
                validation["file_exists"],
                validation["file_size_mb"] > 1,
            ]
            validation["quality_score"] = sum(quality_checks) / len(quality_checks)
        
        return validation
    
    def _export_sequence_metadata(
        self,
        sequence: Any,
        video_result: Any,
        music_features: Dict[str, Any],
        audio_path: str
    ) -> str:
        """Export detailed choreography sequence metadata to JSON with ACCURATE video composition."""
        try:
            # Create metadata directory
            metadata_dir = Path("data/choreography_metadata")
            metadata_dir.mkdir(exist_ok=True)
            
            # Generate metadata filename
            song_name = Path(audio_path).stem
            safe_song_name = "".join(c for c in song_name if c.isalnum() or c in "_-")[:40]
            metadata_filename = f"{safe_song_name}_{self.duration}_{self.quality}_sequence.json"
            metadata_path = metadata_dir / metadata_filename
            
            # Calculate ACTUAL video composition (not planned times)
            actual_composition = []
            current_time = 0.0
            
            for i, move in enumerate(sequence.moves):
                actual_duration = self._get_actual_clip_duration(move.video_path)
                
                composition_entry = {
                    "position": i + 1,
                    "clip_id": move.clip_id,
                    "video_path": move.video_path,
                    "move_name": Path(move.video_path).stem,
                    "move_category": Path(move.video_path).parent.name,
                    "actual_start_time": current_time,
                    "actual_duration": actual_duration,
                    "actual_end_time": current_time + actual_duration,
                    "transition_type": move.transition_type.value if hasattr(move.transition_type, 'value') else str(move.transition_type),
                    "note": "These are the ACTUAL times this clip appears in the final video"
                }
                actual_composition.append(composition_entry)
                current_time += actual_duration
            
            # Create detailed metadata with ACCURATE information
            metadata = {
                "song_info": {
                    "name": song_name,
                    "audio_path": audio_path,
                    "tempo": music_features.get("tempo", 0),
                    "duration": music_features.get("duration", 0),
                    "beats_detected": len(music_features.get("beat_positions", [])),
                    "rhythm_strength": music_features.get("rhythm_pattern_strength", 0),
                    "syncopation_level": music_features.get("syncopation_level", 0)
                },
                "choreography_sequence": {
                    "total_moves": len(sequence.moves),
                    "actual_sequence_duration": current_time,
                    "video_output_duration": video_result.duration,
                    "target_duration": self.duration,
                    "audio_tempo": sequence.audio_tempo,
                    "difficulty_level": sequence.difficulty_level,
                    "composition_method": "natural_clip_durations",
                    "generation_parameters": sequence.generation_parameters or {},
                    "note": "This metadata reflects the ACTUAL video composition, not planned durations"
                },
                "actual_video_composition": actual_composition,
                "musical_sections": music_features.get("sections", []),
                "video_output": {
                    "file_path": video_result.output_path,
                    "actual_duration": video_result.duration,
                    "file_size": video_result.file_size,
                    "file_size_mb": video_result.file_size / (1024 * 1024) if video_result.file_size else 0,
                    "resolution": self.resolution,
                    "quality_setting": self.quality,
                    "processing_time": video_result.processing_time,
                    "encoding_method": "fixed_natural_durations",
                    "note": "Generated with fixed method - no clip freezing or stretching"
                },
                "generation_info": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "generator_version": "1.1-fixed",
                    "fixes_applied": [
                        "Removed clip duration restrictions",
                        "Fixed FFmpeg encoding to prevent freezing/stretching", 
                        "Accurate metadata reflecting actual video composition",
                        "Natural clip durations preserved"
                    ],
                    "settings": {
                        "duration": self.duration,
                        "quality": self.quality,
                        "max_moves_analyzed": self.max_moves,
                        "target_fps": self.target_fps
                    }
                }
            }
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"💾 Sequence metadata saved: {metadata_filename}")
            return str(metadata_path)
            
        except Exception as e:
            print(f"⚠️  Failed to export metadata: {e}")
            return ""

    def _get_actual_clip_duration(self, video_path: str) -> float:
        """Get the actual duration of a video clip using ffprobe."""
        try:
            import subprocess
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                return float(info.get("format", {}).get("duration", 0))
        except Exception as e:
            print(f"⚠️  Could not get duration for {video_path}: {e}")
        
        return 0.0
    
    def _print_summary(self, results: Dict[str, Any], audio_path: str) -> None:
        """Print final summary."""
        print("\n" + "=" * 50)
        print("🎉 CHOREOGRAPHY GENERATION COMPLETE!")
        print("=" * 50)
        
        if results["success"]:
            print("✅ SUCCESS: Choreography video generated!")
            
            print(f"\n🎵 SONG:")
            print(f"   File: {Path(audio_path).name}")
            print(f"   Tempo: {results['music_analysis']['tempo']:.1f} BPM")
            print(f"   Duration: {results['music_analysis']['duration']:.1f}s")
            
            print(f"\n💃 CHOREOGRAPHY:")
            print(f"   Moves analyzed: {results['move_candidates']}")
            print(f"   Sequence moves: {results['sequence']['moves']}")
            print(f"   Video duration: {results['sequence']['duration']:.1f}s")
            print(f"   Quality: {self.quality.upper()}")
            
            video = results["video_output"]
            print(f"\n🎬 OUTPUT:")
            print(f"   File: {Path(video['output_path']).name}")
            print(f"   Size: {video['file_size']/1024/1024:.1f} MB")
            print(f"   Processing time: {results['total_time']:.1f}s")
            
            print(f"\n📁 LOCATIONS:")
            print(f"   🎥 Video: {video['output_path']}")
            if video.get('metadata_path'):
                print(f"   📋 Metadata: {video['metadata_path']}")
            
        else:
            print(f"❌ FAILED: {results.get('error', 'Unknown error')}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate Bachata choreography videos from songs or YouTube URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing song
  python create_bachata_choreography.py "Chayanne"
  python create_bachata_choreography.py "Aventura - Obsesion"
  
  # Download from YouTube
  python create_bachata_choreography.py "https://www.youtube.com/watch?v=abc123"
  
  # Customize duration and quality
  python create_bachata_choreography.py "Chayanne" --duration 30s --quality high
  python create_bachata_choreography.py "Romeo Santos" --duration full --quality fast
        """
    )
    
    parser.add_argument(
        "song",
        nargs="?",
        help="Song name (partial match) or YouTube URL"
    )
    
    parser.add_argument(
        "--duration", "-d",
        choices=["30s", "1min", "full"],
        default="1min",
        help="Video duration (default: 1min)"
    )
    
    parser.add_argument(
        "--quality", "-q",
        choices=["fast", "high"],
        default="fast",
        help="Processing quality (default: fast)"
    )
    
    parser.add_argument(
        "--list-songs", "-l",
        action="store_true",
        help="List available songs and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = BachataChoreoGenerator(
        duration=args.duration,
        quality=args.quality
    )
    
    # List songs if requested
    if args.list_songs:
        songs = generator.list_available_songs()
        print("📋 Available Songs:")
        print("=" * 30)
        for i, song in enumerate(songs, 1):
            print(f"{i:2d}. {song}")
        print(f"\nTotal: {len(songs)} songs")
        return 0
    
    # Check if song argument is provided when not listing
    if not args.song:
        parser.error("Song argument is required unless using --list-songs")
    
    # Generate choreography
    async def run_generation():
        results = await generator.generate_choreography(args.song)
        return 0 if results["success"] else 1
    
    return asyncio.run(run_generation())


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)