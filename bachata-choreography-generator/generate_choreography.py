#!/usr/bin/env python3
"""
Simple choreography generation script with configurable parameters.
Easy-to-use script for generating choreographies with different settings.

Usage Examples:
    python generate_choreography.py
    python generate_choreography.py --song Aventura --difficulty advanced --energy high
    python generate_choreography.py --song Amor --role-focus follow_focus --quality high_quality
"""

import asyncio
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional
import sys

# Add the app directory to the path
sys.path.append('.')
sys.path.append('app')

from app.services.choreography_pipeline import ChoreoGenerationPipeline, PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default settings - change these to your preferred defaults
DEFAULT_SETTINGS = {
    "song": "Aventura",
    "difficulty": "intermediate",
    "energy_level": None,
    "role_focus": None,
    "quality_mode": "balanced",
    "move_types": None,
    "tempo_range": None,
}

# Available move types
AVAILABLE_MOVE_TYPES = [
    "basic_step", "cross_body_lead", "lady_right_turn", "lady_left_turn",
    "forward_backward", "dips", "body_roll", "hammerlock", "shadow_position",
    "combination", "arm_styling", "double_cross_body_lead"
]

# Predefined configurations for quick access
PRESET_CONFIGURATIONS = {
    "beginner_romantic": {
        "difficulty": "beginner",
        "energy_level": "low",
        "role_focus": "both",
        "move_types": ["basic_step", "cross_body_lead", "forward_backward"],
        "tempo_range": [102, 120]
    },
    "intermediate_energetic": {
        "difficulty": "intermediate", 
        "energy_level": "high",
        "role_focus": "both",
        "move_types": ["cross_body_lead", "lady_right_turn", "body_roll"],
        "tempo_range": [120, 140]
    },
    "advanced_showcase": {
        "difficulty": "advanced",
        "energy_level": "high", 
        "role_focus": "both",
        "move_types": ["dips", "hammerlock", "shadow_position", "combination"],
        "tempo_range": [130, 150]
    },
    "lead_focused": {
        "difficulty": "intermediate",
        "energy_level": "medium",
        "role_focus": "lead_focus",
        "move_types": ["cross_body_lead", "double_cross_body_lead", "hammerlock"]
    },
    "follow_styling": {
        "difficulty": "intermediate",
        "energy_level": "medium", 
        "role_focus": "follow_focus",
        "move_types": ["lady_right_turn", "lady_left_turn", "arm_styling", "body_roll"]
    }
}

def get_available_songs() -> List[str]:
    """Get list of available songs."""
    songs_dir = Path("data/songs")
    if not songs_dir.exists():
        logger.error("Songs directory not found!")
        return []
    
    songs = []
    for song_file in songs_dir.glob("*.mp3"):
        songs.append(song_file.stem)
    
    return sorted(songs)

def display_available_options():
    """Display all available options for user reference."""
    songs = get_available_songs()
    
    print("🎵 BACHATA CHOREOGRAPHY GENERATOR")
    print("=" * 50)
    print(f"Available Songs ({len(songs)}):")
    for i, song in enumerate(songs, 1):
        print(f"  {i:2d}. {song}")
    
    print(f"\nAvailable Move Types ({len(AVAILABLE_MOVE_TYPES)}):")
    for i, move_type in enumerate(AVAILABLE_MOVE_TYPES, 1):
        print(f"  {i:2d}. {move_type}")
    
    print(f"\nPreset Configurations ({len(PRESET_CONFIGURATIONS)}):")
    for name, config in PRESET_CONFIGURATIONS.items():
        print(f"  • {name}: {config}")
    
    print("\nDifficulty Levels: beginner, intermediate, advanced")
    print("Energy Levels: low, medium, high (or auto-detect)")
    print("Role Focus: lead_focus, follow_focus, both")
    print("Quality Modes: fast, balanced, high_quality")
    print("Tempo Range: [min, max] BPM (e.g., [120, 140])")
    print("=" * 50)

async def generate_choreography(
    song: str,
    difficulty: str = "intermediate",
    energy_level: Optional[str] = None,
    role_focus: Optional[str] = None,
    move_types: Optional[List[str]] = None,
    tempo_range: Optional[List[int]] = None,
    quality_mode: str = "balanced",
    preset: Optional[str] = None
) -> bool:
    """Generate choreography with specified parameters."""
    
    # Apply preset configuration if specified
    if preset and preset in PRESET_CONFIGURATIONS:
        preset_config = PRESET_CONFIGURATIONS[preset]
        logger.info(f"Applying preset configuration: {preset}")
        
        # Override with preset values (command line args take precedence)
        difficulty = difficulty if difficulty != DEFAULT_SETTINGS["difficulty"] else preset_config.get("difficulty", difficulty)
        energy_level = energy_level if energy_level is not None else preset_config.get("energy_level")
        role_focus = role_focus if role_focus is not None else preset_config.get("role_focus")
        move_types = move_types if move_types is not None else preset_config.get("move_types")
        tempo_range = tempo_range if tempo_range is not None else preset_config.get("tempo_range")
    
    # Validate song exists
    available_songs = get_available_songs()
    if song not in available_songs:
        logger.error(f"Song '{song}' not found. Available songs: {', '.join(available_songs)}")
        return False
    
    song_path = Path(f"data/songs/{song}.mp3")
    
    # Display configuration
    print("🎵 CHOREOGRAPHY GENERATION CONFIGURATION")
    print("=" * 50)
    print(f"Song: {song}")
    print(f"Difficulty: {difficulty}")
    print(f"Energy Level: {energy_level or 'Auto-detect'}")
    print(f"Role Focus: {role_focus or 'Both partners'}")
    print(f"Quality Mode: {quality_mode}")
    if move_types:
        print(f"Move Types: {', '.join(move_types)}")
    if tempo_range:
        print(f"Tempo Range: {tempo_range[0]}-{tempo_range[1]} BPM")
    if preset:
        print(f"Preset Used: {preset}")
    print("=" * 50)
    
    try:
        # Create pipeline configuration
        config = PipelineConfig(
            quality_mode=quality_mode,
            enable_caching=True,
            enable_qdrant=True,
            auto_populate_qdrant=True,
            max_workers=4,
            cleanup_after_generation=True
        )
        
        # Initialize pipeline
        logger.info("Initializing choreography pipeline...")
        pipeline = ChoreoGenerationPipeline(config)
        
        # Check Qdrant status
        qdrant_status = pipeline.get_qdrant_health_status()
        if qdrant_status.get("enabled"):
            logger.info(f"Qdrant status: {qdrant_status.get('status', 'unknown')}")
        
        # Generate choreography
        logger.info("🚀 Starting choreography generation...")
        start_time = time.time()
        
        result = await pipeline.generate_choreography(
            audio_input=str(song_path),
            difficulty=difficulty,
            energy_level=energy_level,
            role_focus=role_focus,
            move_types=move_types,
            tempo_range=tempo_range
        )
        
        generation_time = time.time() - start_time
        
        if result.success:
            print("\n✅ CHOREOGRAPHY GENERATION SUCCESSFUL!")
            print("=" * 50)
            print(f"🎬 Output Video: {result.output_path}")
            print(f"📝 Metadata: {result.metadata_path}")
            print(f"⏱️  Total Time: {generation_time:.2f}s")
            print(f"🎯 Processing Time: {result.processing_time:.2f}s")
            print(f"💃 Moves Analyzed: {result.moves_analyzed}")
            print(f"🎯 Recommendations: {result.recommendations_generated}")
            print(f"⏱️  Sequence Duration: {result.sequence_duration:.1f}s")
            
            # File size info
            if result.output_path and Path(result.output_path).exists():
                file_size = Path(result.output_path).stat().st_size / (1024 * 1024)
                print(f"📁 File Size: {file_size:.1f} MB")
            
            # Qdrant statistics
            if result.qdrant_enabled:
                print(f"🔍 Qdrant Embeddings Stored: {result.qdrant_embeddings_stored}")
                print(f"📥 Qdrant Embeddings Retrieved: {result.qdrant_embeddings_retrieved}")
                print(f"⚡ Qdrant Search Time: {result.qdrant_search_time:.3f}s")
            
            # Cache statistics
            print(f"💾 Cache Hits: {result.cache_hits}")
            print(f"🔄 Cache Misses: {result.cache_misses}")
            
            print("=" * 50)
            print("🎉 Choreography ready! Check the output video.")
            
            return True
            
        else:
            print("\n❌ CHOREOGRAPHY GENERATION FAILED!")
            print("=" * 50)
            print(f"Error: {result.error_message}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print("=" * 50)
            
            return False
            
    except Exception as e:
        logger.error(f"Generation failed with exception: {e}")
        print(f"\n❌ GENERATION FAILED: {e}")
        return False

def parse_move_types(move_types_str: str) -> List[str]:
    """Parse comma-separated move types string."""
    if not move_types_str:
        return None
    
    move_types = [mt.strip() for mt in move_types_str.split(',')]
    
    # Validate move types
    invalid_types = [mt for mt in move_types if mt not in AVAILABLE_MOVE_TYPES]
    if invalid_types:
        logger.warning(f"Invalid move types: {invalid_types}")
        logger.info(f"Available types: {', '.join(AVAILABLE_MOVE_TYPES)}")
    
    return [mt for mt in move_types if mt in AVAILABLE_MOVE_TYPES]

def parse_tempo_range(tempo_range_str: str) -> List[int]:
    """Parse tempo range string like '120,140' or '120-140'."""
    if not tempo_range_str:
        return None
    
    # Handle both comma and dash separators
    if ',' in tempo_range_str:
        parts = tempo_range_str.split(',')
    elif '-' in tempo_range_str:
        parts = tempo_range_str.split('-')
    else:
        logger.error("Tempo range should be in format 'min,max' or 'min-max'")
        return None
    
    try:
        min_tempo = int(parts[0].strip())
        max_tempo = int(parts[1].strip())
        
        if min_tempo >= max_tempo:
            logger.error("Minimum tempo must be less than maximum tempo")
            return None
        
        if min_tempo < 80 or max_tempo > 180:
            logger.error("Tempo range should be between 80 and 180 BPM")
            return None
        
        return [min_tempo, max_tempo]
        
    except (ValueError, IndexError):
        logger.error("Invalid tempo range format. Use 'min,max' or 'min-max'")
        return None

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate Bachata choreography with configurable parameters"
    )
    
    parser.add_argument("--song", type=str, default=DEFAULT_SETTINGS["song"],
                       help=f"Song name (default: {DEFAULT_SETTINGS['song']})")
    parser.add_argument("--difficulty", type=str, default=DEFAULT_SETTINGS["difficulty"],
                       choices=["beginner", "intermediate", "advanced"],
                       help=f"Difficulty level (default: {DEFAULT_SETTINGS['difficulty']})")
    parser.add_argument("--energy", type=str, choices=["low", "medium", "high"],
                       help="Energy level (default: auto-detect)")
    parser.add_argument("--role-focus", type=str, choices=["lead_focus", "follow_focus", "both"],
                       help="Role focus (default: both)")
    parser.add_argument("--quality", type=str, default=DEFAULT_SETTINGS["quality_mode"],
                       choices=["fast", "balanced", "high_quality"],
                       help=f"Quality mode (default: {DEFAULT_SETTINGS['quality_mode']})")
    parser.add_argument("--move-types", type=str,
                       help="Comma-separated list of move types (e.g., 'basic_step,cross_body_lead')")
    parser.add_argument("--tempo-range", type=str,
                       help="Tempo range in BPM (e.g., '120,140' or '120-140')")
    parser.add_argument("--preset", type=str, choices=list(PRESET_CONFIGURATIONS.keys()),
                       help="Use a preset configuration")
    parser.add_argument("--list-options", action="store_true",
                       help="List all available options and exit")
    
    args = parser.parse_args()
    
    if args.list_options:
        display_available_options()
        return
    
    # Parse complex arguments
    move_types = parse_move_types(args.move_types)
    tempo_range = parse_tempo_range(args.tempo_range)
    
    # Generate choreography
    success = await generate_choreography(
        song=args.song,
        difficulty=args.difficulty,
        energy_level=args.energy,
        role_focus=args.role_focus,
        move_types=move_types,
        tempo_range=tempo_range,
        quality_mode=args.quality,
        preset=args.preset
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())