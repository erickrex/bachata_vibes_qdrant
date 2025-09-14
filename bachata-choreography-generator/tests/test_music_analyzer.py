"""
Test script for MusicAnalyzer to validate tempo detection accuracy with Bachata songs.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from services.music_analyzer import MusicAnalyzer
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_features_to_dict(features, song_name: str, analyzer: MusicAnalyzer) -> Dict[str, Any]:
    """Convert MusicFeatures to a JSON-serializable dictionary."""
    
    # Calculate energy profile statistics
    energy_stats = {
        'min': float(min(features.energy_profile)),
        'max': float(max(features.energy_profile)),
        'mean': float(sum(features.energy_profile) / len(features.energy_profile)),
        'std': float(np.std(features.energy_profile))
    }
    
    # Calculate embedding statistics
    embedding_stats = {
        'min': float(min(features.audio_embedding)),
        'max': float(max(features.audio_embedding)),
        'mean': float(sum(features.audio_embedding) / len(features.audio_embedding)),
        'non_zero_count': sum(1 for x in features.audio_embedding if abs(x) > 1e-6),
        'dimensions': len(features.audio_embedding)
    }
    
    # Convert sections to dictionaries
    sections = []
    for section in features.sections:
        sections.append({
            'start_time': float(section.start_time),
            'end_time': float(section.end_time),
            'section_type': section.section_type,
            'energy_level': float(section.energy_level),
            'tempo_stability': float(section.tempo_stability),
            'recommended_move_types': section.recommended_move_types
        })
    
    # Validate tempo
    tempo_valid = analyzer.validate_tempo_accuracy(features.tempo)
    
    return {
        'song_name': song_name,
        'analysis_results': {
            'duration': float(features.duration),
            'tempo': float(features.tempo),
            'tempo_validation': {
                'is_valid': tempo_valid,
                'is_bachata_range': 90 <= features.tempo <= 150
            },
            'beats_detected': len(features.beat_positions),
            'rhythm_features': {
                'pattern_strength': float(features.rhythm_pattern_strength),
                'syncopation_level': float(features.syncopation_level)
            },
            'spectral_features': {
                'mfcc_shape': list(features.mfcc_features.shape),
                'chroma_shape': list(features.chroma_features.shape),
                'spectral_centroid_shape': list(features.spectral_centroid.shape)
            },
            'energy_profile': energy_stats,
            'audio_embedding': embedding_stats,
            'musical_sections': {
                'count': len(sections),
                'sections': sections
            }
        }
    }


def test_music_analyzer(output_format: str = "text", output_file: str = None):
    """Test the MusicAnalyzer with downloaded Bachata songs."""
    
    # Initialize the analyzer
    analyzer = MusicAnalyzer()
    
    # Path to the songs directory
    songs_dir = Path("data/songs")
    
    if not songs_dir.exists():
        logger.error(f"Songs directory not found: {songs_dir}")
        return
    
    # Get all MP3 files in the songs directory
    mp3_files = list(songs_dir.glob("*.mp3"))
    
    if not mp3_files:
        logger.error("No MP3 files found in the songs directory")
        return
    
    logger.info(f"Found {len(mp3_files)} MP3 files to analyze")
    
    # Sort files for consistent output
    mp3_files.sort()
    
    # Store results for JSON output
    all_results = []
    
    # Test each song
    for song_path in mp3_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {song_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            # Analyze the audio
            features = analyzer.analyze_audio(str(song_path))
            
            if output_format == "json":
                # Convert to dictionary for JSON output
                result_dict = convert_features_to_dict(features, song_path.name, analyzer)
                all_results.append(result_dict)
            else:
                # Display results in text format
                print(f"\nAnalysis Results for '{song_path.name}':")
                print(f"  Duration: {features.duration:.2f} seconds")
                print(f"  Tempo: {features.tempo:.1f} BPM")
                print(f"  Beats detected: {len(features.beat_positions)}")
                print(f"  MFCC shape: {features.mfcc_features.shape}")
                print(f"  Chroma shape: {features.chroma_features.shape}")
                print(f"  Spectral centroid shape: {features.spectral_centroid.shape}")
                print(f"  Audio embedding dimensions: {len(features.audio_embedding)}")
                
                # Validate tempo for Bachata
                is_valid = analyzer.validate_tempo_accuracy(features.tempo)
                print(f"  Tempo validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
                
                # Show energy profile statistics
                energy_stats = {
                    'min': min(features.energy_profile),
                    'max': max(features.energy_profile),
                    'mean': sum(features.energy_profile) / len(features.energy_profile)
                }
                print(f"  Energy profile: min={energy_stats['min']:.4f}, "
                      f"max={energy_stats['max']:.4f}, mean={energy_stats['mean']:.4f}")
                
                # Show Bachata-specific rhythm features
                print(f"  Rhythm pattern strength: {features.rhythm_pattern_strength:.3f}")
                print(f"  Syncopation level: {features.syncopation_level:.3f}")
                
                # Show musical structure
                print(f"  Musical sections detected: {len(features.sections)}")
                for i, section in enumerate(features.sections):
                    print(f"    Section {i+1}: {section.section_type} "
                          f"({section.start_time:.1f}s - {section.end_time:.1f}s, "
                          f"energy: {section.energy_level:.3f})")
                    print(f"      Recommended moves: {', '.join(section.recommended_move_types)}")
                
                # Validate embedding quality
                embedding_stats = {
                    'min': min(features.audio_embedding),
                    'max': max(features.audio_embedding),
                    'mean': sum(features.audio_embedding) / len(features.audio_embedding),
                    'non_zero': sum(1 for x in features.audio_embedding if abs(x) > 1e-6)
                }
                print(f"  Embedding stats: min={embedding_stats['min']:.4f}, "
                      f"max={embedding_stats['max']:.4f}, mean={embedding_stats['mean']:.4f}, "
                      f"non-zero: {embedding_stats['non_zero']}/128")
            
        except Exception as e:
            logger.error(f"Error analyzing {song_path.name}: {e}")
            if output_format == "json":
                all_results.append({
                    'song_name': song_path.name,
                    'error': str(e),
                    'analysis_results': None
                })
            continue
    
    # Handle JSON output
    if output_format == "json":
        json_output = {
            'analysis_summary': {
                'total_songs': len(mp3_files),
                'successful_analyses': len([r for r in all_results if 'error' not in r]),
                'failed_analyses': len([r for r in all_results if 'error' in r])
            },
            'results': all_results
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            print(f"\nJSON results saved to: {output_file}")
        else:
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
    
    logger.info(f"\n{'='*60}")
    logger.info("Music analysis testing completed!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MusicAnalyzer with Bachata songs")
    parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file path for JSON format (optional, prints to stdout if not specified)"
    )
    
    args = parser.parse_args()
    
    test_music_analyzer(output_format=args.format, output_file=args.output)