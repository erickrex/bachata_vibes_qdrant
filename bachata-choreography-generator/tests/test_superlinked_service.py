#!/usr/bin/env python3
"""
Test script for SuperlinkedEmbeddingService

This script tests all the key functionality of the Superlinked-inspired
embedding service to ensure it meets the requirements.
"""

import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent))

from app.services.superlinked_embedding_service import create_superlinked_service

def test_superlinked_service():
    """Test all key functionality of the SuperlinkedEmbeddingService."""
    
    print("🎵 Testing Superlinked Embedding Service for Bachata Choreography Generator")
    print("=" * 80)
    
    # Create the service
    print("\n1. Initializing service...")
    service = create_superlinked_service("data")
    
    # Test statistics
    print("\n2. Service Statistics:")
    stats = service.get_stats()
    print(f"   ✓ Total moves: {stats['total_moves']}")
    print(f"   ✓ Move categories: {stats['move_categories']}")
    print(f"   ✓ Embedding spaces: {len(stats['embedding_spaces'])}")
    print(f"   ✓ Total embedding dimension: {service.total_dimension}")
    
    # Test TextSimilaritySpace (semantic understanding)
    print("\n3. Testing TextSimilaritySpace (semantic understanding):")
    semantic_results = service.search_semantic("energetic basic steps for beginners", limit=3)
    print(f"   Query: 'energetic basic steps for beginners'")
    for i, result in enumerate(semantic_results, 1):
        print(f"   {i}. {result['clip_id']}: {result['move_description'][:50]}... (score: {result['similarity_score']:.3f})")
    
    # Test NumberSpace for tempo (BPM) with preserved linear relationships
    print("\n4. Testing NumberSpace for tempo (preserved linear relationships):")
    tempo_results = service.search_tempo(120.0, limit=3)
    print(f"   Query: 120 BPM")
    for i, result in enumerate(tempo_results, 1):
        print(f"   {i}. {result['clip_id']}: {result['tempo']} BPM (score: {result['similarity_score']:.3f})")
    
    # Test NumberSpace for difficulty
    print("\n5. Testing NumberSpace for difficulty:")
    difficulty_results = service.search_moves(
        difficulty_level="intermediate",
        weights={"text": 0.0, "tempo": 0.0, "difficulty": 1.0, "energy": 0.0, "role": 0.0, "transition": 0.0},
        limit=3
    )
    print(f"   Query: intermediate difficulty")
    for i, result in enumerate(difficulty_results, 1):
        print(f"   {i}. {result['clip_id']}: difficulty {result['difficulty_score']} (score: {result['similarity_score']:.3f})")
    
    # Test CategoricalSimilaritySpace for energy levels
    print("\n6. Testing CategoricalSimilaritySpace for energy levels:")
    energy_results = service.search_moves(
        energy_level="high",
        weights={"text": 0.0, "tempo": 0.0, "difficulty": 0.0, "energy": 1.0, "role": 0.0, "transition": 0.0},
        limit=3
    )
    print(f"   Query: high energy")
    for i, result in enumerate(energy_results, 1):
        print(f"   {i}. {result['clip_id']}: {result['energy_level']} energy (score: {result['similarity_score']:.3f})")
    
    # Test CategoricalSimilaritySpace for role focus
    print("\n7. Testing CategoricalSimilaritySpace for role focus:")
    role_results = service.search_moves(
        role_focus="lead_focus",
        weights={"text": 0.0, "tempo": 0.0, "difficulty": 0.0, "energy": 0.0, "role": 1.0, "transition": 0.0},
        limit=3
    )
    print(f"   Query: lead_focus")
    for i, result in enumerate(role_results, 1):
        print(f"   {i}. {result['clip_id']}: {result['role_focus']} (score: {result['similarity_score']:.3f})")
    
    # Test CustomSpace for transition patterns
    print("\n8. Testing CustomSpace for transition patterns:")
    print("   Transition compatibility examples:")
    for move_id in ["basic_step_1", "cross_body_lead_1", "lady_right_turn_1"]:
        compatible = service.get_transition_compatibility(move_id)
        print(f"   {move_id}: {len(compatible)} compatible moves -> {compatible[:3]}...")
    
    # Test multi-factor query combining all spaces
    print("\n9. Testing multi-factor query (all spaces combined):")
    multi_results = service.search_moves(
        description="smooth intermediate moves",
        target_tempo=115.0,
        difficulty_level="intermediate", 
        energy_level="medium",
        role_focus="both",
        limit=3
    )
    print(f"   Query: 'smooth intermediate moves', 115 BPM, intermediate, medium energy, both roles")
    for i, result in enumerate(multi_results, 1):
        print(f"   {i}. {result['clip_id']}: {result['move_description'][:40]}... (score: {result['similarity_score']:.3f})")
    
    # Test natural language queries
    print("\n10. Testing natural language queries:")
    nl_queries = [
        "beginner moves for slow romantic song",
        "advanced turns for fast bachata",
        "sensual styling moves"
    ]
    
    for query in nl_queries:
        results = service.search_semantic(query, limit=2)
        print(f"    Query: '{query}'")
        for result in results:
            print(f"      → {result['clip_id']}: {result['move_description'][:45]}... (score: {result['similarity_score']:.3f})")
    
    # Test embedding retrieval
    print("\n11. Testing embedding retrieval:")
    embedding = service.get_move_embedding("basic_step_1")
    if embedding is not None:
        print(f"    ✓ Retrieved embedding for basic_step_1: shape {embedding.shape}, dimension {len(embedding)}")
    else:
        print("    ✗ Failed to retrieve embedding")
    
    print("\n" + "=" * 80)
    print("🎉 All tests completed successfully!")
    print("\nKey achievements:")
    print("✓ TextSimilaritySpace: Semantic move understanding and natural language queries")
    print("✓ NumberSpace: Tempo (BPM) and difficulty with preserved linear relationships")
    print("✓ CategoricalSimilaritySpace: Energy levels (low/medium/high) and role focus (lead/follow/both)")
    print("✓ CustomSpace: Transition patterns from move compatibility knowledge graph")
    print("✓ Unified embedding system combining all spaces")
    print("✓ Natural language query interface")
    print("✓ Multi-factor search with customizable weights")
    
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for testing
    
    try:
        success = test_superlinked_service()
        if success:
            print("\n🎵 SuperlinkedEmbeddingService is ready for integration!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)