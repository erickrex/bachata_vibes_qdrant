#!/usr/bin/env python3
"""
Setup script for local Qdrant vector database instance.
Provides easy deployment using Docker and initial data migration.
"""

import subprocess
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add app to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from app.services.qdrant_service import create_qdrant_service, QdrantConfig, setup_local_qdrant_docker
    from app.services.annotation_interface import AnnotationInterface
    from app.services.move_analyzer import MoveAnalyzer
    from app.services.music_analyzer import MusicAnalyzer
    from app.services.feature_fusion import FeatureFusion
    from app.services.recommendation_engine import MoveCandidate
    SERVICES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import services: {e}")
    SERVICES_AVAILABLE = False


class QdrantSetupManager:
    """Manager for setting up and configuring Qdrant vector database."""
    
    def __init__(self):
        self.docker_container_name = "bachata-qdrant"
        self.qdrant_port = 6333
        self.storage_dir = Path("qdrant_storage")
        
    def check_docker_available(self) -> bool:
        """Check if Docker is available on the system."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Docker available: {result.stdout.strip()}")
                return True
            else:
                logger.error("Docker command failed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("Docker not found or not responding")
            return False
    
    def check_qdrant_running(self) -> bool:
        """Check if Qdrant container is already running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.docker_container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            running_containers = result.stdout.strip().split('\n')
            is_running = self.docker_container_name in running_containers
            
            if is_running:
                logger.info(f"Qdrant container '{self.docker_container_name}' is already running")
            else:
                logger.info(f"Qdrant container '{self.docker_container_name}' is not running")
            
            return is_running
            
        except Exception as e:
            logger.error(f"Error checking Qdrant status: {e}")
            return False
    
    def start_qdrant_container(self) -> bool:
        """Start Qdrant Docker container."""
        if self.check_qdrant_running():
            return True
        
        # Create storage directory
        self.storage_dir.mkdir(exist_ok=True)
        
        # Docker run command
        docker_cmd = [
            "docker", "run", "-d",
            "--name", self.docker_container_name,
            "-p", f"{self.qdrant_port}:6333",
            "-p", "6334:6334",
            "-v", f"{self.storage_dir.absolute()}:/qdrant/storage:z",
            "qdrant/qdrant"
        ]
        
        try:
            logger.info("Starting Qdrant Docker container...")
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Qdrant container started successfully: {result.stdout.strip()}")
                
                # Wait for Qdrant to be ready
                return self.wait_for_qdrant_ready()
            else:
                logger.error(f"Failed to start Qdrant container: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Qdrant container: {e}")
            return False
    
    def wait_for_qdrant_ready(self, timeout: int = 30) -> bool:
        """Wait for Qdrant to be ready to accept connections."""
        logger.info("Waiting for Qdrant to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to connect to Qdrant
                if SERVICES_AVAILABLE:
                    config = QdrantConfig(host="localhost", port=self.qdrant_port)
                    service = create_qdrant_service(config)
                    health = service.health_check()
                    
                    if health.get("qdrant_available", False):
                        logger.info("✅ Qdrant is ready!")
                        return True
                else:
                    # Fallback: try HTTP request
                    import urllib.request
                    import json
                    
                    url = f"http://localhost:{self.qdrant_port}/collections"
                    response = urllib.request.urlopen(url, timeout=5)
                    if response.status == 200:
                        logger.info("✅ Qdrant is ready!")
                        return True
                
            except Exception:
                pass  # Continue waiting
            
            time.sleep(2)
        
        logger.error(f"❌ Qdrant not ready after {timeout} seconds")
        return False
    
    def stop_qdrant_container(self) -> bool:
        """Stop Qdrant Docker container."""
        try:
            logger.info("Stopping Qdrant container...")
            result = subprocess.run(
                ["docker", "stop", self.docker_container_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("Qdrant container stopped")
                return True
            else:
                logger.error(f"Failed to stop container: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping Qdrant container: {e}")
            return False
    
    def remove_qdrant_container(self) -> bool:
        """Remove Qdrant Docker container."""
        try:
            # Stop first
            self.stop_qdrant_container()
            
            logger.info("Removing Qdrant container...")
            result = subprocess.run(
                ["docker", "rm", self.docker_container_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("Qdrant container removed")
                return True
            else:
                logger.error(f"Failed to remove container: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing Qdrant container: {e}")
            return False
    
    def migrate_existing_data(self) -> Dict[str, Any]:
        """Migrate existing move data to Qdrant."""
        if not SERVICES_AVAILABLE:
            logger.error("Services not available for data migration")
            return {"error": "Services not available"}
        
        logger.info("Starting data migration to Qdrant...")
        
        try:
            # Initialize services
            annotation_interface = AnnotationInterface(data_dir="data")
            move_analyzer = MoveAnalyzer(target_fps=10, min_detection_confidence=0.3)
            music_analyzer = MusicAnalyzer()
            feature_fusion = FeatureFusion()
            
            # Load annotations
            collection = annotation_interface.load_annotations("bachata_annotations.json")
            logger.info(f"Loaded {collection.total_clips} move clips")
            
            # Analyze a sample song for embeddings
            sample_songs = list(Path("data/songs").glob("*.mp3"))
            if not sample_songs:
                logger.error("No sample songs found for embedding generation")
                return {"error": "No sample songs available"}
            
            sample_song = sample_songs[0]
            logger.info(f"Using sample song for embeddings: {sample_song.name}")
            music_features = music_analyzer.analyze_audio(str(sample_song))
            
            # Create move candidates
            move_candidates = []
            for i, clip in enumerate(collection.clips[:10]):  # Limit to first 10 for testing
                try:
                    video_path = Path("data") / clip.video_path
                    if not video_path.exists():
                        continue
                    
                    logger.info(f"Analyzing move {i+1}/10: {clip.move_label}")
                    
                    # Analyze move
                    analysis_result = move_analyzer.analyze_move_clip(str(video_path))
                    
                    # Create multimodal embedding
                    multimodal_embedding = feature_fusion.create_multimodal_embedding(
                        music_features, analysis_result
                    )
                    
                    # Create candidate
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
                    logger.warning(f"Failed to analyze {clip.clip_id}: {e}")
                    continue
            
            if not move_candidates:
                logger.error("No move candidates created")
                return {"error": "No move candidates created"}
            
            # Initialize Qdrant service
            config = QdrantConfig(host="localhost", port=self.qdrant_port)
            qdrant_service = create_qdrant_service(config)
            
            # Migrate data
            migration_result = qdrant_service.migrate_from_memory_cache(move_candidates)
            
            logger.info(f"✅ Data migration completed: {migration_result}")
            return migration_result
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return {"error": str(e)}
    
    def test_qdrant_performance(self) -> Dict[str, Any]:
        """Test Qdrant performance with sample queries."""
        if not SERVICES_AVAILABLE:
            logger.error("Services not available for performance testing")
            return {"error": "Services not available"}
        
        logger.info("Testing Qdrant performance...")
        
        try:
            config = QdrantConfig(host="localhost", port=self.qdrant_port)
            qdrant_service = create_qdrant_service(config)
            
            # Get collection info
            collection_info = qdrant_service.get_collection_info()
            logger.info(f"Collection info: {collection_info}")
            
            if collection_info.get("points_count", 0) == 0:
                logger.warning("No data in collection for performance testing")
                return {"error": "No data in collection"}
            
            # Perform test searches
            import numpy as np
            
            test_results = []
            for i in range(5):
                # Generate random query vector
                query_vector = np.random.random(512)  # 512D vector
                
                start_time = time.time()
                results = qdrant_service.search_similar_moves(
                    query_embedding=query_vector,
                    limit=10
                )
                search_time = (time.time() - start_time) * 1000  # Convert to ms
                
                test_results.append({
                    "search_id": i + 1,
                    "search_time_ms": search_time,
                    "results_count": len(results)
                })
                
                logger.info(f"Search {i+1}: {search_time:.2f}ms, {len(results)} results")
            
            # Calculate statistics
            search_times = [r["search_time_ms"] for r in test_results]
            performance_stats = {
                "total_searches": len(test_results),
                "avg_search_time_ms": sum(search_times) / len(search_times),
                "min_search_time_ms": min(search_times),
                "max_search_time_ms": max(search_times),
                "collection_points": collection_info.get("points_count", 0),
                "individual_results": test_results
            }
            
            logger.info(f"✅ Performance test completed: avg {performance_stats['avg_search_time_ms']:.2f}ms")
            return performance_stats
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return {"error": str(e)}


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Qdrant vector database for Bachata choreography")
    parser.add_argument("action", choices=["start", "stop", "restart", "remove", "migrate", "test", "status"],
                       help="Action to perform")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port (default: 6333)")
    
    args = parser.parse_args()
    
    # Create setup manager
    setup_manager = QdrantSetupManager()
    setup_manager.qdrant_port = args.port
    
    if args.action == "start":
        print("🚀 Starting Qdrant vector database...")
        
        if not setup_manager.check_docker_available():
            print("❌ Docker is not available. Please install Docker first.")
            print("   Visit: https://docs.docker.com/get-docker/")
            return
        
        if setup_manager.start_qdrant_container():
            print("✅ Qdrant started successfully!")
            print(f"   🌐 Web UI: http://localhost:{args.port}/dashboard")
            print(f"   📡 API: http://localhost:{args.port}")
            print("\n💡 Next steps:")
            print("   1. Run 'python setup_qdrant.py migrate' to load existing data")
            print("   2. Run 'python setup_qdrant.py test' to verify performance")
        else:
            print("❌ Failed to start Qdrant")
    
    elif args.action == "stop":
        print("🛑 Stopping Qdrant...")
        if setup_manager.stop_qdrant_container():
            print("✅ Qdrant stopped")
        else:
            print("❌ Failed to stop Qdrant")
    
    elif args.action == "restart":
        print("🔄 Restarting Qdrant...")
        setup_manager.stop_qdrant_container()
        time.sleep(2)
        if setup_manager.start_qdrant_container():
            print("✅ Qdrant restarted successfully!")
        else:
            print("❌ Failed to restart Qdrant")
    
    elif args.action == "remove":
        print("🗑️  Removing Qdrant container...")
        if setup_manager.remove_qdrant_container():
            print("✅ Qdrant container removed")
        else:
            print("❌ Failed to remove Qdrant container")
    
    elif args.action == "migrate":
        print("📦 Migrating existing data to Qdrant...")
        
        if not setup_manager.check_qdrant_running():
            print("❌ Qdrant is not running. Start it first with: python setup_qdrant.py start")
            return
        
        result = setup_manager.migrate_existing_data()
        if "error" in result:
            print(f"❌ Migration failed: {result['error']}")
        else:
            print(f"✅ Migration completed!")
            print(f"   📊 Migrated: {result.get('successful_migrations', 0)} moves")
            print(f"   ⏱️  Time: {result.get('migration_time_seconds', 0):.2f}s")
    
    elif args.action == "test":
        print("🧪 Testing Qdrant performance...")
        
        if not setup_manager.check_qdrant_running():
            print("❌ Qdrant is not running. Start it first with: python setup_qdrant.py start")
            return
        
        result = setup_manager.test_qdrant_performance()
        if "error" in result:
            print(f"❌ Performance test failed: {result['error']}")
        else:
            print(f"✅ Performance test completed!")
            print(f"   📊 Average search time: {result.get('avg_search_time_ms', 0):.2f}ms")
            print(f"   🎯 Collection size: {result.get('collection_points', 0)} points")
    
    elif args.action == "status":
        print("📊 Qdrant Status:")
        
        if not setup_manager.check_docker_available():
            print("   ❌ Docker: Not available")
            return
        else:
            print("   ✅ Docker: Available")
        
        if setup_manager.check_qdrant_running():
            print("   ✅ Qdrant: Running")
            
            if SERVICES_AVAILABLE:
                try:
                    config = QdrantConfig(host="localhost", port=args.port)
                    service = create_qdrant_service(config)
                    health = service.health_check()
                    collection_info = service.get_collection_info()
                    
                    print(f"   📡 API: {'✅ Accessible' if health.get('qdrant_available') else '❌ Not accessible'}")
                    print(f"   📦 Collection: {'✅ Exists' if health.get('collection_exists') else '❌ Missing'}")
                    print(f"   🎯 Points: {collection_info.get('points_count', 0)}")
                    print(f"   🌐 Web UI: http://localhost:{args.port}/dashboard")
                    
                except Exception as e:
                    print(f"   ⚠️  Status check failed: {e}")
            else:
                print("   ⚠️  Services not available for detailed status")
        else:
            print("   ❌ Qdrant: Not running")
            print("\n💡 To start Qdrant:")
            print("   python setup_qdrant.py start")


if __name__ == "__main__":
    main()