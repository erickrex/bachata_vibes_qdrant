#!/usr/bin/env python3
"""
Startup script for the Bachata Choreography Generator FastAPI application.
"""
import uvicorn
import logging
from pathlib import Path

def main():
    """Start the FastAPI server with appropriate configuration."""
    
    # Ensure we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: main.py not found. Please run this script from the project root directory.")
        return 1
    
    print("🚀 Starting Bachata Choreography Generator API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🎵 Web interface at: http://localhost:8000")
    print("\n" + "="*50)
    
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    exit(main())