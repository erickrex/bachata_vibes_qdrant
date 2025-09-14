#!/usr/bin/env python3
"""
Test script for YouTube audio download service.
Run this from the root of the bachata-choreography-generator project.
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from services.youtube_service import YouTubeService


async def main():
    """Test the YouTube service and download multiple Bachata songs."""
    
    print("🎵 YouTube Service Test & Song Downloader")
    print("=" * 60)
    
    # Initialize service with custom output directory
    service = YouTubeService(output_dir="data/songs")
    
    # Songs to be downloaded
    to_be_downloaded = [
        'https://www.youtube.com/watch?v=mq8mZEA9Jus',
        'https://www.youtube.com/watch?v=76P_63F1_SY',
        'https://www.youtube.com/watch?v=HW6hPM0qxPU',
        'https://www.youtube.com/watch?v=OxfFmMkN79Q',
        'https://www.youtube.com/watch?v=dsbbce1zy9s',
        'https://www.youtube.com/watch?v=OjEM7R0MUGs',
        'https://www.youtube.com/watch?v=TheGR-r8vl4',
        'https://www.youtube.com/watch?v=n6yzAKHUn5M',
        'https://www.youtube.com/watch?v=KhHdsUTbGHg',
        'https://www.youtube.com/watch?v=Vuk2mkjx5vw',
    ]
    
    # Test URLs for validation
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (good for testing)
        "https://youtu.be/dQw4w9WgXcQ",  # Short format
        "https://www.google.com",  # Invalid
        "not-a-url",  # Invalid
        "",  # Invalid
    ]
    
    print("\n1. Testing URL validation:")
    for url in test_urls:
        is_valid = service.validate_url(url)
        status = "✅" if is_valid else "❌"
        print(f"   {status} {url or '(empty)'}")
    
    print("\n2. Testing video ID extraction:")
    valid_urls = [url for url in test_urls[:2] if url]
    for url in valid_urls:
        video_id = service.extract_video_id(url)
        print(f"   🎬 {url} -> ID: {video_id}")
    
    print("\n3. Downloading Bachata songs:")
    successful_downloads = 0
    failed_downloads = 0
    
    for i, url in enumerate(to_be_downloaded, 1):
        print(f"\n   📥 [{i}/{len(to_be_downloaded)}] Downloading from: {url}")
        result = await service.download_audio(url)
        
        if result.success:
            successful_downloads += 1
            print(f"   ✅ Success!")
            print(f"   📝 Title: {result.title}")
            print(f"   ⏱️  Duration: {result.duration} seconds ({result.duration/60:.1f} minutes)")
            print(f"   📁 File: {result.file_path}")
            
            # Check file size
            if result.file_path:
                file_path = Path(result.file_path)
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    print(f"   📊 File size: {file_size / 1024 / 1024:.2f} MB")
                    
                    if file_size > 100000:  # At least 100KB
                        print("   ✅ File size looks good")
                    else:
                        print("   ⚠️  File size seems small")
                else:
                    print("   ❌ File not found")
        else:
            failed_downloads += 1
            print(f"   ❌ Failed: {result.error_message}")
    
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {successful_downloads}")
    print(f"   ❌ Failed: {failed_downloads}")
    print(f"   📈 Success rate: {(successful_downloads/len(to_be_downloaded)*100):.1f}%")
    
    print("\n4. Testing error handling:")
    invalid_result = await service.download_audio("https://www.google.com")
    if not invalid_result.success:
        print(f"   ✅ Invalid URL correctly rejected: {invalid_result.error_message}")
    else:
        print("   ❌ Should have failed for invalid URL")
    
    print("\n5. File management:")
    print("   💾 Songs saved in data/songs/ directory")
    print("   ℹ️  Files will be kept for future use (no cleanup)")
    
    print("\n" + "=" * 60)
    print("✅ YouTube Service test and song download completed!")


if __name__ == "__main__":
    asyncio.run(main())