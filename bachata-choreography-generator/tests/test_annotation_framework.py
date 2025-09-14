#!/usr/bin/env python3
"""
Test script for the annotation framework.
Validates that all components work correctly with existing data.
"""

import sys
from pathlib import Path

# Add app to path for imports
sys.path.append(str(Path(__file__).parent))

from app.services.annotation_validator import AnnotationValidator
from app.services.directory_organizer import DirectoryOrganizer  
from app.services.annotation_interface import AnnotationInterface
from app.models.annotation_schema import AnnotationCollection, MoveAnnotation


def test_annotation_loading():
    """Test loading and validating annotation data."""
    print("🧪 Testing annotation loading...")
    
    try:
        interface = AnnotationInterface(data_dir="data")
        collection = interface.load_annotations("bachata_annotations.json")
        
        print(f"✓ Loaded {collection.total_clips} clips")
        print(f"✓ Found {len(collection.move_categories)} categories")
        
        # Test individual clip access
        first_clip = collection.clips[0]
        print(f"✓ First clip: {first_clip.clip_id} - {first_clip.move_label}")
        
        # Test category filtering
        basic_clips = collection.get_clips_by_category("basic_step")
        print(f"✓ Found {len(basic_clips)} basic step clips")
        
        return True
    except Exception as e:
        print(f"✗ Annotation loading failed: {e}")
        return False


def test_csv_export_import():
    """Test CSV export and import functionality."""
    print("\n🧪 Testing CSV export/import...")
    
    try:
        interface = AnnotationInterface(data_dir="data")
        
        # Test export
        if interface.export_to_csv("bachata_annotations.json", "test_export.csv"):
            print("✓ CSV export successful")
        else:
            print("✗ CSV export failed")
            return False
        
        # Test import (should recreate the same data)
        if interface.import_from_csv("test_export.csv", "test_import.json"):
            print("✓ CSV import successful")
        else:
            print("✗ CSV import failed")
            return False
        
        # Verify data integrity
        original = interface.load_annotations("bachata_annotations.json")
        imported = interface.load_annotations("test_import.json")
        
        if original.total_clips == imported.total_clips:
            print(f"✓ Data integrity verified: {original.total_clips} clips")
        else:
            print(f"✗ Data integrity issue: {original.total_clips} vs {imported.total_clips}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ CSV export/import failed: {e}")
        return False


def test_annotation_validation():
    """Test annotation validation functionality."""
    print("\n🧪 Testing annotation validation...")
    
    try:
        validator = AnnotationValidator(data_dir="data")
        
        # Test loading annotations
        collection = validator.load_annotations("bachata_annotations.json")
        print(f"✓ Validator loaded {collection.total_clips} clips")
        
        # Test individual annotation validation
        first_clip = collection.clips[0]
        validation_result = validator.validate_annotation_data(first_clip)
        
        print(f"✓ Validated clip: {validation_result['clip_id']}")
        if validation_result['issues']:
            print(f"  Issues found: {len(validation_result['issues'])}")
        if validation_result['warnings']:
            print(f"  Warnings: {len(validation_result['warnings'])}")
        
        return True
    except Exception as e:
        print(f"✗ Annotation validation failed: {e}")
        return False


def test_directory_organization():
    """Test directory organization functionality."""
    print("\n🧪 Testing directory organization...")
    
    try:
        organizer = DirectoryOrganizer(data_dir="data")
        
        # Test structure analysis
        structure = organizer.analyze_current_structure()
        print(f"✓ Structure analysis: {structure.get('organization_status', 'unknown')}")
        
        # Test organization simulation
        org_result = organizer.organize_clips_by_annotations("bachata_annotations.json", dry_run=True)
        
        if org_result["success"]:
            summary = org_result["summary"]
            print(f"✓ Organization simulation successful")
            print(f"  Total clips: {summary['total_clips']}")
            print(f"  Would move: {summary['moved']}")
            print(f"  Already organized: {summary['already_organized']}")
            print(f"  Missing files: {summary['missing_files']}")
        else:
            print(f"✗ Organization simulation failed: {org_result.get('error', 'Unknown error')}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Directory organization failed: {e}")
        return False


def test_annotation_schema():
    """Test annotation schema validation."""
    print("\n🧪 Testing annotation schema...")
    
    try:
        # Test creating a valid annotation
        test_annotation = MoveAnnotation(
            clip_id="test_clip_1",
            video_path="test/path.mp4",
            move_label="basic_step",
            energy_level="medium",
            estimated_tempo=120,
            difficulty="beginner",
            lead_follow_roles="both",
            notes="Test annotation for validation"
        )
        
        print(f"✓ Created valid annotation: {test_annotation.clip_id}")
        print(f"✓ Auto-derived category: {test_annotation.category}")
        
        # Test collection creation
        test_collection = AnnotationCollection(
            instructions="Test collection",
            move_categories=["basic_step"],
            clips=[test_annotation]
        )
        
        print(f"✓ Created collection with {test_collection.total_clips} clips")
        
        return True
    except Exception as e:
        print(f"✗ Schema validation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🎵 Bachata Annotation Framework - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Annotation Loading", test_annotation_loading),
        ("CSV Export/Import", test_csv_export_import),
        ("Annotation Validation", test_annotation_validation),
        ("Directory Organization", test_directory_organization),
        ("Annotation Schema", test_annotation_schema)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Annotation framework is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)