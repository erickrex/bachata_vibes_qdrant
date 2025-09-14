#!/usr/bin/env python3
"""
Setup script for Bachata video annotation framework.
Validates existing data, creates directory structure, and generates reports.
"""

import sys
from pathlib import Path

# Add app to path for imports
sys.path.append(str(Path(__file__).parent))

from app.services.annotation_validator import AnnotationValidator
from app.services.directory_organizer import DirectoryOrganizer  
from app.services.annotation_interface import AnnotationInterface


def main():
    """Main setup function."""
    print("🎵 Bachata Choreography Generator - Annotation Framework Setup")
    print("=" * 60)
    
    # Initialize services
    validator = AnnotationValidator(data_dir="data")
    organizer = DirectoryOrganizer(data_dir="data")
    interface = AnnotationInterface(data_dir="data")
    
    # Step 1: Validate existing annotations
    print("\n📋 Step 1: Validating existing annotation data...")
    try:
        validation_report = validator.generate_validation_report()
        
        # Save validation report
        with open("data/validation_report.md", "w") as f:
            f.write(validation_report)
        
        print("✓ Validation complete - Report saved to: data/validation_report.md")
        
        # Show summary
        validation_result = validator.validate_collection()
        if validation_result["success"]:
            summary = validation_result["summary"]
            print(f"  • Total clips: {summary['total_clips']}")
            print(f"  • Videos with issues: {summary['videos_with_issues']}")
            print(f"  • Annotations with issues: {summary['annotations_with_issues']}")
            print(f"  • Overall quality: {summary['overall_quality']}")
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Step 2: Analyze directory structure
    print("\n📁 Step 2: Analyzing directory structure...")
    try:
        organization_report = organizer.generate_organization_report()
        
        # Save organization report
        with open("data/organization_report.md", "w") as f:
            f.write(organization_report)
        
        print("✓ Directory analysis complete - Report saved to: data/organization_report.md")
        
        # Show current structure summary
        current_structure = organizer.analyze_current_structure()
        if current_structure.get("exists", False):
            print(f"  • Total video files: {current_structure.get('total_files', 0)}")
            print(f"  • Organization status: {current_structure.get('organization_status', 'unknown')}")
            print(f"  • Existing directories: {len(current_structure.get('directories', []))}")
        else:
            print(f"  • Issue: {current_structure.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"✗ Directory analysis failed: {e}")
    
    # Step 3: Create annotation tools
    print("\n🛠️  Step 3: Setting up annotation tools...")
    try:
        # Create CSV template and instructions
        if interface.create_annotation_template():
            print("✓ Annotation template created: data/annotation_template.csv")
            print("✓ Instructions created: data/annotation_instructions.md")
        
        # Export existing annotations to CSV for editing
        if interface.export_to_csv():
            print("✓ Current annotations exported: data/bachata_annotations.csv")
        
    except Exception as e:
        print(f"✗ Annotation tools setup failed: {e}")
    
    # Step 4: Create directory structure (dry run)
    print("\n🏗️  Step 4: Preparing directory organization...")
    try:
        dir_result = organizer.create_directory_structure(dry_run=True)
        
        if dir_result["success"]:
            if dir_result["created_directories"]:
                print("✓ Directory structure ready to create:")
                for directory in dir_result["created_directories"]:
                    print(f"  • {directory}")
            
            if dir_result["existing_directories"]:
                print("✓ Existing directories found:")
                for directory in dir_result["existing_directories"]:
                    print(f"  • {directory}")
        else:
            print("✗ Directory structure issues:")
            for error in dir_result["errors"]:
                print(f"  • {error}")
        
    except Exception as e:
        print(f"✗ Directory preparation failed: {e}")
    
    # Summary and next steps
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    print("\n✅ Created Files:")
    created_files = [
        "data/validation_report.md - Quality assessment of current annotations",
        "data/organization_report.md - Directory structure analysis", 
        "data/annotation_template.csv - Template for new annotations",
        "data/annotation_instructions.md - Detailed annotation guide",
        "data/bachata_annotations.csv - Current annotations in CSV format"
    ]
    
    for file_desc in created_files:
        print(f"  • {file_desc}")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review validation_report.md to identify any data quality issues")
    print("2. Review organization_report.md to understand current file organization")
    print("3. Use annotation_template.csv to add new move annotations")
    print("4. Edit bachata_annotations.csv to enhance existing annotations")
    print("5. Run directory organization when ready to restructure files")
    
    print("\n💡 USAGE TIPS:")
    print("• Use the CSV files for bulk editing annotations")
    print("• Follow annotation_instructions.md for consistent labeling")
    print("• Validate changes by re-running this setup script")
    print("• Keep backups before reorganizing directory structure")
    
    print("\n🎵 Annotation framework setup complete!")


if __name__ == "__main__":
    main()