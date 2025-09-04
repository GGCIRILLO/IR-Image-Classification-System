#!/usr/bin/env python3
"""
Demonstration script for rollback functionality.

This script shows how to use the rollback feature without actually executing it.
"""

import sys
from pathlib import Path

# Add the scripts directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from test_set_manager import TestSetManager

def demo_rollback():
    """Demonstrate rollback functionality without executing."""
    
    print("=== TEST SET ROLLBACK DEMONSTRATION ===\n")
    
    manager = TestSetManager()
    
    # Show current status
    print("Current status:")
    manager.print_status()
    
    print("\n" + "="*60)
    print("ROLLBACK SIMULATION")
    print("="*60)
    
    # Load metadata to show what would be rolled back
    metadata = manager.load_metadata()
    if metadata:
        print(f"If rollback were executed, it would:")
        print(f"  • Move {metadata['total_test_images']} images back to training set")
        print(f"  • Restore {len(metadata['classes'])} class directories")
        print(f"  • Remove the test set directory: data/test_set/")
        print(f"  • Remove metadata file: data/test_set_metadata.json")
        print(f"  • Create backup of metadata before rollback")
        
        print(f"\nExample files that would be moved:")
        count = 0
        for class_name, class_data in metadata['classes'].items():
            if count >= 3:  # Show only first 3 classes
                break
            print(f"  {class_name}:")
            for _, filename in enumerate(class_data['test_files'][:2]):  # Show 2 files per class
                print(f"    • {filename}")
            if len(class_data['test_files']) > 2:
                print(f"    ... and {len(class_data['test_files']) - 2} more")
            count += 1
        
        if len(metadata['classes']) > 3:
            print(f"  ... and {len(metadata['classes']) - 3} more classes")
    
    print(f"\nTo actually perform rollback, run:")
    print(f"  python scripts/test_set_manager.py rollback")
    print(f"\nTo perform rollback without confirmation:")
    print(f"  python scripts/test_set_manager.py rollback --confirm")


if __name__ == "__main__":
    demo_rollback()