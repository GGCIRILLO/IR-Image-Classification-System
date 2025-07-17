#!/usr/bin/env python3
"""
Quick test script for run_mission.py

Tests the mission runner with various configurations to ensure it works correctly.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_help():
    """Test help output."""
    print("Testing help output...")
    result = subprocess.run([
        sys.executable, "scripts/run_mission.py", "--help"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Help command works")
        return True
    else:
        print(f"❌ Help command failed: {result.stderr}")
        return False


def test_basic_validation():
    """Test input validation."""
    print("\nTesting input validation...")
    
    # Test with non-existent image
    result = subprocess.run([
        sys.executable, "scripts/run_mission.py",
        "--image", "nonexistent.png",
        "--database", "data/chroma_db_final"
    ], capture_output=True, text=True)
    
    if result.returncode != 0 and "not found" in result.stderr:
        print("✅ Input validation works (correctly rejects non-existent files)")
        return True
    else:
        print(f"❌ Input validation failed: {result.stderr}")
        return False


def test_with_sample_image():
    """Test with a real image if available."""
    print("\nTesting with sample image...")
    
    # Look for sample images
    sample_paths = [
        "data/processed/M-1A1 Abrams Tank/image_001.png",
        "data/processed/*/image_001.png",
        "data/processed/*/*"
    ]
    
    sample_image = None
    for pattern in sample_paths:
        matches = list(Path().glob(pattern))
        if matches:
            # Find first PNG or JPG file
            for match in matches:
                if match.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    sample_image = str(match)
                    break
            if sample_image:
                break
    
    if not sample_image:
        print("⚠️  No sample images found - skipping real image test")
        return True
    
    if not Path("data/chroma_db_final").exists():
        print("⚠️  Database not found - skipping real image test")
        return True
    
    print(f"Using sample image: {sample_image}")
    
    # Test with basic configuration
    result = subprocess.run([
        sys.executable, "scripts/run_mission.py",
        "--image", sample_image,
        "--database", "data/chroma_db_final",
        "--quiet",
        "--format", "json",
        "--max-results", "3"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Real image test works")
        print(f"Sample output: {result.stdout[:200]}...")
        return True
    else:
        print(f"❌ Real image test failed: {result.stderr}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing run_mission.py script")
    print("=" * 50)
    
    tests = [
        test_help,
        test_basic_validation,
        test_with_sample_image
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! run_mission.py is ready to use.")
        print("\nExample usage:")
        print("python scripts/run_mission.py --image query.png --database data/chroma_db_final")
    else:
        print("⚠️  Some tests failed. Check the implementation.")


if __name__ == "__main__":
    main()
