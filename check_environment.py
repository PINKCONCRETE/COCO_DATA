#!/usr/bin/env python3
"""
Environment Check Script
========================
Verify that all dependencies and components are properly set up.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (Need >= 3.8)")
        return False


def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (not installed)")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print("\nChecking Python dependencies...")
    
    packages = [
        ("ultralytics", "ultralytics"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("openai", "openai"),
        ("httpx", "httpx"),
        ("imageio", "imageio"),
        ("tqdm", "tqdm"),
        ("pyyaml", "yaml"),
    ]
    
    all_ok = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_ok = False
    
    return all_ok


def check_vllm_server():
    """Check if vLLM server is running."""
    print("\nChecking vLLM server...")
    uds_path = Path("/tmp/vllm-server.sock")
    
    if uds_path.exists():
        print("  ✓ vLLM server socket found")
        return True
    else:
        print("  ✗ vLLM server not running")
        print("    Start with: cd preprocessor && bash start_vllm_server.sh")
        return False


def check_model_files():
    """Check if model files exist."""
    print("\nChecking model files...")
    
    model_files = [
        "yolo26x.pt",
        "ultralytics/yolo26x.pt",
    ]
    
    found = False
    for model in model_files:
        if Path(model).exists():
            print(f"  ✓ Found: {model}")
            found = True
            break
    
    if not found:
        print("  ⚠ No YOLO model file found")
        print("    Download from: https://github.com/ultralytics/ultralytics")
    
    return found


def check_directories():
    """Check if necessary directories exist."""
    print("\nChecking directories...")
    
    dirs = [
        "input_data",
        "preprocessor",
        "SceneAnnotation",
    ]
    
    all_ok = True
    for dir_name in dirs:
        if Path(dir_name).exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (not found)")
            all_ok = False
    
    return all_ok


def check_scripts():
    """Check if main scripts exist."""
    print("\nChecking scripts...")
    
    scripts = [
        "generate_coco_dataset.py",
        "train_coco.py",
        "validate_coco_dataset.py",
        "quickstart.sh",
    ]
    
    all_ok = True
    for script in scripts:
        if Path(script).exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (not found)")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("=" * 80)
    print("Environment Check for COCO Dataset Generator")
    print("=" * 80)
    
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Model Files": check_model_files(),
        "Directories": check_directories(),
        "Scripts": check_scripts(),
        "vLLM Server": check_vllm_server(),
    }
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:20s} {status}")
    
    all_passed = all(checks.values())
    
    print("=" * 80)
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to generate datasets.")
        print("\nNext steps:")
        print("  1. Add videos/images to input_data/")
        print("  2. Ensure vLLM server is running")
        print("  3. Run: python generate_coco_dataset.py")
    else:
        print("\n⚠ Some checks failed. Please address the issues above.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements_coco.txt")
    
    print("")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
