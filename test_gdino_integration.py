#!/usr/bin/env python3
"""
Test script for GroundingDINO integration
=========================================
Verify that GroundingDINO is properly set up and working.
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_gdino_installation():
    """Check if GroundingDINO is installed."""
    logger.info("Checking GroundingDINO installation...")
    
    try:
        sys.path.append(str(Path("GroundingDINO")))
        import groundingdino
        logger.info("✓ GroundingDINO module found")
        return True
    except ImportError as e:
        logger.error(f"✗ GroundingDINO not found: {e}")
        logger.error("Please install: cd GroundingDINO && pip install -e .")
        return False


def check_gdino_files():
    """Check if GroundingDINO config and weights exist."""
    logger.info("\nChecking GroundingDINO files...")
    
    config_path = Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    weights_path = Path("GroundingDINO/weights/groundingdino_swint_ogc.pth")
    
    all_ok = True
    
    if config_path.exists():
        logger.info(f"✓ Config found: {config_path}")
    else:
        logger.error(f"✗ Config not found: {config_path}")
        all_ok = False
    
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Weights found: {weights_path} ({size_mb:.1f} MB)")
    else:
        logger.error(f"✗ Weights not found: {weights_path}")
        logger.error("Download from: https://github.com/IDEA-Research/GroundingDINO/releases")
        all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability."""
    logger.info("\nChecking CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            logger.warning("⚠ CUDA not available, will use CPU")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking CUDA: {e}")
        return False


def test_gdino_inference():
    """Test GroundingDINO inference on a sample image."""
    logger.info("\nTesting GroundingDINO inference...")
    
    try:
        from generate_coco_with_gdino import COCOConfig, GroundingDINODetector
        from PIL import Image
        import numpy as np
        
        # Create a test image
        test_img_path = Path("test_gdino_image.jpg")
        if not test_img_path.exists():
            logger.info("Creating test image...")
            img = Image.new('RGB', (640, 480), color=(128, 128, 128))
            img.save(test_img_path)
        
        # Create config
        config = COCOConfig(
            gdino_device="cuda" if check_cuda() else "cpu"
        )
        
        # Test detection
        detector = GroundingDINODetector(config)
        
        logger.info("Loading GroundingDINO model...")
        detector.load_model()
        logger.info("✓ Model loaded successfully")
        
        logger.info("Running test detection...")
        test_categories = ["person", "car", "dog"]
        results = detector.detect_boxes(test_img_path, test_categories)
        
        logger.info(f"✓ Detection completed: {len(results)} objects found")
        
        # Cleanup
        if test_img_path.exists():
            test_img_path.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_vlm_server():
    """Check if VLM server is running."""
    logger.info("\nChecking VLM server...")
    
    uds_path = Path("/tmp/vllm-server.sock")
    if uds_path.exists():
        logger.info("✓ VLM server appears to be running")
        return True
    else:
        logger.warning("⚠ VLM server not running")
        logger.info("Start with: cd preprocessor && bash start_vllm_server.sh")
        return False


def main():
    """Run all checks."""
    logger.info("=" * 80)
    logger.info("GroundingDINO Integration Test")
    logger.info("=" * 80)
    
    checks = {
        "GroundingDINO Installation": check_gdino_installation(),
        "GroundingDINO Files": check_gdino_files(),
        "CUDA Support": check_cuda(),
        "VLM Server": check_vlm_server(),
    }
    
    # Only test inference if basic checks pass
    if checks["GroundingDINO Installation"] and checks["GroundingDINO Files"]:
        checks["GroundingDINO Inference"] = test_gdino_inference()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{check_name:30s} {status}")
    
    logger.info("=" * 80)
    
    all_passed = all(checks.values())
    
    if all_passed:
        logger.info("\n✓ All checks passed! Ready to generate datasets.")
        logger.info("\nNext steps:")
        logger.info("  1. Add data to input_data/")
        logger.info("  2. Run: python generate_coco_with_gdino.py")
    else:
        logger.warning("\n⚠ Some checks failed. Please fix the issues above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
