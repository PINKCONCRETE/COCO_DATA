#!/usr/bin/env python3
"""
YOLO Inference Script
=====================
Run inference using a trained YOLO model on separate images or videos.

Usage:
    python predict_coco.py --model runs/detect/train/weights/best.pt --source input_video.mp4
"""

import argparse
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video")
    parser.add_argument("--output", type=str, default="inference_output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--show", action="store_true", help="Show results in window")
    parser.add_argument("--save-txt", action="store_true", help="Save results to txt")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True
    )
    
    print(f"\nResults saved to: {output_dir}")
    
    if args.show:
        # For video/image display if requested
        pass  # ultralytics 'show=True' handles this during predict usually, 
              # but often fails in headless envs.

if __name__ == "__main__":
    main()
