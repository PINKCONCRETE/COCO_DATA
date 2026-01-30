#!/usr/bin/env python3
"""
COCO Dataset Generator using VLM + GroundingDINO
==================================================
Two-stage detection pipeline:
1. VLM identifies object categories
2. GroundingDINO generates precise bounding boxes

Author: Generated based on VLM and GroundingDINO integration
Date: 2026-01-29
"""

import json
import logging
import shutil
import subprocess
import socket
import time
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import cv2
import imageio
import numpy as np
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from openai import OpenAI
import httpx

# GroundingDINO imports
sys.path.append(str(Path(__file__).parent / "GroundingDINO"))
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


import argparse

@dataclass
class COCOConfig:
    """Configuration for COCO dataset generation with GroundingDINO."""
    
    # Input/Output paths
    input_path: Path = Path("input_data")
    output_dir: Path = Path("coco_dataset")
    
    # Frame extraction
    frame_interval: int = 10
    
    # VLM configuration (for object category detection)
    vlm_model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    vlm_uds_path: str = "/tmp/vllm-server.sock"
    vlm_api_base_url: str = "http://localhost/v1"
    vlm_api_key: str = "EMPTY"
    
    # GroundingDINO configuration
    gdino_config_path: Path = Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    gdino_checkpoint_path: Path = Path("GroundingDINO/weights/groundingdino_swint_ogc.pth")
    gdino_box_threshold: float = 0.35
    gdino_text_threshold: float = 0.25
    gdino_device: str = "cuda"  # or "cpu"
    
    # Image server for VLM
    image_server_port: int = 8082
    
    # Processing configuration
    num_workers: int = 4  # For frame extraction
    vlm_workers: int = 8   # For VLM inference
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Visualization
    visualize: bool = True
    vis_output_dir: Path = field(init=False)
    
    # COCO dataset structure
    images_dir: Path = field(init=False)
    annotations_file: Path = field(init=False)
    
    # Filtering settings
    excluded_categories: List[str] = field(default_factory=list)
    
    # Split settings
    split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)  # Train, Val, Test
    
    # Workflow settings
    review_mode: bool = False
    use_global_prompt: bool = False
    resume_from_json: Optional[Path] = None
    
    # Supported formats
    video_extensions: frozenset = field(
        default_factory=lambda: frozenset({'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'})
    )
    image_extensions: frozenset = field(
        default_factory=lambda: frozenset({'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'})
    )
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.output_dir = Path(self.output_dir)
        self.images_dir = self.output_dir / "images"
        self.annotations_file = self.output_dir / "annotations.json"
        self.vis_output_dir = self.output_dir / "visualizations"
        
        self.input_path = Path(self.input_path)
        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {self.input_path}")
        
        # Validate GroundingDINO paths
        self.gdino_config_path = Path(self.gdino_config_path)
        self.gdino_checkpoint_path = Path(self.gdino_checkpoint_path)
        
        if not self.gdino_config_path.exists():
            logger.warning(f"GroundingDINO config not found: {self.gdino_config_path}")
        if not self.gdino_checkpoint_path.exists():
            logger.warning(f"GroundingDINO checkpoint not found: {self.gdino_checkpoint_path}")


class LocalImageServer:
    """Context manager for a local HTTP image server."""
    
    def __init__(self, root_dir: Path, port: int):
        self.root_dir = Path(root_dir)
        self.port = port
        self.process: Optional[subprocess.Popen] = None
    
    def __enter__(self):
        """Start the image server."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', self.port)) == 0:
                logger.warning(f"Port {self.port} already in use. Assuming it's compatible.")
                return self
        
        logger.info(f"Starting image server on port {self.port}...")
        self.process = subprocess.Popen(
            ["python", "-m", "http.server", str(self.port), "--directory", str(self.root_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the image server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Image server stopped.")


class FrameExtractor:
    """Extract frames from videos or copy images."""
    
    def __init__(self, config: COCOConfig):
        self.config = config
    
    def find_media_files(self) -> Tuple[List[Path], List[Path]]:
        """Find all video and image files."""
        videos = []
        images = []
        
        for file_path in self.config.input_path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix in self.config.video_extensions:
                    videos.append(file_path)
                elif file_path.suffix in self.config.image_extensions:
                    images.append(file_path)
        
        return sorted(videos), sorted(images)
    
    def extract_frames_from_video(self, video_path: Path, start_id: int) -> List[Tuple[int, Path]]:
        """Extract frames from a single video."""
        results = []
        
        try:
            reader = imageio.get_reader(video_path)
            total_frames = reader.count_frames()
            
            frame_id = start_id
            for frame_idx in range(0, total_frames, self.config.frame_interval):
                try:
                    frame = reader.get_data(frame_idx)
                    save_path = self.config.images_dir / f"image_{frame_id:08d}.jpg"
                    Image.fromarray(frame).save(save_path, quality=95)
                    results.append((frame_id, save_path))
                    frame_id += 1
                except (IndexError, ValueError):
                    break
            
            reader.close()
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
        
        return results
    
    def process_image(self, image_path: Path, image_id: int) -> Optional[Tuple[int, Path]]:
        """Copy and convert image to JPG."""
        try:
            save_path = self.config.images_dir / f"image_{image_id:08d}.jpg"
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(save_path, quality=95)
            return (image_id, save_path)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def extract_all_frames(self) -> List[Tuple[int, Path]]:
        """Extract frames from all videos and process all images."""
        self.config.images_dir.mkdir(parents=True, exist_ok=True)
        
        videos, images = self.find_media_files()
        logger.info(f"Found {len(videos)} videos and {len(images)} images")
        
        all_results = []
        current_id = 0
        
        if videos:
            logger.info("Extracting frames from videos...")
            for video in tqdm(videos, desc="Processing videos"):
                results = self.extract_frames_from_video(video, current_id)
                all_results.extend(results)
                current_id = max([r[0] for r in results], default=current_id) + 1
        
        if images:
            logger.info("Processing images...")
            for img_path in tqdm(images, desc="Processing images"):
                result = self.process_image(img_path, current_id)
                if result:
                    all_results.append(result)
                    current_id += 1
        
        logger.info(f"Total frames/images extracted: {len(all_results)}")
        return all_results


class VLMCategoryDetector:
    """Use VLM to detect object categories (not bounding boxes)."""
    
    def __init__(self, config: COCOConfig):
        self.config = config
        self.prompt = self._create_category_prompt()
    
    def _create_category_prompt(self) -> str:
        """Create prompt for category detection only."""
        return """Analyze this image and list ALL objects you can see.

Output ONLY a JSON array of object names in this exact format:
["person", "car", "dog", "cat", "chair", "book", "cup"]

Requirements:
- List ALL visible objects
- Use common, specific object class names
- Be as detailed as possible
- Output ONLY the JSON array, no other text
- Use lowercase names
- Separate similar objects (e.g., "orange", "orange cup", "pink plate")"""
    
    def detect_categories(self, image_id: int, image_path: Path) -> Dict[str, Any]:
        """Detect object categories using VLM."""
        try:
            rel_path = image_path.relative_to(self.config.images_dir.parent)
            image_url = f"http://localhost:{self.config.image_server_port}/{rel_path}"
        except ValueError:
            logger.error(f"Image path {image_path} is not within expected directory")
            return {"image_id": image_id, "categories": []}
        
        transport = httpx.HTTPTransport(uds=self.config.vlm_uds_path)
        client = OpenAI(
            api_key=self.config.vlm_api_key,
            base_url=self.config.vlm_api_base_url,
            http_client=httpx.Client(transport=transport),
            max_retries=0
        )
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.config.vlm_model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=256,
                    temperature=0.0
                )
                
                content = response.choices[0].message.content.strip()
                categories = self._parse_categories(content)
                
                return {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "categories": categories
                }
                
            except Exception as e:
                if attempt < self.config.max_retries:
                    sleep_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed for image {image_id}, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to process image {image_id}: {e}")
        
        return {"image_id": image_id, "image_path": str(image_path), "categories": []}
    
    def _parse_categories(self, content: str) -> List[str]:
        """Parse category list from VLM output."""
        try:
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                categories = json.loads(json_str)
                
                if isinstance(categories, list):
                    # Filter and clean categories
                    filtered = []
                    for cat in categories:
                        if not cat:
                            continue
                        cat_clean = str(cat).strip().lower()
                        
                        # Filter excluded categories
                        is_excluded = False
                        for excluded in self.config.excluded_categories:
                            if excluded in cat_clean:  # Partial match e.g. "wooden table" vs "table"
                                is_excluded = True
                                break
                        
                        if not is_excluded:
                            filtered.append(cat_clean)
                            
                    return filtered
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse categories: {e}")
        
        return []


class GroundingDINODetector:
    """Use GroundingDINO to detect bounding boxes based on text prompts."""
    
    def __init__(self, config: COCOConfig):
        self.config = config
        self.model = None
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def load_model(self):
        """Load GroundingDINO model."""
        if self.model is not None:
            return
        
        logger.info(f"Loading GroundingDINO model from {self.config.gdino_checkpoint_path}")
        
        args = SLConfig.fromfile(str(self.config.gdino_config_path))
        args.device = self.config.gdino_device
        
        model = build_model(args)
        checkpoint = torch.load(str(self.config.gdino_checkpoint_path), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        
        if self.config.gdino_device == "cuda":
            model = model.cuda()
        
        self.model = model
        logger.info("GroundingDINO model loaded successfully")
    
    def detect_boxes(self, image_path: Path, categories: List[str]) -> List[Dict[str, Any]]:
        """Detect bounding boxes for given categories."""
        if not categories:
            return []
        
        if self.model is None:
            self.load_model()
        
        try:
            # Load and transform image
            image_pil = Image.open(image_path).convert("RGB")
            image_tensor, _ = self.transform(image_pil, None)
            
            # Create text prompt from categories
            text_prompt = ", ".join(categories)
            if not text_prompt.endswith("."):
                text_prompt = text_prompt + "."
            text_prompt = text_prompt.lower()
            
            # Run inference
            device = self.config.gdino_device
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                outputs = self.model(image_tensor[None], captions=[text_prompt])
            
            logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
            boxes = outputs["pred_boxes"][0]  # (nq, 4)
            
            # Filter by threshold
            logits_filt = logits.cpu()
            boxes_filt = boxes.cpu()
            filt_mask = logits_filt.max(dim=1)[0] > self.config.gdino_box_threshold
            logits_filt = logits_filt[filt_mask]
            boxes_filt = boxes_filt[filt_mask]
            
            # Get predicted phrases
            tokenizer = self.model.tokenizer
            tokenized = tokenizer(text_prompt)
            
            detections = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(
                    logit > self.config.gdino_text_threshold, 
                    tokenized, 
                    tokenizer
                )
                
                # Box is in cxcywh format, normalized [0, 1]
                # Convert to xyxy format
                cx, cy, w, h = box.tolist()
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                # Clamp to [0, 1]
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                
                detections.append({
                    'class': pred_phrase.strip(),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': logit.max().item()
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting boxes for {image_path}: {e}")
            return []


class Visualizer:
    """Visualize detection results."""
    
    def __init__(self, config: COCOConfig):
        self.config = config
        if config.visualize:
            self.config.vis_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        self.class_colors: Dict[str, Tuple[int, int, int]] = {}
    
    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get consistent color for a class."""
        if class_name not in self.class_colors:
            color_idx = len(self.class_colors) % len(self.colors)
            self.class_colors[class_name] = self.colors[color_idx]
        return self.class_colors[class_name]
    
    def visualize_detections(self, image_path: Path, detections: List[Dict[str, Any]], image_id: int):
        """Draw bounding boxes on image."""
        if not self.config.visualize:
            return
        
        try:
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            for det in detections:
                class_name = det['class']
                bbox = det['bbox']
                
                # Convert normalized to pixel coordinates
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                color = self._get_color(class_name)
                
                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label = f"{class_name}"
                if 'confidence' in det:
                    label += f" {det['confidence']:.2f}"
                
                bbox_text = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle(bbox_text, fill=color)
                draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
            
            vis_path = self.config.vis_output_dir / f"vis_{image_id:08d}.jpg"
            img.save(vis_path, quality=95)
            
        except Exception as e:
            logger.error(f"Error visualizing image {image_id}: {e}")


class COCODatasetBuilder:
    """Build COCO format dataset."""
    
    def __init__(self, config: COCOConfig):
        self.config = config
        self.category_name_to_id: Dict[str, int] = {}
        self.next_category_id = 1
        self.next_annotation_id = 1
    
    def build_dataset(self, detection_results: List[Dict[str, Any]], split_name: str = "train") -> Dict[str, Any]:
        """Build COCO format dataset for a specific split."""
        logger.info(f"Building COCO dataset for split: {split_name}...")
        
        coco_dataset = {
            "info": {
                "description": f"COCO Dataset ({split_name}) generated by VLM + GroundingDINO",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        for result in tqdm(detection_results, desc=f"Building {split_name} annotations"):
            image_id = result['image_id']
            image_path = Path(result['image_path'])
            detections = result['detections']
            
            try:
                img = Image.open(image_path)
                width, height = img.size
                
                coco_dataset['images'].append({
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height
                })
                
                for det in detections:
                    category_id = self._get_or_create_category(det['class'])
                    
                    # Convert normalized xyxy to COCO format (x, y, w, h) in pixels
                    x1, y1, x2, y2 = det['bbox']
                    x1_px = x1 * width
                    y1_px = y1 * height
                    x2_px = x2 * width
                    y2_px = y2 * height
                    
                    bbox_width = x2_px - x1_px
                    bbox_height = y2_px - y1_px
                    area = bbox_width * bbox_height
                    
                    annotation = {
                        "id": self.next_annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1_px, y1_px, bbox_width, bbox_height],
                        "area": area,
                        "iscrowd": 0
                    }
                    
                    if 'confidence' in det:
                        annotation['score'] = det['confidence']
                    
                    coco_dataset['annotations'].append(annotation)
                    self.next_annotation_id += 1
                    
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}")
        
        # Add categories
        coco_dataset['categories'] = [
            {"id": cat_id, "name": cat_name, "supercategory": "object"}
            for cat_name, cat_id in sorted(self.category_name_to_id.items(), key=lambda x: x[1])
        ]
        
        logger.info(f"Created dataset with {len(coco_dataset['images'])} images, "
                   f"{len(coco_dataset['annotations'])} annotations, "
                   f"{len(coco_dataset['categories'])} categories")
        
        return coco_dataset
    
    def _get_or_create_category(self, category_name: str) -> int:
        """Get or create category ID."""
        if category_name not in self.category_name_to_id:
            self.category_name_to_id[category_name] = self.next_category_id
            self.next_category_id += 1
        return self.category_name_to_id[category_name]
    
    def save_dataset(self, coco_dataset: Dict[str, Any], split_name: str = "train"):
        """Save COCO dataset."""
        # Save annotations
        ann_dir = self.config.output_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_file = ann_dir / f"{split_name}.json"
        
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(coco_dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {split_name} annotations to {ann_file}")

    def generate_yolo_labels(self, detection_results: List[Dict[str, Any]], split_name: str):
        """Generate YOLO format labels (.txt files)."""
        labels_dir = self.config.output_dir / "labels" / split_name
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for result in detection_results:
            image_path = Path(result['image_path'])
            txt_path = labels_dir / f"{image_path.stem}.txt"
            
            with open(txt_path, 'w') as f:
                for det in result['detections']:
                    class_name = det['class']
                    # Ensure category ID exists and is consistent
                    cat_id = self._get_or_create_category(class_name)
                    yolo_class_id = cat_id - 1  # 0-indexed for YOLO
                    
                    # bbox is normalized [x1, y1, x2, y2]
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Convert to cx, cy, w, h
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Clamp to [0,1] just in case
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    
                    f.write(f"{yolo_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        logger.info(f"Generated YOLO labels for {split_name} in {labels_dir}")

    def create_yolo_config(self, categories: List[Dict[str, Any]]):
        """Create YAML config for YOLO."""
        yaml_path = self.config.output_dir / "dataset.yaml"
        
        # Ensure categories are sorted
        sorted_cats = sorted(categories, key=lambda x: x['id'])
        class_names = [cat['name'] for cat in sorted_cats]
        
        # Standard directory structure for YOLO with COCO json
        # Assuming we have:
        # dataset/
        #   images/
        #     train/
        #     val/
        #     test/
        #   annotations/
        #     train.json
        #     val.json
        #     test.json
        
        yaml_content = f"""# COCO Dataset Configuration for YOLO
# Auto-generated on {datetime.now().isoformat()}

path: {self.config.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
"""
        for idx, name in enumerate(class_names):
            yaml_content += f"  {idx}: {name}\n"
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"Created YOLO config: {yaml_path}")
        return yaml_path


class COCODatasetGenerator:
    """Main orchestrator for COCO dataset generation."""
    
    def __init__(self, config: COCOConfig):
        self.config = config
        self.frame_extractor = FrameExtractor(config)
        self.vlm_detector = VLMCategoryDetector(config)
        self.gdino_detector = GroundingDINODetector(config)
        self.visualizer = Visualizer(config)
        self.dataset_builder = COCODatasetBuilder(config)
    
    def generate(self):
        """Main pipeline."""
        logger.info("=" * 80)
        logger.info("COCO Dataset Generation Pipeline (VLM + GroundingDINO)")
        logger.info("=" * 80)
        logger.info(f"Input: {self.config.input_path}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info(f"Frame interval: {self.config.frame_interval}")
        if self.config.use_global_prompt:
            logger.info("Mode: GLOBAL PROMPT (Single prompt for all images)")
        else:
            logger.info("Mode: PER-IMAGE (VLM runs on every image)")
        logger.info("=" * 80)
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        category_results = []
        
        # Check if resuming
        if self.config.resume_from_json and self.config.resume_from_json.exists():
            logger.info(f"\n[Resuming] Loading predictions from {self.config.resume_from_json}")
            with open(self.config.resume_from_json, 'r') as f:
                loaded_data = json.load(f)
            
            # Handle list of strings (Global Prompt Resume)
            if isinstance(loaded_data, list) and len(loaded_data) > 0 and isinstance(loaded_data[0], str):
                logger.info(f"Detected Global Prompt list: {loaded_data}")
                global_categories = loaded_data
                
                # Must scan images to apply global prompt
                logger.info("Scanning for images to apply global prompt...")
                image_list = self.frame_extractor.extract_all_frames()
                if not image_list:
                    logger.error("No images found to process. Exiting.")
                    return
                
                category_results = [
                    {
                        "image_id": img_id,
                        "image_path": str(img_path),
                        "categories": global_categories
                    }
                    for img_id, img_path in image_list
                ]
            else:
                # Normal per-image results
                category_results = loaded_data
                
            logger.info(f"Loaded/Generated tasks for {len(category_results)} images")
        
        else:
            # Step 1: Extract frames
            logger.info("\n[Step 1/5] Extracting frames...")
            image_list = self.frame_extractor.extract_all_frames()
            
            if not image_list:
                logger.error("No frames extracted. Exiting.")
                return
            
            # Step 2: VLM category detection
            logger.info(f"\n[Step 2/5] Detecting object categories with VLM...")
            
            if self.config.use_global_prompt:
                # GLOBAL MODE: Run on first image only
                logger.info("Running VLM on SAMPLE image only (Global Mode)...")
                sample_id, sample_path = image_list[0]
                
                with LocalImageServer(self.config.images_dir.parent, self.config.image_server_port):
                    result = self.vlm_detector.detect_categories(sample_id, sample_path)
                
                global_categories = result['categories']
                logger.info(f"Proposed Global Categories: {global_categories}")
                
                # Save just the list
                global_json_path = self.config.output_dir / "global_categories.json"
                with open(global_json_path, 'w') as f:
                    json.dump(global_categories, f, indent=2)
                
                if self.config.review_mode:
                    logger.info("\n" + "=" * 80)
                    logger.info("⏸️  PAUSED FOR GLOBAL REVIEW")
                    logger.info("=" * 80)
                    logger.info(f"Global Prompt File: {global_json_path}")
                    logger.info("Please edit this file to define the EXACT categories for the entire dataset.")
                    logger.info("Resume with:")
                    logger.info(f"  python generate_coco_with_gdino.py --global --resume {global_json_path.name}")
                    return
                else:
                    # Apply immediately
                    category_results = [
                        {
                            "image_id": img_id,
                            "image_path": str(img_path),
                            "categories": global_categories
                        }
                        for img_id, img_path in image_list
                    ]

            else:
                # PER-IMAGE MODE
                with LocalImageServer(self.config.images_dir.parent, self.config.image_server_port):
                    with ProcessPoolExecutor(max_workers=self.config.vlm_workers) as executor:
                        futures = [
                            executor.submit(self.vlm_detector.detect_categories, img_id, img_path)
                            for img_id, img_path in image_list
                        ]
                        
                        for future in tqdm(as_completed(futures), total=len(futures), desc="VLM detection"):
                            result = future.result()
                            if result['categories']:
                                category_results.append(result)
                
                logger.info(f"Detected categories in {len(category_results)} images")
                
                # Save intermediate results
                intermediate_path = self.config.output_dir / "vlm_predictions.json"
                with open(intermediate_path, 'w') as f:
                    json.dump(category_results, f, indent=2)
                logger.info(f"Saved intermediate VLM predictions to {intermediate_path}")
                
                # Pause if in review mode
                if self.config.review_mode:
                    logger.info("\n" + "=" * 80)
                    logger.info("⏸️  PAUSED FOR HUMAN REVIEW")
                    logger.info("=" * 80)
                    logger.info(f"Please edit the file below to improve dataset quality:")
                    logger.info(f"  {intermediate_path}")
                    logger.info("\nRemove unwanted objects, fix typos, or add missing categories.")
                    logger.info("\nOnce finished, resume generation with:")
                    logger.info(f"  python generate_coco_with_gdino.py --resume {intermediate_path.name}")
                    return

        if not category_results:
            logger.warning("No categories detected. Exiting.")
            return
        
        # Step 3: GroundingDINO bbox detection
        logger.info(f"\n[Step 3/5] Detecting bounding boxes with GroundingDINO...")
        self.gdino_detector.load_model()
        
        detection_results = []
        for result in tqdm(category_results, desc="GroundingDINO detection"):
            image_path = Path(result['image_path'])
            categories = result['categories']
            
            detections = self.gdino_detector.detect_boxes(image_path, categories)
            
            if detections:
                detection_results.append({
                    'image_id': result['image_id'],
                    'image_path': str(image_path),
                    'detections': detections
                })
        
        logger.info(f"Generated bounding boxes for {len(detection_results)} images")
        
        # Step 4: Visualize
        if self.config.visualize:
            logger.info(f"\n[Step 4/5] Generating visualizations...")
            for result in tqdm(detection_results, desc="Creating visualizations"):
                self.visualizer.visualize_detections(
                    Path(result['image_path']),
                    result['detections'],
                    result['image_id']
                )
        else:
            logger.info("\n[Step 4/5] Skipping visualization (disabled)")
        
        # Step 5: Split and Build COCO dataset
        logger.info("\n[Step 5/5] Splitting and Building COCO dataset...")
        
        # Shuffle results
        random.seed(42)
        random.shuffle(detection_results)
        
        total_images = len(detection_results)
        train_ratio, val_ratio, test_ratio = self.config.split_ratio
        
        n_train = int(total_images * train_ratio)
        n_val = int(total_images * val_ratio)
        
        train_set = detection_results[:n_train]
        val_set = detection_results[n_train:n_train+n_val]
        test_set = detection_results[n_train+n_val:]
        
        splits = {
            "train": train_set,
            "val": val_set,
            "test": test_set
        }
        
        # Prepare directories
        for split_name in splits:
            (self.config.images_dir / split_name).mkdir(parents=True, exist_ok=True)
        
        all_categories = []
        
        for split_name, results in splits.items():
            if not results:
                continue
                
            # Move images
            logger.info(f"Moving {len(results)} images to {split_name}...")
            for result in results:
                src = Path(result['image_path'])
                dst = self.config.images_dir / split_name / src.name
                shutil.move(src, dst)
                result['image_path'] = str(dst)  # Update path
            
            # Build and save dataset (COCO JSON)
            coco_ds = self.dataset_builder.build_dataset(results, split_name)
            self.dataset_builder.save_dataset(coco_ds, split_name)
            
            # Generate YOLO TXT labels
            self.dataset_builder.generate_yolo_labels(results, split_name)
            
            # Collect categories for YAML
            if not all_categories and coco_ds['categories']:
                all_categories = coco_ds['categories']
        
        yaml_path = self.dataset_builder.create_yolo_config(all_categories)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("Dataset Generation Complete!")
        logger.info("=" * 80)
        logger.info(f"Total Images: {total_images}")
        logger.info(f"  - Train: {len(train_set)}")
        logger.info(f"  - Val:   {len(val_set)}")
        logger.info(f"  - Test:  {len(test_set)}")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  - Images: {self.config.images_dir}/{{train,val,test}}")
        logger.info(f"  - Annotations: {self.config.output_dir}/annotations/")
        logger.info(f"  - YOLO config: {yaml_path}")
        if self.config.visualize:
            logger.info(f"  - Visualizations: {self.config.vis_output_dir}")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate COCO dataset with VLM + GroundingDINO")
    parser.add_argument("--input", type=Path, default=Path("input_data"), help="Input directory")
    parser.add_argument("--output", type=Path, default=Path("coco_dataset"), help="Output directory")
    parser.add_argument("--review", action="store_true", help="Enable human review mode")
    parser.add_argument("--global", dest="use_global", action="store_true", help="Use a single GLOBAL prompt for all images")
    parser.add_argument("--resume", type=Path, help="Resume from JSON file")
    parser.add_argument("--exclude", nargs="+", help="Additional categories to exclude")
    
    args = parser.parse_args()
    
    # Base excluded categories (Empty by default)
    excluded = []
    if args.exclude:
        excluded.extend(args.exclude)
    
    config = COCOConfig(
        input_path=args.input,
        output_dir=args.output,
        frame_interval=15,
        num_workers=4,
        vlm_workers=8,
        visualize=True,
        excluded_categories=excluded,
        review_mode=args.review,
        use_global_prompt=args.use_global,
        resume_from_json=args.output / args.resume if args.resume else None,
        
        # GroundingDINO settings
        gdino_config_path=Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        gdino_checkpoint_path=Path("GroundingDINO/weights/groundingdino_swint_ogc.pth"),
        gdino_box_threshold=0.35,
        gdino_text_threshold=0.25,
        gdino_device="cuda",  # or "cpu"
    )
    
    generator = COCODatasetGenerator(config)
    generator.generate()


if __name__ == "__main__":
    main()
