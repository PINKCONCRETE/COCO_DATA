#!/usr/bin/env python3
"""
Validation and Testing Script for Generated COCO Dataset
=========================================================
Validate COCO dataset structure and visualize sample annotations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class COCOValidator:
    """Validate COCO dataset format and content."""
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.annotations_file = self.dataset_dir / "annotations.json"
        self.images_dir = self.dataset_dir / "images"
        self.dataset: Dict[str, Any] = {}
    
    def load_dataset(self) -> bool:
        """Load COCO annotations."""
        if not self.annotations_file.exists():
            logger.error(f"Annotations file not found: {self.annotations_file}")
            return False
        
        try:
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            logger.info("Successfully loaded COCO dataset")
            return True
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate COCO dataset structure."""
        logger.info("Validating dataset structure...")
        
        required_keys = ['images', 'annotations', 'categories', 'info']
        for key in required_keys:
            if key not in self.dataset:
                logger.error(f"Missing required key: {key}")
                return False
        
        logger.info("✓ All required keys present")
        return True
    
    def validate_content(self) -> bool:
        """Validate dataset content."""
        logger.info("Validating dataset content...")
        
        # Check images
        images = self.dataset.get('images', [])
        if not images:
            logger.warning("No images in dataset")
            return False
        logger.info(f"✓ Found {len(images)} images")
        
        # Check annotations
        annotations = self.dataset.get('annotations', [])
        if not annotations:
            logger.warning("No annotations in dataset")
        else:
            logger.info(f"✓ Found {len(annotations)} annotations")
        
        # Check categories
        categories = self.dataset.get('categories', [])
        if not categories:
            logger.warning("No categories in dataset")
            return False
        logger.info(f"✓ Found {len(categories)} categories")
        
        # Validate image files exist
        missing_files = 0
        for img_info in images[:10]:  # Check first 10
            img_path = self.images_dir / img_info['file_name']
            if not img_path.exists():
                missing_files += 1
        
        if missing_files > 0:
            logger.warning(f"Found {missing_files} missing image files (checked first 10)")
        else:
            logger.info("✓ Image files exist")
        
        return True
    
    def print_statistics(self):
        """Print dataset statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("Dataset Statistics")
        logger.info("=" * 80)
        
        # Basic stats
        num_images = len(self.dataset.get('images', []))
        num_annotations = len(self.dataset.get('annotations', []))
        num_categories = len(self.dataset.get('categories', []))
        
        logger.info(f"Total Images: {num_images}")
        logger.info(f"Total Annotations: {num_annotations}")
        logger.info(f"Total Categories: {num_categories}")
        
        if num_images > 0:
            logger.info(f"Average Annotations per Image: {num_annotations / num_images:.2f}")
        
        # Category distribution
        logger.info("\nCategory Distribution:")
        category_counts = {}
        for ann in self.dataset.get('annotations', []):
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        # Get category names
        category_id_to_name = {
            cat['id']: cat['name']
            for cat in self.dataset.get('categories', [])
        }
        
        for cat_id, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            cat_name = category_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            logger.info(f"  {cat_name}: {count} annotations")
        
        # Image size distribution
        if self.dataset.get('images'):
            widths = [img['width'] for img in self.dataset['images']]
            heights = [img['height'] for img in self.dataset['images']]
            logger.info(f"\nImage Dimensions:")
            logger.info(f"  Width range: {min(widths)} - {max(widths)}")
            logger.info(f"  Height range: {min(heights)} - {max(heights)}")
        
        logger.info("=" * 80)
    
    def visualize_samples(self, num_samples: int = 5, save_dir: Path = None):
        """Visualize sample annotations."""
        if not self.dataset:
            logger.error("Dataset not loaded")
            return
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nVisualizing {num_samples} sample images...")
        
        images = self.dataset.get('images', [])
        annotations = self.dataset.get('annotations', [])
        categories = self.dataset.get('categories', [])
        
        # Create category lookup
        cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
        
        # Group annotations by image
        img_to_anns = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Visualize samples
        samples_visualized = 0
        for img_info in images[:num_samples * 2]:  # Try more in case some fail
            if samples_visualized >= num_samples:
                break
            
            img_id = img_info['id']
            img_path = self.images_dir / img_info['file_name']
            
            if not img_path.exists():
                continue
            
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                draw = ImageDraw.Draw(img)
                
                # Try to load font
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Draw annotations
                anns = img_to_anns.get(img_id, [])
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                         (255, 0, 255), (0, 255, 255)]
                
                for i, ann in enumerate(anns):
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    color = colors[i % len(colors)]
                    
                    # Draw box
                    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
                    
                    # Draw label
                    cat_name = cat_id_to_name.get(ann['category_id'], 'Unknown')
                    label = f"{cat_name}"
                    
                    bbox_text = draw.textbbox((x, y), label, font=font)
                    draw.rectangle(bbox_text, fill=color)
                    draw.text((x, y), label, fill=(255, 255, 255), font=font)
                
                # Save or show
                if save_dir:
                    save_path = save_dir / f"validation_{samples_visualized:03d}.jpg"
                    img.save(save_path, quality=95)
                    logger.info(f"Saved visualization: {save_path}")
                else:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Image {img_id}: {len(anns)} annotations")
                    plt.tight_layout()
                    plt.show()
                
                samples_visualized += 1
                
            except Exception as e:
                logger.error(f"Error visualizing image {img_id}: {e}")
        
        logger.info(f"Visualized {samples_visualized} images")
    
    def validate_all(self) -> bool:
        """Run all validations."""
        logger.info("\n" + "=" * 80)
        logger.info("COCO Dataset Validation")
        logger.info("=" * 80)
        logger.info(f"Dataset directory: {self.dataset_dir}")
        logger.info("=" * 80 + "\n")
        
        # Load dataset
        if not self.load_dataset():
            return False
        
        # Validate structure
        if not self.validate_structure():
            return False
        
        # Validate content
        if not self.validate_content():
            return False
        
        # Print statistics
        self.print_statistics()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Validation Complete - Dataset is valid!")
        logger.info("=" * 80)
        
        return True


def main():
    """Main validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate COCO dataset')
    parser.add_argument('dataset_dir', type=str, nargs='?', default='coco_dataset',
                       help='Path to COCO dataset directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample annotations')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--save-vis', type=str, default=None,
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create validator
    validator = COCOValidator(Path(args.dataset_dir))
    
    # Run validation
    if not validator.validate_all():
        logger.error("Validation failed!")
        sys.exit(1)
    
    # Visualize if requested
    if args.visualize:
        save_dir = Path(args.save_vis) if args.save_vis else None
        validator.visualize_samples(num_samples=args.num_samples, save_dir=save_dir)
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
