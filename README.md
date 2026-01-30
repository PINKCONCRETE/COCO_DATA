# COCO DATA

## 🎯 System Architecture

Combine the strengths of two models:
- **VLM (Qwen-VL)**: Identify object categories in images (Specialized in Recognition)
- **GroundingDINO**: Generate precise bounding boxes based on categories (Specialized in Localization)

## 🔄 Workflow

```
Video/Image
    ↓
[FrameExtractor]
Extract JPG image sequence
    ↓
[VLMCategoryDetector]
VLM identifies object categories
Output: ["person", "car", "dog", ...]
    ↓
[GroundingDINODetector]
Generate precise bbox based on categories
Output: [{class: "person", bbox: [x1,y1,x2,y2]}, ...]
    ↓
[Visualizer]
Visualize results
    ↓
[COCODatasetBuilder]
Generate COCO dataset
    ↓
YOLO Model Training
```

## 🚀 Quick Start

### 1. Environment Preparation

#### 1.1 Clone Code

```
git clone -r https://github.com/PINKCONCRETE/COCO_DATA
```

#### 1.2 Configure Base Environment

```bash
conda create -n coco_data python=3.11

conda activate coco_data
# Install base dependencies
pip install -r requirements.txt
```

```bash
# Install GroundingDINO dependencies
cd GroundingDINO
pip install -r requirements.txt
pip install -e . # If issues try: python -m pip install --no-build-isolation -e .
```

### 2. Download GroundingDINO Model

```bash
cd GroundingDINO
mkdir -p weights

# Download model weights
wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 3. Configure and Start VLM Server

```bash
conda create -n vllm python=3.11
pip install vllm
bash start_vllm_server.sh # Note: Select appropriate VLM for your machine
```

### 4. Prepare Data

```bash
mkdir -p input_data
cp your_videos/*.mp4 input_data/
cp your_images/*.jpg input_data/
```

### 5. Run Generation

#### 5.1 Fully Automatic Mode (Default)
Run directly, the program will automatically complete all steps:
```bash
python generate_coco_with_gdino.py
```

#### 5.2 Global Review Mode (Recommended)
Scans only the first frame of each video/image to generate a global object list. After editing and confirming, apply to all frames. Suitable for data with fixed scenes.
```bash
# 1. Scan to generate global list
python generate_coco_with_gdino.py --review --global

# 2. Program pauses, manually edit coco_dataset/global_categories.json

# 3. Resume generation
python generate_coco_with_gdino.py --global --resume global_categories.json
```

#### 5.3 Per-Image Review Mode
VLM identifies every image to generate a complete result list. You can check and correct recognition results one by one. Suitable for scenarios with large changes where every image has different objects.
```bash
# 1. Run and generate initial results
python generate_coco_with_gdino.py --review

# 2. Program pauses, manually edit coco_dataset/vlm_predictions.json
#    (File contains image_id and corresponding categories for each image)

# 3. Resume generation (Directly use your corrected results to run GroundingDINO)
python generate_coco_with_gdino.py --resume vlm_predictions.json
```

### 6. Train Model

After data generation is complete, directly run the training script:

```bash
python train_coco.py
```
- Defaults to `yolo26x.pt` (High accuracy). For speedup, modify code to use `yolo26n.pt`.
- Training results (weights, logs) are saved in `coco_data/runs/train/coco_finetune/` directory.

### 7. Model Inference (Prediction)

Use the trained model to detect new images or videos:

```bash
# Image Inference
python predict_coco.py --model model.pt --source image.jpg

# Video Inference
python predict_coco.py --model model.pt --source video.mp4 --output results/
```
Parameters:
- `--model`: Path to trained model (`.pt`)
- `--source`: Input image or video path
- `--conf`: Confidence threshold (default 0.25)
- `--show`: Show results in real-time (GUI required)
- `--save-txt`: Save detection box coordinates to txt file

## ⚙️ Configuration

### Core Config

```python
config = COCOConfig(
    # Input/Output
    input_path=Path("input_data"),
    output_dir=Path("coco_dataset"),
    frame_interval=30,  # Video frame sampling interval
    
    # VLM Config
    vlm_model_id="Qwen/Qwen3-VL-8B-Instruct",
    vlm_workers=8,  # VLM parallelism
    
    # GroundingDINO Config
    gdino_config_path=Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    gdino_checkpoint_path=Path("GroundingDINO/weights/groundingdino_swint_ogc.pth"),
    gdino_box_threshold=0.35,      # Bounding Box threshold
    gdino_text_threshold=0.25,     # Text threshold
    gdino_device="cuda",           # GPU Device
    
    # Others
    visualize=True,  # Generate visualization
)
```

### Threshold Adjustment

- **gdino_box_threshold** (0.35)
  - Higher: Stricter detection, fewer false positives
  - Lower: Looser detection, possible false positives
  
- **gdino_text_threshold** (0.25)
  - Controls strictness of text matching
  - Higher: Keep only high confidence detections

## 🏗️ Core Classes

### VLMCategoryDetector
- **Function**: Identify object categories using VLM
- **Input**: Image
- **Output**: Category list `["person", "car", ...]`
- **Prompt**: Specifically designed for category identification

```python
# Output Example
{
    "image_id": 0,
    "image_path": "images/image_00000000.jpg",
    "categories": ["person", "orange", "book", "pink plate", "orange cup"]
}
```

### GroundingDINODetector
- **Function**: Generate precise bounding boxes based on categories
- **Input**: Image + Category List
- **Output**: Detection results (Class + bbox)

```python
# Output Example
{
    "image_id": 0,
    "image_path": "images/image_00000000.jpg",
    "detections": [
        {
            "class": "person",
            "bbox": [0.1, 0.2, 0.5, 0.8],  # xyxy normalized
            "confidence": 0.95
        },
        {
            "class": "orange",
            "bbox": [0.6, 0.3, 0.8, 0.5],
            "confidence": 0.87
        }
    ]
}
```

## 📊 Output Format

### Generated Dataset Structure
The program will automatically split data into train/val/test (7:2:1) and generate both COCO and YOLO format annotations.

```
coco_dataset/
├── dataset.yaml         # YOLO training config
├── images/
│   ├── train/           # Train images
│   ├── val/             # Validation images
│   └── test/            # Test images
├── labels/              # YOLO format labels (.txt)
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/         # COCO format labels (.json)
│   ├── train.json
│   ├── val.json
│   └── test.json
└── visualizations/      # Visualization results
```

### Annotation File Examples

#### COCO Format (annotations/train.json)
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [100, 200, 300, 400],  # [x, y, w, h]
      "area": 120000,
      "iscrowd": 0
    }
  ],
  "categories": [...]
}
```

#### YOLO Format (labels/train/image_0.txt)
```text
# <class_id> <x_center> <y_center> <width> <height> (Normalized)
0 0.532 0.485 0.15 0.35
1 0.221 0.334 0.12 0.22
```

## 🎨 Two-Stage Detection Example

### Stage 1: VLM Category Recognition

**Input Image**: test1.jpg

**VLM Output**:
```json
["person", "orange", "book", "pink plate", "bamboo basket", "orange cup"]
```

### Stage 2: GroundingDINO Bounding Box Detection

**Input**: test1.jpg + "person, orange, book, pink plate, bamboo basket, orange cup"

**GroundingDINO Output**:
```json
[
  {"class": "person", "bbox": [0.2, 0.1, 0.8, 0.9], "confidence": 0.95},
  {"class": "orange", "bbox": [0.3, 0.5, 0.4, 0.6], "confidence": 0.88},
  {"class": "book", "bbox": [0.5, 0.4, 0.7, 0.5], "confidence": 0.82},
  {"class": "pink plate", "bbox": [0.1, 0.6, 0.3, 0.8], "confidence": 0.79},
  {"class": "bamboo basket", "bbox": [0.7, 0.3, 0.9, 0.6], "confidence": 0.85},
  {"class": "orange cup", "bbox": [0.4, 0.7, 0.5, 0.85], "confidence": 0.76}
]
```

## 🔧 Performance Optimization

### GPU Memory Optimization

```python
config = COCOConfig(
    gdino_device="cuda",
    vlm_workers=4,  # Reduce VLM workers
    num_workers=2,  # Reduce frame extraction workers
)
```

### Speed Optimization

```python
config = COCOConfig(
    frame_interval=60,  # Increase sample interval
    visualize=False,    # Disable visualization
    vlm_workers=16,     # Increase VLM workers
)
```

### Quality Optimization

```python
config = COCOConfig(
    gdino_box_threshold=0.4,   # Increase threshold
    gdino_text_threshold=0.3,  # Increase threshold
)
```

## 🐛 Common Issues

### 1. GroundingDINO Model Load Failure

**Issue**: Config or weights file not found

**Solution**:
```bash
# Check if files exist
ls coco_data/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
ls coco_data/GroundingDINO/weights/groundingdino_swint_ogc.pth

# If not, re-download
cd coco_data/GroundingDINO
git pull
mkdir -p weights
wget -P weights <download_link>
```

### 2. CUDA Out of Memory

**Solution**:
1. Reduce VLM worker count
2. Use CPU for GroundingDINO: `gdino_device="cpu"`
3. Batch Processing: Process fewer images at a time

### 3. Empty Detection Results

**Possible Causes**:
- VLM failed to identify categories
- GroundingDINO threshold too high
- Image quality issues

**Solution**:
1. Check VLM output: Check categories in logs
2. Lower threshold: `gdino_box_threshold=0.25`
3. Check if images are clear

### 4. Category Name Mismatch

**Issue**: VLM output categories not recognized by GroundingDINO

**Solution**: VLM prompt is optimized to output names GroundingDINO usually understands. If issues persist, add category mapping.

### 5. GroundingDINO Label Issues

**Solution**: Modify label names directly in `dataset.yaml` under `names` field. Refer to visualization results in `coco_dataset/visualizations/` correctly identify GroundingDINO outputs.

## 🎓 Advanced Usage

### Custom VLM Prompts

Modify in `VLMCategoryDetector._create_category_prompt()`:

```python
def _create_category_prompt(self) -> str:
    return """List all visible objects in this image in detail.

Requirements:
- Use English names
- Be as detailed and specific as possible
- Distinguish similar objects (e.g., red apple, green apple)

Output JSON array format:
["person", "red apple", "wooden table"]"""
```

### Add Post-Processing

```python
class GroundingDINODetector:
    def detect_boxes(self, image_path, categories):
        detections = self._raw_detect(image_path, categories)
        
        # Non-Maximum Suppression
        detections = self._apply_nms(detections, iou_threshold=0.5)
        
        # Filter small objects
        detections = [d for d in detections if self._box_area(d['bbox']) > 0.01]
        
        return detections
```

## 📚 References

- [GroundingDINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [Qwen-VL Documentation](https://github.com/QwenLM/Qwen-VL)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [YOLO Training Guide](https://docs.ultralytics.com)

## ✅ Checklist

Confirm before use:
- [ ] VLM server started
- [ ] GroundingDINO model downloaded
- [ ] Input data prepared
- [ ] GPU/CPU settings correct
- [ ] Dependencies installed

Start Generation:
```bash
python generate_coco_with_gdino.py
```

## Roadmap

1. Add UI
2. Add support for SAM3
