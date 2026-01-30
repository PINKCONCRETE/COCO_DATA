# COCO DATA

## 🎯 系统架构

结合两个模型的优势：
- **VLM (Qwen-VL)**: 识别图片中的物体类别（擅长识别）
- **GroundingDINO**: 基于类别生成精确边界框（擅长定位）

## 🔄 工作流程

```
视频/图片
    ↓
[FrameExtractor]
提取JPG图片序列
    ↓
[VLMCategoryDetector]
VLM识别物体类别
输出: ["person", "car", "dog", ...]
    ↓
[GroundingDINODetector]
基于类别生成精确bbox
输出: [{class: "person", bbox: [x1,y1,x2,y2]}, ...]
    ↓
[Visualizer]
可视化结果
    ↓
[COCODatasetBuilder]
生成COCO数据集
    ↓
YOLO模型训练
```

## 🚀 快速开始

### 1. 环境准备

#### 1.1 下载代码

```
git clone -r https://github.com/PINKCONCRETE/COCO_DATA
```

#### 1.2 配置基础环境

```bash
conda create -n coco_data python=3.11

conda activate coco_data
# 安装基础依赖
pip install -r requirements.txt
```



```bash

# 安装GroundingDINO依赖
cd GroundingDINO
pip install -r requirements.txt
pip install -e . # 若报错可尝试 python -m pip install --no-build-isolation -e .
```

### 2. 下载GroundingDINO模型

```bash
cd GroundingDINO
mkdir -p weights

# 下载模型权重
wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

```

### 3. 配置并启动VLM服务器

```bash
conda create -n vllm python=3.11
pip install vllm
bash start_vllm_server.sh # 注意，需要根据机器选择合适的VLM
```

### 4. 准备数据

```bash
mkdir -p input_data
cp your_videos/*.mp4 input_data/
cp your_images/*.jpg input_data/
```

### 5. 运行生成

#### 5.1 全自动模式 (默认)
直接运行，程序会自动完成所有步骤：
```bash
python generate_coco_with_gdino.py
```

#### 5.2 全局人工审核模式 (Global Review, 推荐)
只扫描每个视频/图片的第一帧，生成一个全局物体列表。编辑确认后，应用于所有帧。适合场景固定的数据。
```bash
# 1. 扫描生成全局列表
python generate_coco_with_gdino.py --review --global

# 2. 程序暂停，手动编辑 coco_dataset/global_categories.json

# 3. 恢复生成
python generate_coco_with_gdino.py --global --resume global_categories.json
```

#### 5.3 逐帧审核模式 (Per-Image Review)
VLM 识别每一张图片，生成完整的结果列表。你可以逐张检查和修正识别结果。适合场景变化大、每张图物体都不一样的情况。
```bash
# 1. 运行并生成初始结果
python generate_coco_with_gdino.py --review

# 2. 程序暂停，手动编辑 coco_dataset/vlm_predictions.json
#    (文件内包含了每张图片的 image_id 和对应的 categories)

# 3. 恢复生成 (直接利用你修正后的结果跑 GroundingDINO)
python generate_coco_with_gdino.py --resume vlm_predictions.json
```

### 6. 训练模型

数据生成完成后，直接运行训练脚本：

```bash
python train_coco.py
```
- 默认使用 `yolo26x.pt` (高精度)。如需加速，请在代码中修改为 `yolo26n.pt`。
- 训练结果（权重、日志）保存在 `coco_data/runs/train/coco_finetune/` 目录下。

### 7. 模型推理 (预测)

使用训练好的模型对新图片或视频进行检测：

```bash
# 图片推理
python predict_coco.py --model model.pt --source image.jpg

# 视频推理
python predict_coco.py --model model.pt --source video.mp4 --output results/
```
参数说明：
- `--model`: 训练好的模型路径 (`.pt`)
- `--source`: 输入图片或视频路径
- `--conf`: 置信度阈值 (默认0.25)
- `--show`: 实时显示结果 (需要GUI环境)
- `--save-txt`: 保存检测框坐标到txt文件

## ⚙️ 配置说明

### 核心配置

```python
config = COCOConfig(
    # 输入输出
    input_path=Path("input_data"),
    output_dir=Path("coco_dataset"),
    frame_interval=30,  # 视频采样间隔
    
    # VLM配置
    vlm_model_id="Qwen/Qwen3-VL-8B-Instruct",
    vlm_workers=8,  # VLM并行数
    
    # GroundingDINO配置
    gdino_config_path=Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    gdino_checkpoint_path=Path("GroundingDINO/weights/groundingdino_swint_ogc.pth"),
    gdino_box_threshold=0.35,      # 边界框阈值
    gdino_text_threshold=0.25,     # 文本阈值
    gdino_device="cuda",           # GPU设备
    
    # 其他
    visualize=True,  # 生成可视化
)
```

### 阈值调整

- **gdino_box_threshold** (0.35)
  - 越高：检测越严格，假阳性越少
  - 越低：检测越宽松，可能有假阳性
  
- **gdino_text_threshold** (0.25)
  - 控制文本匹配的严格程度
  - 越高：只保留高置信度的检测

## 🏗️ 核心类说明

### VLMCategoryDetector
- **功能**: 使用VLM识别物体类别
- **输入**: 图片
- **输出**: 类别列表 `["person", "car", ...]`
- **提示词**: 专门设计用于类别识别

```python
# 示例输出
{
    "image_id": 0,
    "image_path": "images/image_00000000.jpg",
    "categories": ["person", "orange", "book", "pink plate", "orange cup"]
}
```

### GroundingDINODetector
- **功能**: 基于类别生成精确边界框
- **输入**: 图片 + 类别列表
- **输出**: 检测结果（类别 + bbox）

```python
# 示例输出
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

## 📊 输出格式

### 生成的数据集结构
程序会自动将数据按 7:2:1 划分为训练集、验证集和测试集，并同时生成 COCO 格式和 YOLO 格式的标注。

```
coco_dataset/
├── dataset.yaml         # YOLO 训练配置
├── images/
│   ├── train/           # 训练集图片
│   ├── val/             # 验证集图片
│   └── test/            # 测试集图片
├── labels/              # YOLO格式标签 (.txt)
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/         # COCO格式标签 (.json)
│   ├── train.json
│   ├── val.json
│   └── test.json
└── visualizations/      # 可视化结果
```

### 标注文件示例

#### COCO 格式 (annotations/train.json)
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

#### YOLO 格式 (labels/train/image_0.txt)
```text
# <class_id> <x_center> <y_center> <width> <height> (归一化)
0 0.532 0.485 0.15 0.35
1 0.221 0.334 0.12 0.22
```

## 🎨 两阶段检测流程示例

### 阶段1: VLM类别识别

**输入图片**: test1.jpg

**VLM输出**:
```json
["person", "orange", "book", "pink plate", "bamboo basket", "orange cup"]
```

### 阶段2: GroundingDINO边界框检测

**输入**: test1.jpg + "person, orange, book, pink plate, bamboo basket, orange cup"

**GroundingDINO输出**:
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

## 🔧 性能优化

### GPU内存优化

```python
config = COCOConfig(
    gdino_device="cuda",
    vlm_workers=4,  # 减少VLM并行数
    num_workers=2,  # 减少帧提取并行数
)
```

### 速度优化

```python
config = COCOConfig(
    frame_interval=60,  # 增大采样间隔
    visualize=False,    # 关闭可视化
    vlm_workers=16,     # 增加VLM并行
)
```

### 质量优化

```python
config = COCOConfig(
    gdino_box_threshold=0.4,   # 提高阈值
    gdino_text_threshold=0.3,  # 提高阈值
)
```

## 🐛 常见问题

### 1. GroundingDINO模型加载失败

**问题**: 找不到配置文件或权重文件

**解决**:
```bash
# 检查文件是否存在
ls coco_data/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
ls coco_data/GroundingDINO/weights/groundingdino_swint_ogc.pth

# 如果不存在，重新下载
cd coco_data/GroundingDINO
git pull
mkdir -p weights
wget -P weights <下载链接>
```

### 2. CUDA内存不足

**解决方案**:
1. 减少VLM并行worker数
2. 使用CPU运行GroundingDINO: `gdino_device="cpu"`
3. 批量处理：一次处理少量图片

### 3. 检测结果为空

**可能原因**:
- VLM没有识别出类别
- GroundingDINO阈值过高
- 图片质量问题

**解决**:
1. 检查VLM输出: 查看日志中的categories
2. 降低阈值: `gdino_box_threshold=0.25`
3. 检查图片是否清晰

### 4. 类别名称不匹配

**问题**: VLM输出的类别GroundingDINO识别不了

**解决**: VLM的prompt已经优化为输出GroundingDINO能理解的类别名称。如果仍有问题，可以添加类别映射。

### 5. GroundingDINO 标签问题

**解决**：可直接修改数据集下的dataset.yaml中的names来修改标签名，可参考数据集的visualize/下找到GroudingDINO的识别结果进行修改。

## 🎓 高级用法

### 自定义VLM提示词

在 `VLMCategoryDetector._create_category_prompt()` 中修改：

```python
def _create_category_prompt(self) -> str:
    return """针对这张图片，详细列出所有可见的物体。

要求：
- 使用英文名称
- 尽可能详细和具体
- 区分相似物体（如：red apple, green apple）

输出JSON数组格式：
["person", "red apple", "wooden table"]"""
```

### 添加后处理

```python
class GroundingDINODetector:
    def detect_boxes(self, image_path, categories):
        detections = self._raw_detect(image_path, categories)
        
        # 非极大值抑制
        detections = self._apply_nms(detections, iou_threshold=0.5)
        
        # 过滤小目标
        detections = [d for d in detections if self._box_area(d['bbox']) > 0.01]
        
        return detections
```

## 📚 参考资料

- [GroundingDINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- [Qwen-VL文档](https://github.com/QwenLM/Qwen-VL)
- [COCO数据集格式](https://cocodataset.org/#format-data)
- [YOLO训练指南](https://docs.ultralytics.com)

## ✅ 检查清单

使用前确认：
- [ ] VLM服务器已启动
- [ ] GroundingDINO模型已下载
- [ ] 输入数据已准备
- [ ] GPU/CPU设置正确
- [ ] 依赖包已安装

开始生成：
```bash
python generate_coco_with_gdino.py
```

## 后续规划

1. 添加UI
2. 添加对SAM3的支持
