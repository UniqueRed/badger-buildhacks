# Quick Start: Using Any Dataset

Yes! You can use **any dataset** - just change the path. The script will auto-detect the dataset structure.

## Step 1: Prepare Your Dataset

Your dataset should be in **YOLOv8 format** (or Roboflow format). Structure:

```
your_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

**OR** Roboflow format:
```
your_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/  (or val/)
│   ├── images/
│   └── labels/
└── test/  (optional)
    ├── images/
    └── labels/
```

## Step 2: Create data.yaml

Open `train_yolov8.py` and uncomment Example 1:

```python
data_yaml = create_data_yaml(
    dataset_path="./data/your_dataset",  # <-- Change this!
    output_path="my_dataset.yaml"
)
```

That's it! The script will:
- Auto-detect your dataset structure
- Use default classes (or you can specify custom ones)

## Step 3: Customize Classes (Optional)

If your dataset has different class names, use Example 2:

```python
data_yaml = create_data_yaml(
    dataset_path="./data/your_dataset",
    output_path="my_dataset.yaml",
    classes=['your_class_1', 'your_class_2', 'your_class_3']  # <-- Your classes
)
```

## Step 4: Train

Uncomment the training section:

```python
model, results = train_yolov8_seg(
    data_yaml="my_dataset.yaml",  # <-- Use the yaml you created
    model_size='s',
    epochs=50,
    imgsz=1024,
    batch_size=16,
    device='0'  # or 'cpu'
)
```

## Converting from COCO Format

If you have a COCO-format dataset, convert it first:

1. **Use Roboflow** (easiest):
   - Upload your COCO dataset to [roboflow.com](https://roboflow.com)
   - Export as YOLOv8 format
   - Download

2. **Use conversion script**:
   - See `convert_coco_to_yolo_format()` function (placeholder)
   - Or use: https://github.com/ultralytics/ultralytics (has conversion tools)

## That's It!

The script works with **any YOLOv8-format dataset**. Just change the path!

