"""
YOLOv8-seg Training Script for Vehicle Damage Detection

This script trains a YOLOv8-segmentation model on car damage data
in COCO format (CarDD dataset or similar).
"""

from ultralytics import YOLO
from pathlib import Path
import yaml


def create_data_yaml(
    dataset_path: str,
    output_path: str = "dataset.yaml",
    classes: list = None,
    train_path: str = None,
    val_path: str = None,
    test_path: str = None
):
    """
    Create YOLOv8 data.yaml file for any dataset.
    
    Args:
        dataset_path: Path to dataset root directory
        output_path: Path to save data.yaml
        classes: List of class names (if None, will use default car damage classes)
        train_path: Relative path to train images (default: auto-detect)
        val_path: Relative path to val images (default: auto-detect)
        test_path: Relative path to test images (default: auto-detect, optional)
    
    Dataset structure should be one of:
        Option 1 (YOLO format):
            dataset/
                images/
                    train/
                    val/
                    test/ (optional)
                labels/
                    train/
                    val/
                    test/ (optional)
        
        Option 2 (Roboflow format):
            dataset/
                train/
                    images/
                    labels/
                valid/ or val/
                    images/
                    labels/
                test/ (optional)
                    images/
                    labels/
    """
    dataset_path = Path(dataset_path)
    
    # Default classes if not provided
    if classes is None:
        classes = [
            'scratch',
            'dent',
            'broken_glass',
            'crack',
            'paint_transfer',
            'structural_damage'
        ]
        print(f"Using default classes: {classes}")
        print("  (To customize, pass classes=['class1', 'class2', ...] to create_data_yaml)")
    
    # Auto-detect dataset structure
    if train_path is None or val_path is None:
        train_path, val_path, test_path = _detect_dataset_structure(dataset_path)
    
    # Create data.yaml content
    data_yaml = {
        'path': str(dataset_path.absolute()),
        'train': train_path,
        'val': val_path,
        'nc': len(classes),
        'names': classes
    }
    
    # Add test if available
    if test_path:
        data_yaml['test'] = test_path
    
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created {output_path}")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    if test_path:
        print(f"  Test: {test_path}")
    print(f"  Classes ({len(classes)}): {classes}")
    
    return output_path


def _detect_dataset_structure(dataset_path: Path) -> tuple:
    """
    Auto-detect dataset structure and return train/val/test paths.
    
    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    dataset_path = Path(dataset_path)
    
    # Check for YOLO format (images/train, images/val)
    if (dataset_path / "images" / "train").exists():
        train_path = "images/train"
        val_path = "images/val"
        test_path = "images/test" if (dataset_path / "images" / "test").exists() else None
        return train_path, val_path, test_path
    
    # Check for Roboflow format (train/, valid/ or val/)
    if (dataset_path / "train").exists():
        train_path = "train/images"
        # Check for 'valid' or 'val'
        if (dataset_path / "valid").exists():
            val_path = "valid/images"
        elif (dataset_path / "val").exists():
            val_path = "val/images"
        else:
            raise ValueError("Could not find validation split. Expected 'valid/' or 'val/' directory.")
        
        test_path = None
        if (dataset_path / "test").exists():
            test_path = "test/images"
        
        return train_path, val_path, test_path
    
    # Default fallback
    print("Warning: Could not auto-detect dataset structure. Using defaults.")
    print("  Expected structure:")
    print("    dataset/images/train/ and dataset/images/val/")
    print("    OR")
    print("    dataset/train/images/ and dataset/valid/images/")
    return "images/train", "images/val", None


def train_yolov8_detect(
    data_yaml: str,
    model_size: str = 's',  # n, s, m, l, x
    epochs: int = 50,
    imgsz: int = 1024,
    batch_size: int = 16,
    device: str = None,  # Auto-detect if None, '0' for GPU, 'cpu' for CPU
    project: str = 'runs/detect',
    name: str = 'car_damage_yolov8'
):
    """
    Train YOLOv8 detection model for vehicle damage detection.
    
    Note: This uses detection (bounding boxes) instead of segmentation (masks).
    For segmentation, you need polygon/mask annotations in your dataset.
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Input image size (longest side)
        batch_size: Batch size
        device: Device to use (None=auto-detect, '0' for GPU, 'cpu' for CPU)
        project: Project directory for outputs
        name: Experiment name
    """
    # Auto-detect device if not specified
    if device is None:
        try:
            import torch
            if torch.cuda.is_available():
                device = '0'
                print(f"✓ GPU detected: Using device {device}")
            else:
                device = 'cpu'
                print("⚠ No GPU detected: Using CPU (training will be slower)")
                # Reduce batch size and image size for CPU to avoid OOM
                if batch_size > 4:
                    print(f"  Reducing batch size from {batch_size} to 4 for CPU training (memory optimization)")
                    batch_size = 4
                if imgsz > 640:
                    print(f"  Reducing image size from {imgsz} to 640 for CPU training (memory optimization)")
                    imgsz = 640
        except ImportError:
            device = 'cpu'
            print("⚠ PyTorch not available: Using CPU")
    
    # Load model (detection, not segmentation, since dataset has bboxes not masks)
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading model: {model_name}")
    print("Note: Using detection model (your dataset has bounding boxes, not masks)")
    model = YOLO(model_name)
    
    # Train
    print(f"Starting training with config:")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    
    # Reduce workers for CPU/Rosetta compatibility and memory usage
    workers = 2 if device == 'cpu' else 8
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=workers,  # Reduced for CPU/Rosetta compatibility
        project=project,
        name=name,
        # Augmentation settings
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,   # Saturation augmentation
        hsv_v=0.4,   # Value augmentation
        degrees=10,  # Rotation
        translate=0.1,
        scale=0.5,
        shear=2,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5 if device == 'cpu' else 1.0,  # Reduce mosaic for CPU (memory)
        mixup=0.0 if device == 'cpu' else 0.1,  # Disable mixup for CPU (memory)
        copy_paste=0.0,
        # Training settings
        patience=10,  # Early stopping patience
        save=True,
        save_period=5,  # Save checkpoint every 5 epochs (more frequent for CPU)
        val=True,
        plots=True,
        # Loss weights (adjust for class imbalance)
        cls=0.5,
        box=7.5,
        dfl=1.5,
    )
    
    print(f"Training completed!")
    print(f"Results saved to: {Path(project) / name}")
    print(f"Best model: {Path(project) / name / 'weights' / 'best.pt'}")
    
    return model, results


def convert_coco_to_yolo_format(
    coco_json_path: str,
    images_dir: str,
    output_dir: str
):
    """
    Convert COCO format annotations to YOLOv8 format.
    
    This is a placeholder - you'll need to implement COCO to YOLO conversion
    or use existing tools like roboflow or pycocotools.
    
    Args:
        coco_json_path: Path to COCO annotations JSON
        images_dir: Directory containing images
        output_dir: Output directory for YOLO format
    """
    print("COCO to YOLO conversion not implemented in this script.")
    print("Use tools like:")
    print("  - Roboflow (roboflow.com) - can export directly to YOLOv8")
    print("  - pycocotools + custom script")
    print("  - https://github.com/ultralytics/ultralytics (has conversion utilities)")


if __name__ == "__main__":
    """
    Ready to train! Your dataset is configured and ready to go.
    """
    import os
    
    # Fix for Apple M1 running under Rosetta (x86-64 emulation)
    # Set environment variable to skip Polars CPU check to avoid bus errors
    os.environ['POLARS_SKIP_CPU_CHECK'] = '1'
    
    # Dataset path (already configured)
    dataset_path = "./data"
    data_yaml_path = "./data/data.yaml"
    
    # Your dataset has 17 classes - all configured in data.yaml
    # Classes: Bodypanel-Dent, Front-Windscreen-Damage, Headlight-Damage, etc.
    
    print("=" * 60)
    print("Dataset Configuration:")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Data YAML: {data_yaml_path}")
    print(f"  Classes: 17 damage types (see data/data.yaml)")
    print("=" * 60)
    print()
    
    # Train model
    print("Starting training...")
    print("Note: Using smaller settings for CPU training to avoid memory issues")
    print("Note: Training detection model (bboxes) - dataset doesn't have segmentation masks")
    model, results = train_yolov8_detect(
        data_yaml=data_yaml_path,
        model_size='n',  # Use 'n' (nano) for CPU to reduce memory usage
        epochs=50,
        imgsz=640,  # Smaller image size for CPU
        batch_size=4,  # Smaller batch for CPU
        device=None  # Auto-detect GPU/CPU
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.val()
    print(f"\nResults:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"\nModel saved to: runs/detect/car_damage_yolov8/weights/best.pt")
    print("\nTraining complete! 🎉")
    print("\nNote: This is a detection model (bounding boxes).")
    print("For segmentation (masks), you'll need a dataset with polygon/mask annotations.")

