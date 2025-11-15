# Ready to Train! 🚀

Your dataset is configured and ready to go!

## Dataset Info
- **Train images**: 2,311
- **Val images**: 679
- **Test images**: 80
- **Total**: 3,070 images
- **Classes**: 17 damage types (dents on various car parts)

## Quick Start

Just run:

```bash
python3 train_yolov8.py
```

Or if you have a GPU:

```bash
python3 train_yolov8.py
# The script will use GPU automatically if available
```

## What Will Happen

1. ✅ Loads YOLOv8-seg model (small version for fast training)
2. ✅ Trains for 50 epochs on your dataset
3. ✅ Saves checkpoints every 10 epochs
4. ✅ Evaluates on validation set
5. ✅ Saves best model to: `runs/segment/car_damage_yolov8/weights/best.pt`

## Configuration

- **Model size**: `s` (small) - good balance of speed and accuracy
- **Image size**: 1024px (preserves small damage details)
- **Batch size**: 16 (adjust if you run out of memory)
- **Epochs**: 50
- **Device**: Auto-detects GPU/CPU

## Adjust Settings (Optional)

If you want to change settings, edit `train_yolov8.py`:

```python
model, results = train_yolov8_seg(
    data_yaml="./data/data.yaml",
    model_size='s',      # Change to 'm', 'l', or 'x' for larger models
    epochs=50,            # More epochs = better but slower
    imgsz=1024,           # Image size (640, 1024, 1280)
    batch_size=16,        # Reduce if out of memory
    device='0'            # '0' for GPU, 'cpu' for CPU
)
```

## After Training

Your trained model will be at:
```
runs/segment/car_damage_yolov8/weights/best.pt
```

Use it in your pipeline:
```python
from ultralytics import YOLO
model = YOLO('runs/segment/car_damage_yolov8/weights/best.pt')
```

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` to 8 or 4
- Reduce `imgsz` to 640
- Use smaller model: `model_size='n'`

**Want faster training?**
- Use `model_size='n'` (nano - fastest)
- Reduce `epochs` to 20-30
- Reduce `imgsz` to 640

**Want better accuracy?**
- Use `model_size='m'` or `'l'` (medium/large)
- Increase `epochs` to 100
- Increase `imgsz` to 1280

Happy training! 🎉

