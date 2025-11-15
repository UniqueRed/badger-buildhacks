# Vehicle Damage Assessment System

End-to-end pipeline for automated vehicle damage detection, severity scoring, and repair cost estimation using instance segmentation and hybrid rule-based + ML cost modeling.

## Project Structure

```
.
├── severity_features.py      # Feature extraction from damage masks
├── cost_estimator.py         # Rule-based + ML cost estimation
├── pipeline.py               # Main end-to-end pipeline
├── train_yolov8.py           # YOLOv8-seg training script
├── part_costs.csv            # Part cost table (rule-based)
├── example_workflow.py       # Example usage and calculations
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** For GPU support, install PyTorch separately:
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision
```

### 2. Run Example Workflow

```bash
python example_workflow.py
```

This demonstrates the complete pipeline with the example calculation from the plan:
- Mask area: 2,500 px
- Area ratio: 1%
- Part: front bumper
- Rule-based cost: ~$270
- ML cost: $320 (when available)
- Final cost (α=0.4): $296

### 3. Train Your Model

#### Option A: YOLOv8-seg (Recommended for fast iteration)

1. Prepare your dataset in YOLOv8 format (or use Roboflow to convert from COCO)
2. Create `car_damage.yaml`:
   ```yaml
   path: ./data/car_damage
   train: images/train
   val: images/val
   nc: 6
   names: ['scratch', 'dent', 'broken_glass', 'crack', 'paint_transfer', 'structural_damage']
   ```
3. Train:
   ```bash
   python train_yolov8.py
   ```
   Or use Ultralytics CLI:
   ```bash
   yolo task=segment mode=train model=yolov8s-seg.pt data=car_damage.yaml epochs=50 imgsz=1024
   ```

#### Option B: Mask R-CNN (Detectron2) - For best accuracy

See Detectron2 documentation: https://github.com/facebookresearch/detectron2

### 4. Use the Pipeline

```python
from pipeline import DamageAssessmentPipeline
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/segment/car_damage_yolov8/weights/best.pt')

# Initialize pipeline
pipeline = DamageAssessmentPipeline(
    segmentation_model=model,
    cost_table_path="part_costs.csv",
    ml_model_path=None,  # Add when ML model is trained
    hourly_rate=80.0,
    ml_weight=0.3
)

# Process image
result = pipeline.process_image(
    image_path="path/to/damaged_vehicle.jpg",
    vehicle_info={
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2020,
        'mileage': 30000
    }
)

print(f"Found {result['num_damages']} damage instances")
print(f"Total estimated cost: ${result['total_estimated_cost']:.2f}")
```

## Components

### 1. Severity Feature Extraction (`severity_features.py`)

Extracts geometric and visual features from damage masks:
- **Mask area** and **area ratio** (damage area / vehicle area)
- **Compactness** (perimeter²/area) - distinguishes scratches (high) from dents (low)
- **Texture/color change metrics** - detects paint transfer vs bare metal
- **Depth proxy** (optional, requires depth model)
- **Part overlap** information

Computes severity class: `minor`, `moderate`, or `severe`

### 2. Cost Estimation (`cost_estimator.py`)

Hybrid approach combining:
- **Rule-based estimator**: Uses part cost table (`part_costs.csv`) with labor rates
- **ML estimator**: XGBoost/LightGBM regressor (train on historical claims)
- **Blended prediction**: `final_cost = rule_cost * (1-α) + ml_cost * α`

### 3. Main Pipeline (`pipeline.py`)

End-to-end processing:
1. Run instance segmentation model
2. Extract severity features for each damage instance
3. Determine affected part (heuristic or part detection model)
4. Estimate repair cost
5. Return comprehensive assessment

### 4. Training Script (`train_yolov8.py`)

Ready-to-use YOLOv8-seg training script with:
- Data configuration
- Augmentation settings
- Training hyperparameters
- Evaluation metrics

## Dataset Recommendations

1. **CarDD (Car Damage Detection)**: 4,000 images, ~9k instances, 6 damage categories
   - Download: [CarDD Dataset](https://github.com/maijiang/CarDD)
   - Convert to COCO/YOLO format

2. **Roboflow**: Multiple car damage datasets
   - Export directly to YOLOv8 format
   - [Roboflow Car Damage](https://roboflow.com/datasets)

3. **Custom collection**: Augment with your own photos covering:
   - Different vehicle makes/models
   - Various lighting conditions
   - Multiple viewpoints
   - Different damage types

## Cost Table Schema

The `part_costs.csv` file contains:
- `part_name`: Part identifier (e.g., 'bumper', 'door')
- `part_cost`: Replacement part cost (USD)
- `replace_labor_hours`: Hours for replacement
- `repair_labor_hours`: Hours for repair
- `part_category`: Category (exterior, glass, lighting)

Customize this table based on your shop rates and part costs.

## Model Training Tips

1. **Start small**: Use YOLOv8s-seg for fast iteration
2. **Image size**: Use 1024px (longest side) to preserve small scratches
3. **Augmentation**: Enable mosaic, mixup, color jitter
4. **Class imbalance**: Oversample rare damage types or use focal loss
5. **Metrics**: Track COCO mAP, small-object AP, mask IoU
6. **Checkpointing**: Save every 10 epochs for recovery

## ML Cost Model Training

To train the ML cost regressor:

1. Collect historical repair data with:
   - Damage features (area ratio, compactness, etc.)
   - Part name
   - Vehicle info (make, model, year, mileage)
   - Actual repair cost (ground truth)

2. Train XGBoost/LightGBM:
   ```python
   import xgboost as xgb
   from cost_estimator import MLCostEstimator
   
   # Prepare features and labels
   X_train, y_train = prepare_training_data(...)
   
   # Train model
   model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
   model.fit(X_train, y_train)
   
   # Save
   import pickle
   with open('cost_model.pkl', 'wb') as f:
       pickle.dump(model, f)
   ```

3. Use in pipeline:
   ```python
   pipeline = DamageAssessmentPipeline(
       ...,
       ml_model_path="cost_model.pkl",
       ml_weight=0.4  # Increase as you trust ML more
   )
   ```

## Evaluation Metrics

- **Segmentation**: COCO mAP (AP@[.5:.95]), AP@0.5, small/medium/large AP
- **Severity classification**: Accuracy, F1, confusion matrix
- **Cost estimation**: MAE, RMSE, MAPE, "within ±10%" rate

## Production Deployment

1. **Inference speed**: YOLOv8-seg is fast; Mask R-CNN more accurate but slower
2. **Edge deployment**: Quantize model (ONNX → TensorRT/TFLite)
3. **Human-in-loop**: Always provide manual override UI
4. **Feedback loop**: Log corrections to retrain models
5. **Explainability**: Show masks, features, and cost breakdown for audit

## Next Steps

1. ✅ Download CarDD dataset and convert to YOLO format
2. ✅ Train YOLOv8-seg model (50+ epochs)
3. ✅ Build feature extractor (done)
4. ✅ Implement rule-based cost estimator (done)
5. ⏳ Collect historical repair data
6. ⏳ Train ML cost regressor
7. ⏳ Build human review UI
8. ⏳ Deploy and iterate

## References

- **CarDD Dataset**: [GitHub](https://github.com/maijiang/CarDD)
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Detectron2**: [Facebook Research](https://github.com/facebookresearch/detectron2)
- **SAM (Segment Anything)**: [Meta AI](https://github.com/facebookresearch/segment-anything)

## License

[Add your license here]

