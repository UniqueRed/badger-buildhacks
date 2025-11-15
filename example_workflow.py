"""
Example Workflow: Complete end-to-end example

This demonstrates the full pipeline with a concrete example calculation
as described in the plan.
"""

import numpy as np
from severity_features import SeverityFeatureExtractor
from cost_estimator import HybridCostEstimator


def example_calculation():
    """
    Example calculation from the plan:
    - Mask area = 2,500 px
    - Vehicle bbox area = 250,000 px → area_ratio = 0.01 (1%)
    - Detected part = front bumper
    - Heuristic: area_ratio 1% → likely repair (not replace)
    - Rule cost = paint + labor_repair = $150 + $120 = $270
    - ML model predicts $320
    - With α = 0.4 → pred_cost = 270*(0.6) + 320*(0.4) = $296
    """
    
    # Initialize components
    feature_extractor = SeverityFeatureExtractor()
    cost_estimator = HybridCostEstimator(
        cost_table_path="part_costs.csv",
        ml_model_path=None,  # No ML model yet
        hourly_rate=80.0,
        ml_weight=0.4  # α = 0.4
    )
    
    # Simulate mask and image
    # Create a small damage mask (e.g., 50x50 pixels = 2500 pixels)
    mask = np.zeros((500, 500), dtype=np.uint8)
    mask[200:250, 200:250] = 1  # 50x50 = 2500 pixels
    
    # Simulate image
    image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    # Vehicle bbox (e.g., 500x500 = 250,000 pixels)
    vehicle_bbox = (0, 0, 500, 500)
    
    # Extract features
    features = feature_extractor.extract_features(
        mask=mask,
        image=image,
        vehicle_bbox=vehicle_bbox,
        confidence=0.95
    )
    
    print("Extracted Features:")
    print(f"  Mask area: {features['mask_area_pixels']:.0f} pixels")
    print(f"  Area ratio: {features['area_ratio']:.4f} ({features['area_ratio']*100:.2f}%)")
    print(f"  Compactness: {features['compactness']:.2f}")
    print(f"  Color change score: {features['color_change_score']:.2f}")
    
    # Compute severity
    severity_class, severity_score = feature_extractor.compute_severity_class(features)
    print(f"\nSeverity: {severity_class} (score: {severity_score:.2f})")
    
    # Estimate cost
    cost_result = cost_estimator.estimate(
        part_name="front_bumper",
        damage_class="scratch",
        severity_class=severity_class,
        features=features,
        vehicle_info={'year': 2020, 'mileage': 30000}
    )
    
    print("\nCost Estimate:")
    print(f"  Rule-based cost: ${cost_result['rule_based_cost']:.2f}")
    print(f"  ML cost: ${cost_result['ml_cost']:.2f} (not available)")
    print(f"  Final cost: ${cost_result['final_cost']:.2f}")
    print(f"  Breakdown:")
    print(f"    Part cost: ${cost_result['rule_breakdown']['part_cost']:.2f}")
    print(f"    Labor cost: ${cost_result['rule_breakdown']['labor_cost']:.2f} ({cost_result['rule_breakdown']['labor_hours']:.1f} hrs)")
    print(f"    Paint cost: ${cost_result['rule_breakdown']['paint_cost']:.2f}")
    print(f"    Action: {cost_result['rule_breakdown']['replace_or_repair']}")
    
    # Expected calculation:
    # Rule cost should be around $270 (paint $150 + labor 1.5h @ $80 = $120)
    # With ML weight 0.4 and ML cost $320:
    # Final = 270 * 0.6 + 320 * 0.4 = 162 + 128 = $290
    
    print("\n" + "="*50)
    print("Expected from plan:")
    print("  Rule cost: $270 (paint $150 + labor $120)")
    print("  ML cost: $320")
    print("  Final (α=0.4): $270*0.6 + $320*0.4 = $296")
    print("="*50)


def example_with_multiple_damages():
    """Example with multiple damage instances."""
    
    feature_extractor = SeverityFeatureExtractor()
    cost_estimator = HybridCostEstimator(
        cost_table_path="part_costs.csv",
        hourly_rate=80.0,
        ml_weight=0.3
    )
    
    # Simulate multiple damages
    damages = [
        {
            'mask_area': 2500,  # Small scratch
            'vehicle_area': 250000,
            'part': 'front_bumper',
            'damage_class': 'scratch',
            'compactness': 150.0  # High = scratch
        },
        {
            'mask_area': 15000,  # Larger dent
            'vehicle_area': 250000,
            'part': 'door',
            'damage_class': 'dent',
            'compactness': 30.0  # Low = dent
        },
        {
            'mask_area': 5000,
            'vehicle_area': 250000,
            'part': 'headlight',
            'damage_class': 'broken_glass',
            'compactness': 50.0
        }
    ]
    
    total_cost = 0.0
    
    print("Processing multiple damages:\n")
    
    for i, damage in enumerate(damages, 1):
        # Create mock features
        features = {
            'mask_area_pixels': damage['mask_area'],
            'area_ratio': damage['mask_area'] / damage['vehicle_area'],
            'compactness': damage['compactness'],
            'color_change_score': 40.0,
            'depth_proxy': 0.0,
            'confidence': 0.9
        }
        
        severity_class, severity_score = feature_extractor.compute_severity_class(features)
        
        cost_result = cost_estimator.estimate(
            part_name=damage['part'],
            damage_class=damage['damage_class'],
            severity_class=severity_class,
            features=features
        )
        
        print(f"Damage {i}: {damage['damage_class']} on {damage['part']}")
        print(f"  Severity: {severity_class}")
        print(f"  Cost: ${cost_result['final_cost']:.2f}")
        print()
        
        total_cost += cost_result['final_cost']
    
    print(f"Total estimated repair cost: ${total_cost:.2f}")


if __name__ == "__main__":
    print("Example 1: Single damage calculation\n")
    example_calculation()
    
    print("\n\nExample 2: Multiple damages\n")
    example_with_multiple_damages()

