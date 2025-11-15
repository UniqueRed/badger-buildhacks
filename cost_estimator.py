"""
Cost Estimation Module

Hybrid rule-based + ML approach for vehicle damage repair cost estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class RuleBasedCostEstimator:
    """
    Rule-based cost estimator using part cost tables and labor rates.
    """
    
    def __init__(self, cost_table_path: str, hourly_rate: float = 80.0):
        """
        Args:
            cost_table_path: Path to CSV file with part costs
            hourly_rate: Shop hourly labor rate in USD
        """
        self.hourly_rate = hourly_rate
        self.cost_table = pd.read_csv(cost_table_path)
        self.cost_table.set_index('part_name', inplace=True)
    
    def estimate_cost(
        self,
        part_name: str,
        damage_class: str,
        severity_class: str,
        area_ratio: float,
        replace_or_repair: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Estimate repair cost for a single damage instance.
        
        Args:
            part_name: Name of affected part (e.g., 'bumper', 'door')
            damage_class: Type of damage (e.g., 'scratch', 'dent', 'broken_glass')
            severity_class: Severity level ('minor', 'moderate', 'severe')
            area_ratio: Ratio of damage area to part area
            replace_or_repair: 'replace' or 'repair' (auto-determined if None)
            
        Returns:
            Dictionary with cost breakdown
        """
        if part_name not in self.cost_table.index:
            # Default values for unknown parts
            part_cost = 500.0
            replace_labor_hours = 3.0
            repair_labor_hours = 1.5
        else:
            row = self.cost_table.loc[part_name]
            part_cost = float(row['part_cost'])
            replace_labor_hours = float(row['replace_labor_hours'])
            repair_labor_hours = float(row['repair_labor_hours'])
        
        # Determine replace vs repair
        if replace_or_repair is None:
            replace_or_repair = self._determine_replace_or_repair(
                damage_class, severity_class, area_ratio
            )
        
        # Base costs
        if replace_or_repair == 'replace':
            labor_hours = replace_labor_hours
            part_cost_used = part_cost
        else:  # repair
            labor_hours = repair_labor_hours
            part_cost_used = 0.0  # No part replacement
        
        labor_cost = labor_hours * self.hourly_rate
        
        # Paint and materials (if needed)
        paint_cost = self._estimate_paint_cost(part_name, severity_class, area_ratio)
        
        # Total
        total_cost = part_cost_used + labor_cost + paint_cost
        
        return {
            'part_cost': part_cost_used,
            'labor_cost': labor_cost,
            'labor_hours': labor_hours,
            'paint_cost': paint_cost,
            'total_cost': total_cost,
            'replace_or_repair': replace_or_repair
        }
    
    def _determine_replace_or_repair(
        self,
        damage_class: str,
        severity_class: str,
        area_ratio: float
    ) -> str:
        """Determine if part should be replaced or repaired."""
        # Special cases
        if damage_class in ['broken_glass', 'shattered']:
            return 'replace'
        
        if severity_class == 'severe' and area_ratio > 0.1:
            return 'replace'
        
        if severity_class == 'minor':
            return 'repair'
        
        # Moderate damage: depends on part type and area
        if area_ratio > 0.05:
            return 'replace'
        
        return 'repair'
    
    def _estimate_paint_cost(
        self,
        part_name: str,
        severity_class: str,
        area_ratio: float
    ) -> float:
        """Estimate paint and materials cost."""
        # Base paint cost per part
        base_paint_cost = {
            'bumper': 150.0,
            'door': 200.0,
            'fender': 180.0,
            'hood': 250.0,
            'trunk': 200.0,
            'headlight': 0.0,  # No paint for headlights
            'windshield': 0.0,
            'side_mirror': 50.0,
        }
        
        cost = base_paint_cost.get(part_name, 150.0)
        
        # Adjust based on severity and area
        if severity_class == 'minor':
            cost *= 0.5  # Touch-up
        elif severity_class == 'moderate':
            cost *= 0.75  # Partial paint
        # severe: full paint (100%)
        
        return cost


class MLCostEstimator:
    """
    ML-based cost estimator using gradient boosted trees.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to trained model file (XGBoost/LightGBM pickle)
        """
        self.model = None
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from file."""
        import pickle
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(
        self,
        features: Dict[str, float],
        part_name: str,
        vehicle_info: Optional[Dict[str, any]] = None
    ) -> float:
        """
        Predict cost using ML model.
        
        Args:
            features: Severity features dictionary
            part_name: Affected part name
            vehicle_info: Optional vehicle make/model/year/mileage
            
        Returns:
            Predicted cost in USD
        """
        if self.model is None:
            # Return placeholder if model not loaded
            return 0.0
        
        # Convert features to model input format
        # This would need to match your training data format
        feature_vector = self._features_to_vector(features, part_name, vehicle_info)
        
        prediction = self.model.predict([feature_vector])[0]
        return max(0.0, float(prediction))  # Ensure non-negative
    
    def _features_to_vector(
        self,
        features: Dict[str, float],
        part_name: str,
        vehicle_info: Optional[Dict[str, any]]
    ) -> np.ndarray:
        """Convert features dict to model input vector."""
        # This is a placeholder - actual implementation depends on model training
        # Would include one-hot encoding for part_name, vehicle make/model, etc.
        vector = [
            features.get('area_ratio', 0.0),
            features.get('compactness', 0.0),
            features.get('color_change_score', 0.0),
            features.get('depth_proxy', 0.0),
            features.get('confidence', 1.0),
        ]
        
        # Add vehicle info if available
        if vehicle_info:
            vector.extend([
                vehicle_info.get('year', 2020),
                vehicle_info.get('mileage', 50000),
            ])
        
        return np.array(vector)


class HybridCostEstimator:
    """
    Combines rule-based and ML estimators with weighted blending.
    """
    
    def __init__(
        self,
        cost_table_path: str,
        ml_model_path: Optional[str] = None,
        hourly_rate: float = 80.0,
        ml_weight: float = 0.3
    ):
        """
        Args:
            cost_table_path: Path to part cost table CSV
            ml_model_path: Optional path to trained ML model
            hourly_rate: Shop hourly labor rate
            ml_weight: Weight for ML prediction (0.0 = rule-based only, 1.0 = ML only)
        """
        self.rule_estimator = RuleBasedCostEstimator(cost_table_path, hourly_rate)
        self.ml_estimator = MLCostEstimator(ml_model_path)
        self.ml_weight = ml_weight
    
    def estimate(
        self,
        part_name: str,
        damage_class: str,
        severity_class: str,
        features: Dict[str, float],
        vehicle_info: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Estimate cost using hybrid approach.
        
        Args:
            part_name: Affected part name
            damage_class: Type of damage
            severity_class: Severity level
            features: Extracted severity features
            vehicle_info: Optional vehicle information
            
        Returns:
            Dictionary with cost breakdown and prediction details
        """
        # Rule-based estimate
        rule_result = self.rule_estimator.estimate_cost(
            part_name=part_name,
            damage_class=damage_class,
            severity_class=severity_class,
            area_ratio=features.get('area_ratio', 0.0)
        )
        rule_cost = rule_result['total_cost']
        
        # ML estimate
        ml_cost = self.ml_estimator.predict(features, part_name, vehicle_info)
        
        # Blend predictions
        if ml_cost > 0:  # ML model available
            blended_cost = rule_cost * (1 - self.ml_weight) + ml_cost * self.ml_weight
        else:
            blended_cost = rule_cost
            self.ml_weight = 0.0  # No ML contribution
        
        return {
            'rule_based_cost': rule_cost,
            'ml_cost': ml_cost,
            'final_cost': blended_cost,
            'ml_weight': self.ml_weight,
            'rule_breakdown': rule_result
        }

