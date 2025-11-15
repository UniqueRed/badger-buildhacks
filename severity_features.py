"""
Severity Feature Extraction Module

Extracts geometric and visual features from predicted damage masks
to compute severity scores for vehicle damage assessment.

FIXED VERSION: Uses OpenCV for perimeter calculation instead of skimage
"""

import numpy as np
from typing import Dict, Tuple, Optional
import cv2


class SeverityFeatureExtractor:
    """
    Extracts features from damage masks to compute severity scores.
    
    Features include:
    - Mask area and area ratio
    - Compactness (perimeter²/area)
    - Texture/color change metrics
    - Depth proxy (optional, requires depth model)
    - Part overlap information
    """
    
    def __init__(self, depth_model=None):
        """
        Args:
            depth_model: Optional monocular depth model (e.g., MiDaS) for depth estimation
        """
        self.depth_model = depth_model
    
    def extract_features(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        vehicle_bbox: Optional[Tuple[int, int, int, int]] = None,
        part_mask: Optional[np.ndarray] = None,
        confidence: float = 1.0
    ) -> Dict[str, float]:
        """
        Extract all severity features from a damage mask.
        
        Args:
            mask: Binary mask (H, W) of damage region
            image: Original RGB image (H, W, 3)
            bbox: Bounding box of damage (x1, y1, x2, y2)
            vehicle_bbox: Bounding box of entire vehicle (x1, y1, x2, y2)
            part_mask: Optional binary mask of affected part (bumper, door, etc.)
            confidence: Detection confidence score
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic mask properties
        mask_area = np.sum(mask > 0)
        features['mask_area_pixels'] = float(mask_area)
        
        # Area ratio
        if vehicle_bbox is not None:
            vehicle_area = (vehicle_bbox[2] - vehicle_bbox[0]) * (vehicle_bbox[3] - vehicle_bbox[1])
            features['area_ratio'] = mask_area / vehicle_area if vehicle_area > 0 else 0.0
        elif bbox is not None:
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            features['area_ratio'] = mask_area / bbox_area if bbox_area > 0 else 0.0
        else:
            features['area_ratio'] = 0.0
        
        # Compactness (perimeter²/area) - scratches are long/thin (high), dents are compact (low)
        perimeter = self._compute_perimeter(mask)
        features['perimeter'] = float(perimeter)
        features['compactness'] = (perimeter ** 2) / mask_area if mask_area > 0 else 0.0
        
        # Texture/color change metrics
        texture_features = self._extract_texture_features(mask, image)
        features.update(texture_features)
        
        # Depth proxy (if depth model available)
        if self.depth_model is not None:
            depth_features = self._extract_depth_features(mask, image)
            features.update(depth_features)
        else:
            features['depth_proxy'] = 0.0  # Placeholder
        
        # Part overlap
        if part_mask is not None:
            overlap = np.sum((mask > 0) & (part_mask > 0))
            features['part_overlap_ratio'] = overlap / mask_area if mask_area > 0 else 0.0
        else:
            features['part_overlap_ratio'] = 0.0
        
        # Detection confidence
        features['confidence'] = confidence
        
        return features
    
    def _compute_perimeter(self, mask: np.ndarray) -> float:
        """
        Compute perimeter of mask using OpenCV contour detection.
        
        FIXED: Uses cv2.findContours instead of skimage.measure.find_contours
        to avoid compatibility issues.
        """
        # Ensure mask is uint8
        mask_uint8 = mask.astype(np.uint8)
        
        # Find contours using OpenCV
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return 0.0
        
        # Sum perimeters of all contours
        total_perimeter = sum(cv2.arcLength(contour, closed=True) for contour in contours)
        
        return total_perimeter
    
    def _extract_texture_features(
        self,
        mask: np.ndarray,
        image: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract texture and color change metrics.
        
        Compares color statistics inside mask vs surrounding region.
        """
        mask_binary = mask > 0
        
        # Get pixels inside mask
        mask_pixels = image[mask_binary]
        
        if len(mask_pixels) == 0:
            return {
                'mask_color_std': 0.0,
                'mask_color_mean': 0.0,
                'surrounding_color_std': 0.0,
                'surrounding_color_mean': 0.0,
                'color_change_score': 0.0
            }
        
        # Dilate mask to get surrounding region
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask_binary.astype(np.uint8), kernel, iterations=3)
        surrounding_mask = (dilated_mask > 0) & (~mask_binary)
        
        if np.sum(surrounding_mask) == 0:
            surrounding_pixels = mask_pixels  # Fallback
        else:
            surrounding_pixels = image[surrounding_mask]
        
        # Compute statistics
        mask_mean = np.mean(mask_pixels, axis=0)
        mask_std = np.std(mask_pixels, axis=0)
        surrounding_mean = np.mean(surrounding_pixels, axis=0)
        surrounding_std = np.std(surrounding_pixels, axis=0)
        
        # Color change score (L2 distance between means)
        color_change = np.linalg.norm(mask_mean - surrounding_mean)
        
        return {
            'mask_color_std': float(np.mean(mask_std)),
            'mask_color_mean': float(np.mean(mask_mean)),
            'surrounding_color_std': float(np.mean(surrounding_std)),
            'surrounding_color_mean': float(np.mean(surrounding_mean)),
            'color_change_score': float(color_change)
        }
    
    def _extract_depth_features(
        self,
        mask: np.ndarray,
        image: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract depth-based features using monocular depth model.
        
        This estimates relative depth changes which can indicate dents vs scratches.
        """
        if self.depth_model is None:
            return {'depth_proxy': 0.0, 'depth_variance': 0.0}
        
        # Run depth model (placeholder - would use actual model)
        # depth_map = self.depth_model(image)
        
        # For now, return placeholder
        # In production, compute depth statistics within mask
        return {
            'depth_proxy': 0.0,  # Mean depth change in mask region
            'depth_variance': 0.0  # Variance of depth in mask region
        }
    
    def compute_severity_class(
        self,
        features: Dict[str, float],
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[str, float]:
        """
        Compute severity class (minor/moderate/severe) from features.
        
        Args:
            features: Extracted feature dictionary
            thresholds: Optional custom thresholds
            
        Returns:
            Tuple of (severity_class, severity_score)
        """
        if thresholds is None:
            thresholds = {
                'area_ratio': (0.01, 0.05),  # (minor_max, moderate_max)
                'compactness': (100.0, 50.0),  # (scratch_threshold, dent_threshold)
                'color_change_score': (30.0, 60.0),  # (minor_max, moderate_max)
            }
        
        area_ratio = features.get('area_ratio', 0.0)
        compactness = features.get('compactness', 0.0)
        color_change = features.get('color_change_score', 0.0)
        depth_proxy = features.get('depth_proxy', 0.0)
        
        # Heuristic-based severity classification
        severity_score = 0.0
        
        # Area-based scoring
        if area_ratio < thresholds['area_ratio'][0]:
            severity_score += 0.3
        elif area_ratio < thresholds['area_ratio'][1]:
            severity_score += 0.6
        else:
            severity_score += 1.0
        
        # Compactness-based (scratches vs dents)
        if compactness > thresholds['compactness'][0]:  # Long scratch
            severity_score += 0.2
        elif compactness < thresholds['compactness'][1]:  # Compact dent
            severity_score += 0.5
        
        # Color change (paint transfer, bare metal)
        if color_change > thresholds['color_change_score'][1]:
            severity_score += 0.3
        
        # Depth proxy (structural damage)
        if depth_proxy > 0.5:  # Significant depth change
            severity_score += 0.4
        
        # Normalize to [0, 1]
        severity_score = min(severity_score / 2.0, 1.0)
        
        # Classify
        if severity_score < 0.33:
            severity_class = 'minor'
        elif severity_score < 0.67:
            severity_class = 'moderate'
        else:
            severity_class = 'severe'
        
        return severity_class, severity_score