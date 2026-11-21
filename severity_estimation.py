"""
severity_estimation.py
-----------------------
Estimates how severe a leaf disease is by measuring what fraction of the
leaf area is discolored / damaged, using classical color-based segmentation
(HSV thresholding). This does NOT require severity-labeled training data,
which is normally the hard part of this project.

Approach:
1. Segment the leaf from the background (green + brown/yellow pixels).
2. Within the leaf, segment "healthy green" pixels vs "damaged" pixels
   (brown spots, yellowing, lesions).
3. severity % = damaged_pixels / total_leaf_pixels * 100
4. Map that percentage to a human-readable severity band.
"""

import cv2
import numpy as np


def _leaf_mask(hsv: np.ndarray) -> np.ndarray:
    """Mask covering the whole leaf (green + yellow/brown regions), separating
    it from a typically white/gray/black background."""
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_brown = np.array([5, 30, 20])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    leaf = cv2.bitwise_or(green_mask, brown_mask)
    kernel = np.ones((5, 5), np.uint8)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, kernel)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, kernel)
    return leaf


def _healthy_mask(hsv: np.ndarray) -> np.ndarray:
    """Mask covering healthy, vividly green tissue only."""
    lower_healthy = np.array([35, 40, 40])
    upper_healthy = np.array([85, 255, 255])
    return cv2.inRange(hsv, lower_healthy, upper_healthy)


def estimate_severity(image_bgr: np.ndarray) -> dict:
    """
    Args:
        image_bgr: image as read by cv2.imread (BGR format), or an RGB numpy
                   array converted to BGR before calling this function.

    Returns:
        dict with keys: severity_percent (float), severity_label (str),
        leaf_pixel_count (int), damaged_pixel_count (int)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    leaf = _leaf_mask(hsv)
    healthy = _healthy_mask(hsv)
    healthy = cv2.bitwise_and(healthy, leaf)  # healthy pixels must be inside leaf

    leaf_count = int(np.count_nonzero(leaf))
    healthy_count = int(np.count_nonzero(healthy))

    if leaf_count == 0:
        return {
            "severity_percent": 0.0,
            "severity_label": "No leaf detected",
            "leaf_pixel_count": 0,
            "damaged_pixel_count": 0,
        }

    damaged_count = max(leaf_count - healthy_count, 0)
    severity_percent = round((damaged_count / leaf_count) * 100, 2)

    if severity_percent < 5:
        label = "Healthy / negligible damage"
    elif severity_percent < 20:
        label = "Mild"
    elif severity_percent < 45:
        label = "Moderate"
    else:
        label = "Severe"

    return {
        "severity_percent": severity_percent,
        "severity_label": label,
        "leaf_pixel_count": leaf_count,
        "damaged_pixel_count": damaged_count,
    }


def get_damage_overlay(image_bgr: np.ndarray) -> np.ndarray:
    """Returns an image with damaged regions highlighted in red, for visual
    display in the app (helps the demo look convincing)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    leaf = _leaf_mask(hsv)
    healthy = _healthy_mask(hsv)
    healthy = cv2.bitwise_and(healthy, leaf)
    damaged = cv2.subtract(leaf, healthy)

    overlay = image_bgr.copy()
    overlay[damaged > 0] = [0, 0, 255]  # red in BGR
    blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
    return blended
