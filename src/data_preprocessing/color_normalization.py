# Color Normalization and Illuminant Invariance
import cv2
import numpy as np

def gray_world_normalization(image_bgr):
    """
    White balance and illuminant correction:
    Adjusts channels proportionally based on maximum illumination while preserving tissue chromaticity.
    """
    img_float = image_bgr.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    # 98th percentile estimate of illuminant white
    p_b = np.percentile(b, 98) + 1e-6
    p_g = np.percentile(g, 98) + 1e-6
    p_r = np.percentile(r, 98) + 1e-6
    
    max_white = max(p_b, p_g, p_r)
    scale_b = np.clip(max_white / p_b, 0.85, 1.25)
    scale_g = np.clip(max_white / p_g, 0.85, 1.25)
    scale_r = np.clip(max_white / p_r, 0.85, 1.25)
    
    b_norm = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g_norm = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r_norm = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    return cv2.merge([b_norm, g_norm, r_norm])

def normalize_with_color_card(image_bgr):
    return gray_world_normalization(image_bgr)
