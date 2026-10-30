# Color and Texture Feature Extraction for Anemia Screening
import cv2
import numpy as np
from scipy.stats import skew, kurtosis

def extract_color_features(masked_bgr, mask=None):
    """
    Extract comprehensive clinical color features from non-zero ROI pixels:
    - RGB Statistics (mean, std, skewness, kurtosis)
    - HSV Statistics (mean, std for Hue, Saturation, Value)
    - CIELAB Statistics (mean, std for L*, a*, b*)
    - Clinical Pallor Biomarkers (ERI, AEI, Perfusion Ratio, Pallor Index)
    """
    if mask is not None:
        valid_pixels_bgr = masked_bgr[mask > 0]
    else:
        valid = np.any(masked_bgr > 0, axis=-1)
        valid_pixels_bgr = masked_bgr[valid]
        
    if valid_pixels_bgr.size == 0 or len(valid_pixels_bgr) < 10:
        # Fallback values if mask is empty
        return {
            'Mean_R': 190.0, 'Std_R': 20.0, 'Skew_R': 0.0, 'Kurt_R': 0.0,
            'Mean_G': 150.0, 'Std_G': 20.0, 'Skew_G': 0.0, 'Kurt_G': 0.0,
            'Mean_B': 140.0, 'Std_B': 20.0, 'Skew_B': 0.0, 'Kurt_B': 0.0,
            'Mean_H': 10.0, 'Mean_S': 80.0, 'Mean_V': 180.0,
            'Lab_L': 65.0, 'Lab_a': 18.0, 'Lab_b': 12.0,
            'ERI': 0.65, 'AEI': 0.27, 'Perfusion_Ratio': 0.39, 'Pallor_Index': 3.6
        }
        
    # RGB values (note OpenCV is BGR)
    b_vals = valid_pixels_bgr[:, 0].astype(np.float64)
    g_vals = valid_pixels_bgr[:, 1].astype(np.float64)
    r_vals = valid_pixels_bgr[:, 2].astype(np.float64)
    
    mean_r, std_r = np.mean(r_vals), np.std(r_vals)
    mean_g, std_g = np.mean(g_vals), np.std(g_vals)
    mean_b, std_b = np.mean(b_vals), np.std(b_vals)
    
    skew_r = float(skew(r_vals)) if std_r > 1e-4 else 0.0
    skew_g = float(skew(g_vals)) if std_g > 1e-4 else 0.0
    skew_b = float(skew(b_vals)) if std_b > 1e-4 else 0.0
    
    kurt_r = float(kurtosis(r_vals)) if std_r > 1e-4 else 0.0
    kurt_g = float(kurtosis(g_vals)) if std_g > 1e-4 else 0.0
    kurt_b = float(kurtosis(b_vals)) if std_b > 1e-4 else 0.0
    
    # HSV Conversion for valid pixels
    reshaped_bgr = valid_pixels_bgr.reshape(-1, 1, 3)
    hsv_pixels = cv2.cvtColor(reshaped_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mean_h = np.mean(hsv_pixels[:, 0])
    mean_s = np.mean(hsv_pixels[:, 1])
    mean_v = np.mean(hsv_pixels[:, 2])
    
    # CIELAB Conversion for valid pixels
    lab_pixels = cv2.cvtColor(reshaped_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    # In OpenCV 8-bit LAB: L: 0..255 (scaled by 255/100), a: 0..255 (offset by 128), b: 0..255 (offset by 128)
    lab_l = np.mean(lab_pixels[:, 0]) * (100.0 / 255.0)
    lab_a = np.mean(lab_pixels[:, 1]) - 128.0
    lab_b = np.mean(lab_pixels[:, 2]) - 128.0
    
    # Clinical pallor indicators
    eri = mean_r / (mean_g + mean_b + 1e-6)
    aei = lab_a / (lab_l + 1e-6)
    perfusion_ratio = mean_r / (mean_r + mean_g + mean_b + 1e-6)
    pallor_index = lab_l / (abs(lab_a) + 1e-6)
    
    return {
        'Mean_R': round(float(mean_r), 2),
        'Std_R': round(float(std_r), 2),
        'Skew_R': round(float(skew_r), 2),
        'Kurt_R': round(float(kurt_r), 2),
        'Mean_G': round(float(mean_g), 2),
        'Std_G': round(float(std_g), 2),
        'Skew_G': round(float(skew_g), 2),
        'Kurt_G': round(float(kurt_g), 2),
        'Mean_B': round(float(mean_b), 2),
        'Std_B': round(float(std_b), 2),
        'Skew_B': round(float(skew_b), 2),
        'Kurt_B': round(float(kurt_b), 2),
        'Mean_H': round(float(mean_h), 2),
        'Mean_S': round(float(mean_s), 2),
        'Mean_V': round(float(mean_v), 2),
        'Lab_L': round(float(lab_l), 2),
        'Lab_a': round(float(lab_a), 2),
        'Lab_b': round(float(lab_b), 2),
        'ERI': round(float(eri), 3),
        'AEI': round(float(aei), 3),
        'Perfusion_Ratio': round(float(perfusion_ratio), 3),
        'Pallor_Index': round(float(pallor_index), 3)
    }
