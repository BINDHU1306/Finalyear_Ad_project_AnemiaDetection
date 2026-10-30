# Robust ROI Segmentation for Anemia Screening
import cv2
import numpy as np

def segment_nail_bed(image_bgr):
    """
    Isolate the vascular nail bed using chromatic contrast and spatial prior.
    """
    h, w = image_bgr.shape[:2]
    b, g, r = cv2.split(image_bgr)
    diff = 2.0 * r.astype(float) - g.astype(float) - b.astype(float)
    
    # Exclude color card in top-left
    work_mask = np.ones((h, w), dtype=np.uint8) * 255
    work_mask[0:int(h * 0.25), 0:int(w * 0.35)] = 0
    
    # Nail bed spatial prior
    center_prior = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(center_prior, (w // 2, int(h * 0.45)), (int(w * 0.35), int(h * 0.35)), 0, 0, 360, 1.0, -1)
    
    scored = diff * center_prior * (work_mask > 0)
    pos_scores = scored[scored > 0]
    if len(pos_scores) > 50:
        thresh_val = np.percentile(pos_scores, 60)
        mask = (scored > thresh_val).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            final_mask = np.zeros_like(mask)
            cv2.drawContours(final_mask, [c], -1, 255, -1)
            x, y, bw, bh = cv2.boundingRect(c)
        else:
            final_mask = np.zeros_like(mask)
            cv2.ellipse(final_mask, (w // 2, int(h * 0.45)), (int(w * 0.25), int(h * 0.25)), 0, 0, 360, 255, -1)
            x, y, bw, bh = w // 4, int(h * 0.2), w // 2, h // 2
    else:
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(final_mask, (w // 2, int(h * 0.45)), (int(w * 0.25), int(h * 0.25)), 0, 0, 360, 255, -1)
        x, y, bw, bh = w // 4, int(h * 0.2), w // 2, h // 2
        
    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=final_mask)
    return masked_img, final_mask, (x, y, bw, bh)

def segment_palm_region(image_bgr):
    """
    Isolate central thenar/hypothenar vascular pallor zones on palm.
    """
    h, w = image_bgr.shape[:2]
    b, g, r = cv2.split(image_bgr)
    diff = 2.0 * r.astype(float) - g.astype(float) - b.astype(float)
    
    work_mask = np.ones((h, w), dtype=np.uint8) * 255
    work_mask[0:int(h * 0.25), 0:int(w * 0.35)] = 0
    
    center_prior = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(center_prior, (w // 2, int(h * 0.58)), (int(w * 0.32), int(h * 0.32)), 0, 0, 360, 1.0, -1)
    
    scored = diff * center_prior * (work_mask > 0)
    pos_scores = scored[scored > 0]
    if len(pos_scores) > 50:
        thresh_val = np.percentile(pos_scores, 60)
        mask = (scored > thresh_val).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            final_mask = np.zeros_like(mask)
            cv2.drawContours(final_mask, [c], -1, 255, -1)
            x, y, bw, bh = cv2.boundingRect(c)
        else:
            final_mask = np.zeros_like(mask)
            cv2.ellipse(final_mask, (w // 2, int(h * 0.58)), (int(w * 0.25), int(h * 0.25)), 0, 0, 360, 255, -1)
            x, y, bw, bh = w // 4, int(h * 0.3), w // 2, h // 2
    else:
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(final_mask, (w // 2, int(h * 0.58)), (int(w * 0.25), int(h * 0.25)), 0, 0, 360, 255, -1)
        x, y, bw, bh = w // 4, int(h * 0.3), w // 2, h // 2
        
    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=final_mask)
    return masked_img, final_mask, (x, y, bw, bh)

def segment_conjunctiva_region(image_bgr):
    """
    Isolate lower palpebral conjunctival mucosal bed.
    """
    h, w = image_bgr.shape[:2]
    b, g, r = cv2.split(image_bgr)
    diff = 2.0 * r.astype(float) - g.astype(float) - b.astype(float)
    
    work_mask = np.ones((h, w), dtype=np.uint8) * 255
    work_mask[0:int(h * 0.25), 0:int(w * 0.35)] = 0
    
    center_prior = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(center_prior, (w // 2, int(h * 0.62)), (int(w * 0.35), int(h * 0.20)), 0, 0, 180, 1.0, -1)
    
    scored = diff * center_prior * (work_mask > 0)
    pos_scores = scored[scored > 0]
    if len(pos_scores) > 50:
        thresh_val = np.percentile(pos_scores, 60)
        mask = (scored > thresh_val).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            final_mask = np.zeros_like(mask)
            cv2.drawContours(final_mask, [c], -1, 255, -1)
            x, y, bw, bh = cv2.boundingRect(c)
        else:
            final_mask = np.zeros_like(mask)
            cv2.ellipse(final_mask, (w // 2, int(h * 0.62)), (int(w * 0.3), int(h * 0.16)), 0, 0, 180, 255, -1)
            x, y, bw, bh = int(w * 0.2), int(h * 0.5), int(w * 0.6), int(h * 0.3)
    else:
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(final_mask, (w // 2, int(h * 0.62)), (int(w * 0.3), int(h * 0.16)), 0, 0, 180, 255, -1)
        x, y, bw, bh = int(w * 0.2), int(h * 0.5), int(w * 0.6), int(h * 0.3)
        
    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=final_mask)
    return masked_img, final_mask, (x, y, bw, bh)

def auto_segment(image_bgr, modality='fingernail'):
    mod = modality.lower()
    if 'nail' in mod:
        return segment_nail_bed(image_bgr)
    elif 'palm' in mod:
        return segment_palm_region(image_bgr)
    elif 'conj' in mod or 'eye' in mod:
        return segment_conjunctiva_region(image_bgr)
    else:
        return segment_nail_bed(image_bgr)
