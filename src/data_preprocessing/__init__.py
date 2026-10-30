# Data Preprocessing Package
from .segmentation import auto_segment, segment_nail_bed, segment_palm_region, segment_conjunctiva_region
from .color_normalization import gray_world_normalization, normalize_with_color_card
from .feature_extraction import extract_color_features

try:
    from . import color_correction
except ImportError:
    pass
