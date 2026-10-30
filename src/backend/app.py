# HemoGlow AI - Medical Backend Server (Flask & REST API)
import os
import sys
import base64
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add root directory to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.predictor import analyze_modality_image, predict_anemia_multimodal

app = Flask(__name__, static_folder=os.path.join(root_dir, 'src', 'frontend', 'static'))
CORS(app)

frontend_dir = os.path.join(root_dir, 'src', 'frontend')
samples_dir = os.path.join(root_dir, 'data', 'samples')
saved_models_dir = os.path.join(root_dir, 'saved_models')

def bgr_to_base64_data_url(bgr_img):
    if bgr_img is None:
        return None
    success, buffer = cv2.imencode('.jpg', bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        return None
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

@app.route('/')
def serve_index():
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    if os.path.exists(os.path.join(frontend_dir, filename)):
        return send_from_directory(frontend_dir, filename)
    elif os.path.exists(os.path.join(frontend_dir, 'static', filename)):
        return send_from_directory(os.path.join(frontend_dir, 'static'), filename)
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'project': 'HemoGlow AI: Multi-Modal Non-Invasive Anemia Detection',
        'institution': 'Malla Reddy University - Department of CSE',
        'models_loaded': ['MobileNetV2', 'ResNet50', 'Random Forest', 'Logistic Regression']
    })

@app.route('/api/models-evaluation', methods=['GET'])
def get_evaluation():
    metrics_path = os.path.join(saved_models_dir, 'evaluation_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    return jsonify({'error': 'Evaluation metrics not found'}), 404

@app.route('/api/sample-images', methods=['GET'])
def get_sample_images():
    """
    Returns pre-loaded clinical cases for quick one-click screening demonstration.
    """
    cases = []
    conditions = [
        ('normal', 'Patient Case #1: Healthy Baseline', 'Female', 24, 'Non-Anemic expected (Normal vascular perfusion)'),
        ('mild', 'Patient Case #2: Mild Pallor', 'Female', 28, 'Mild Anemia expected (Borderline pallor)'),
        ('severe', 'Patient Case #3: Severe Clinical Pallor', 'Male', 45, 'Severe Anemia expected (Significant hypoperfusion)')
    ]
    
    for cond_key, label, gender, age, desc in conditions:
        case_data = {
            'id': cond_key,
            'label': label,
            'gender': gender,
            'age': age,
            'description': desc,
            'images': {}
        }
        for mod in ['fingernail', 'palm', 'conjunctiva']:
            fname = f"{mod}_{cond_key}_sample.jpg"
            fpath = os.path.join(samples_dir, mod, fname)
            if os.path.exists(fpath):
                img = cv2.imread(fpath)
                case_data['images'][mod] = bgr_to_base64_data_url(img)
        cases.append(case_data)
        
    return jsonify({'samples': cases})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        gender = request.form.get('gender', 'Female')
        patient_name = request.form.get('patient_name', 'Anonymous Patient')
        age = request.form.get('age', '26')

        nail_file = request.files.get('nail_image')
        palm_file = request.files.get('palm_image')
        conj_file = request.files.get('conjunctiva_image')

        # Also support base64 encoded images from demo buttons
        nail_b64 = request.form.get('nail_b64')
        palm_b64 = request.form.get('palm_b64')
        conj_b64 = request.form.get('conj_b64')

        def load_image_bytes(file_obj, b64_str):
            if file_obj and file_obj.filename != '':
                return file_obj.read()
            elif b64_str:
                if ',' in b64_str:
                    b64_str = b64_str.split(',', 1)[1]
                return base64.b64decode(b64_str)
            return None

        nail_bytes = load_image_bytes(nail_file, nail_b64)
        palm_bytes = load_image_bytes(palm_file, palm_b64)
        conj_bytes = load_image_bytes(conj_file, conj_b64)

        if not (nail_bytes or palm_bytes or conj_bytes):
            return jsonify({'error': 'Please provide at least one photo (Nail, Palm, or Conjunctiva)'}), 400

        modality_visuals = {}
        nail_data = None
        palm_data = None
        conj_data = None

        if nail_bytes:
            nail_data = analyze_modality_image(nail_bytes, 'fingernail')
            modality_visuals['nail'] = {
                'original': bgr_to_base64_data_url(nail_data['original_bgr']),
                'normalized': bgr_to_base64_data_url(nail_data['normalized_bgr']),
                'masked': bgr_to_base64_data_url(nail_data['masked_bgr']),
                'features': nail_data['features']
            }

        if palm_bytes:
            palm_data = analyze_modality_image(palm_bytes, 'palm')
            modality_visuals['palm'] = {
                'original': bgr_to_base64_data_url(palm_data['original_bgr']),
                'normalized': bgr_to_base64_data_url(palm_data['normalized_bgr']),
                'masked': bgr_to_base64_data_url(palm_data['masked_bgr']),
                'features': palm_data['features']
            }

        if conj_bytes:
            conj_data = analyze_modality_image(conj_bytes, 'conjunctiva')
            modality_visuals['conjunctiva'] = {
                'original': bgr_to_base64_data_url(conj_data['original_bgr']),
                'normalized': bgr_to_base64_data_url(conj_data['normalized_bgr']),
                'masked': bgr_to_base64_data_url(conj_data['masked_bgr']),
                'features': conj_data['features']
            }

        result = predict_anemia_multimodal(
            nail_data=nail_data,
            palm_data=palm_data,
            conj_data=conj_data,
            gender=gender
        )

        result['patient_metadata'] = {
            'patient_name': patient_name,
            'gender': gender,
            'age': age,
            'screening_id': f"MRUH-HG-{np.random.randint(10000, 99999)}",
            'date': '2026-10-30',
            'institution': 'Malla Reddy University (CSE Dept)'
        }
        result['modality_visuals'] = modality_visuals

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting HemoGlow AI Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
