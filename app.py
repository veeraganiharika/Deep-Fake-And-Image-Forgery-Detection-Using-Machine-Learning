from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from pathlib import Path
import json

from data_preprocessing import DataPreprocessor
from mobilenet_model import MobileNetFeatureExtractor
from hybrid_classifier import HybridClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif'}

# Global models
mobilenet_model = None
hybrid_classifier = None
preprocessor = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load all trained models"""
    global mobilenet_model, hybrid_classifier, preprocessor
    
    print("Loading models...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    print("✓ Preprocessor initialized")
    
    # Load MobileNet - FIXED PATH
    mobilenet_path = 'models/mobilenet_best.h5'  # Changed from mobilenet_forgery_detection.h5
    if os.path.exists(mobilenet_path):
        try:
            mobilenet_model = MobileNetFeatureExtractor()
            mobilenet_model.load_model(mobilenet_path)
            print("✓ MobileNet loaded successfully")
        except Exception as e:
            print(f"✗ Error loading MobileNet: {e}")
            mobilenet_model = None
    else:
        print(f"✗ MobileNet model not found at {mobilenet_path}")
        mobilenet_model = None
    
    # Load hybrid classifiers
    voting_path = 'models/voting_classifier.pkl'
    if os.path.exists(voting_path):
        try:
            hybrid_classifier = HybridClassifier()
            hybrid_classifier.load_models('models')
            print("✓ Hybrid classifiers loaded successfully")
        except Exception as e:
            print(f"✗ Error loading Hybrid classifiers: {e}")
            hybrid_classifier = None
    else:
        print(f"✗ Hybrid classifiers not found at {voting_path}")
        hybrid_classifier = None
    
    # Check if all models loaded successfully
    if mobilenet_model is None or hybrid_classifier is None:
        print("\n" + "="*60)
        print("WARNING: Some models failed to load!")
        print("Please ensure you have trained the models first by running:")
        print("  python train_models.py")
        print("="*60 + "\n")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/detect')
def detect():
    """Detection page"""
    return render_template('detect.html')

@app.route('/results')
def results():
    """Results page"""
    return render_template('results.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
    
    # Check if models are loaded
    if mobilenet_model is None or hybrid_classifier is None:
        return jsonify({
            'error': 'Models not loaded. Please train the models first using train_models.py'
        }), 500
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img = preprocessor.extract_ela_features(filepath)
        if img is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
        
        # Get MobileNet prediction
        mobilenet_pred = mobilenet_model.predict(img)
        mobilenet_class = int(np.argmax(mobilenet_pred[0]))
        mobilenet_confidence = float(mobilenet_pred[0][mobilenet_class]) * 100
        
        # Extract features for hybrid classifiers
        features = mobilenet_model.extract_features(img)
        
        # Get predictions from all classifiers
        all_predictions = hybrid_classifier.get_all_predictions(features)
        
        # Prepare results
        results = {
            'filename': filename,
            'filepath': filepath,
            'mobilenet': {
                'prediction': 'Forged' if mobilenet_class == 1 else 'Authentic',
                'confidence': mobilenet_confidence
            },
            'classifiers': {}
        }
        
        for name, pred_data in all_predictions.items():
            if 'error' in pred_data:
                continue
                
            prediction = int(pred_data['prediction'])
            probability = pred_data['probability']
            
            results['classifiers'][name] = {
                'prediction': 'Forged' if prediction == 1 else 'Authentic',
                'confidence': float(probability[prediction]) * 100,
                'authentic_prob': float(probability[0]) * 100,
                'forged_prob': float(probability[1]) * 100
            }
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    info = {
        'models': [
            {
                'name': 'MobileNet CNN',
                'type': 'Deep Learning',
                'description': 'Lightweight convolutional neural network',
                'loaded': mobilenet_model is not None
            },
            {
                'name': 'Support Vector Machine',
                'type': 'Machine Learning',
                'description': 'SVM with RBF kernel',
                'loaded': hybrid_classifier is not None
            },
            {
                'name': 'Logistic Regression',
                'type': 'Machine Learning',
                'description': 'Linear classification model',
                'loaded': hybrid_classifier is not None
            },
            {
                'name': 'Decision Tree',
                'type': 'Machine Learning',
                'description': 'Tree-based classifier',
                'loaded': hybrid_classifier is not None
            },
            {
                'name': 'K-Nearest Neighbors',
                'type': 'Machine Learning',
                'description': 'Instance-based learning',
                'loaded': hybrid_classifier is not None
            },
            {
                'name': 'Random Forest',
                'type': 'Machine Learning',
                'description': 'Ensemble of decision trees',
                'loaded': hybrid_classifier is not None
            },
            {
                'name': 'Voting Classifier',
                'type': 'Ensemble',
                'description': 'Combines all classifiers',
                'loaded': hybrid_classifier is not None
            }
        ],
        'all_loaded': mobilenet_model is not None and hybrid_classifier is not None
    }
    return jsonify(info)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'models_loaded': mobilenet_model is not None and hybrid_classifier is not None
    })

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load models
    load_models()
    
    # Run app
    print("\n" + "="*60)
    print("IMAGE FORGERY DETECTION SYSTEM")
    print("="*60)
    print("\nServer starting at http://127.0.0.1:5000")
    print("Press CTRL+C to quit\n")
    
    if mobilenet_model is not None and hybrid_classifier is not None:
        print("✓ All models loaded successfully!")
        print("Ready to accept requests.\n")
    else:
        print("⚠ WARNING: Models not loaded!")
        print("Run 'python train_models.py' first to train models.\n")
    
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
