"""
Check and Extract CNN Features
===============================
This script checks for existing feature files and extracts them if missing.
"""

import os
import numpy as np
import sys

def check_files():
    """Check what files exist in the models directory"""
    
    print("="*80)
    print("🔍 CHECKING EXISTING FILES")
    print("="*80)
    
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print(f"\n❌ Models directory '{models_dir}' does not exist!")
        print(f"   Please run the full training pipeline first.")
        return False
    
    print(f"\n📁 Contents of '{models_dir}':\n")
    
    all_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            all_files.append((file, size))
            print(f"  ✓ {file:40s} ({size:>8.2f} MB)")
    
    if not all_files:
        print("  ⚠ Directory is empty!")
    
    print(f"\n📊 Total files: {len(all_files)}")
    
    # Check for required files
    print("\n" + "="*80)
    print("🎯 REQUIRED FILES STATUS")
    print("="*80 + "\n")
    
    required_files = {
        'mobilenet_best.h5': 'Trained MobileNet model',
        'cnn_features_train.npy': 'Training features',
        'cnn_features_test.npy': 'Test features',
        'y_train.npy': 'Training labels',
        'y_test.npy': 'Test labels'
    }
    
    missing = []
    for filename, description in required_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename:30s} - {description:25s} ({size:.2f} MB)")
        else:
            print(f"  ✗ {filename:30s} - {description:25s} (MISSING)")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠ Missing {len(missing)} required files:")
        for f in missing:
            print(f"    - {f}")
        return False
    else:
        print(f"\n✅ All required files are present!")
        return True

def extract_features():
    """Extract CNN features from the trained MobileNet model"""
    
    print("\n" + "="*80)
    print("🔄 EXTRACTING CNN FEATURES")
    print("="*80 + "\n")
    
    # Check if MobileNet model exists
    model_path = 'models/mobilenet_best.h5'
    if not os.path.exists(model_path):
        print(f"❌ ERROR: MobileNet model not found at {model_path}")
        print("   Please run the full training pipeline first to train the model.")
        return False
    
    # Check if preprocessed data exists
    data_dir = 'preprocessed_data'
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Preprocessed data directory not found at {data_dir}")
        print("   Please run the preprocessing step first.")
        return False
    
    # Import required libraries
    try:
        from tensorflow.keras.models import load_model, Model
        print("✓ TensorFlow loaded")
    except ImportError:
        print("❌ ERROR: TensorFlow not installed!")
        print("   Install with: pip install tensorflow")
        return False
    
    # Load preprocessed data
    print(f"\n📂 Loading preprocessed data from {data_dir}...")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        print(f"  ✓ Training data: {X_train.shape}")
        print(f"  ✓ Test data: {X_test.shape}")
        print(f"  ✓ Training labels: {y_train.shape}")
        print(f"  ✓ Test labels: {y_test.shape}")
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return False
    
    # Load MobileNet model
    print(f"\n🤖 Loading MobileNet model from {model_path}...")
    try:
        model = load_model(model_path)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return False
    
    # Create feature extractor
    print("\n🔧 Creating feature extractor (layer: 'features')...")
    try:
        feature_extractor = Model(
            inputs=model.input,
            outputs=model.get_layer('features').output
        )
        print("  ✓ Feature extractor created")
    except Exception as e:
        print(f"❌ ERROR: Layer 'features' not found in model!")
        print(f"   Available layers: {[layer.name for layer in model.layers]}")
        return False
    
    # Extract training features
    print(f"\n🔄 Extracting training features from {len(X_train)} images...")
    try:
        train_features = feature_extractor.predict(X_train, batch_size=16, verbose=1)
        print(f"  ✓ Training features shape: {train_features.shape}")
    except Exception as e:
        print(f"❌ ERROR extracting training features: {e}")
        return False
    
    # Extract test features
    print(f"\n🔄 Extracting test features from {len(X_test)} images...")
    try:
        test_features = feature_extractor.predict(X_test, batch_size=16, verbose=1)
        print(f"  ✓ Test features shape: {test_features.shape}")
    except Exception as e:
        print(f"❌ ERROR extracting test features: {e}")
        return False
    
    # Save features
    print("\n💾 Saving extracted features...")
    os.makedirs('models', exist_ok=True)
    
    try:
        np.save('models/cnn_features_train.npy', train_features)
        print("  ✓ Training features saved to models/cnn_features_train.npy")
        
        np.save('models/cnn_features_test.npy', test_features)
        print("  ✓ Test features saved to models/cnn_features_test.npy")
        
        np.save('models/y_train.npy', y_train)
        print("  ✓ Training labels saved to models/y_train.npy")
        
        np.save('models/y_test.npy', y_test)
        print("  ✓ Test labels saved to models/y_test.npy")
        
        print("\n✅ Feature extraction completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR saving features: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("  CNN FEATURE EXTRACTION UTILITY")
    print("="*80 + "\n")
    
    # Check existing files
    files_ok = check_files()
    
    if files_ok:
        print("\n🎉 All required files found! You can run ml.py now.")
        print("\nRun: python ml.py")
        return
    
    # Ask user if they want to extract features
    print("\n" + "="*80)
    print("⚠️  MISSING REQUIRED FILES")
    print("="*80)
    print("\nOptions:")
    print("  1. Extract features now (if MobileNet model exists)")
    print("  2. Exit and run full training pipeline first")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        success = extract_features()
        if success:
            print("\n" + "="*80)
            print("✅ SUCCESS!")
            print("="*80)
            print("\nYou can now run: python ml.py")
        else:
            print("\n" + "="*80)
            print("❌ FEATURE EXTRACTION FAILED")
            print("="*80)
            print("\nPlease run the full training pipeline first:")
            print("  python train_models.py")
    else:
        print("\nPlease run the full training pipeline:")
        print("  python train_models.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)