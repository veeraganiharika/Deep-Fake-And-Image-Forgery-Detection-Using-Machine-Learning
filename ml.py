"""
Train ML Classifiers Separately
================================
This script trains only the ML classifiers using pre-extracted CNN features.
Use this when you already have:
  - Trained MobileNet model (mobilenet_best.h5)
  - Extracted features (cnn_features_train.npy, cnn_features_test.npy)
  - Labels (y_train.npy, y_test.npy)
"""

import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the fixed hybrid classifier
from hybrid_classifier import HybridClassifier

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")

def main():
    print_header("🚀 STANDALONE ML CLASSIFIER TRAINING")
    
    # Configuration
    models_dir = 'models'
    
    print("📋 Configuration:")
    print(f"  Models directory: {models_dir}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if required files exist
    print("\n🔍 Checking for required files...")
    
    required_files = [
        os.path.join(models_dir, 'cnn_features_train.npy'),
        os.path.join(models_dir, 'cnn_features_test.npy'),
        os.path.join(models_dir, 'y_train.npy'),
        os.path.join(models_dir, 'y_test.npy')
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ Found: {file}")
        else:
            print(f"  ✗ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ ERROR: Missing required files!")
        print(f"   Please run the full training pipeline first to extract CNN features.")
        print(f"   Missing files: {missing_files}")
        sys.exit(1)
    
    # Load features and labels
    print_header("📂 LOADING DATA")
    
    print("Loading CNN features and labels...")
    X_train = np.load(os.path.join(models_dir, 'cnn_features_train.npy'))
    X_test = np.load(os.path.join(models_dir, 'cnn_features_test.npy'))
    y_train = np.load(os.path.join(models_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(models_dir, 'y_test.npy'))
    
    print(f"✓ Data loaded successfully!")
    print(f"  Training features: {X_train.shape}")
    print(f"  Test features: {X_test.shape}")
    print(f"  Training labels: {y_train.shape}")
    print(f"  Test labels: {y_test.shape}")
    
    # Display class distribution
    train_authentic = np.sum(y_train == 0)
    train_forged = np.sum(y_train == 1)
    test_authentic = np.sum(y_test == 0)
    test_forged = np.sum(y_test == 1)
    
    print(f"\n📊 Class Distribution:")
    print(f"  Training: Authentic={train_authentic}, Forged={train_forged}")
    print(f"  Test: Authentic={test_authentic}, Forged={test_forged}")
    
    # Initialize and train classifiers
    print_header("🤖 TRAINING ML CLASSIFIERS")
    
    print("Initializing Hybrid Classifier...")
    hybrid = HybridClassifier(use_scaler=True)
    
    print(f"\n📚 Training {len(hybrid.classifiers)} ML algorithms...")
    print("This may take several minutes depending on your CPU...\n")
    
    # Train all classifiers
    results = hybrid.train(X_train, y_train, X_test, y_test)
    
    # Create voting classifier
    print_header("🗳️  CREATING ENSEMBLE")
    
    print("Creating Voting Classifier from top 5 performers...")
    hybrid.create_voting_classifier(top_n=5)
    
    # Train voting classifier
    print("\nTraining Voting Classifier...")
    if hybrid.use_scaler:
        X_train_scaled = hybrid.scaler.transform(X_train)
        X_test_scaled = hybrid.scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    hybrid.voting_classifier.fit(X_train_scaled, y_train)
    
    # Evaluate voting classifier
    print("\n📊 Evaluating Voting Classifier...")
    voting_accuracy = hybrid.evaluate(X_test, y_test)
    
    # Save all models
    print_header("💾 SAVING MODELS")
    
    hybrid.save_models(models_dir)
    
    # Final summary
    print_header("✨ TRAINING COMPLETE")
    
    print("📊 Final Results Summary:")
    print(f"  Total classifiers trained: {len(hybrid.classifiers)}")
    print(f"  Voting classifier accuracy: {voting_accuracy:.4f}")
    print(f"  Models saved to: {models_dir}/")
    
    print("\n📁 Saved Files:")
    saved_files = [
        'scaler.pkl',
        'random_forest.pkl',
        'gradient_boosting.pkl',
        'adaboost.pkl',
        'logistic_regression.pkl',
        'svm.pkl',
        'knn.pkl',
        'naive_bayes.pkl',
        'decision_tree.pkl',
        'mlp.pkl',
        'voting_classifier.pkl'
    ]
    
    for file in saved_files:
        filepath = os.path.join(models_dir, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"  ✓ {file:30s} ({size:.2f} MB)")
    
    print("\n" + "="*80)
    print("🎉 SUCCESS! All ML classifiers trained and saved successfully!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Use these models for prediction on new images")
    print("  2. Deploy the voting_classifier.pkl for best results")
    print("  3. Check the confusion matrix and classification report above")
    
    return results, voting_accuracy


if __name__ == "__main__":
    try:
        results, accuracy = main()
        print(f"\n✅ Training completed successfully!")
        print(f"   Best voting classifier accuracy: {accuracy:.4f}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)