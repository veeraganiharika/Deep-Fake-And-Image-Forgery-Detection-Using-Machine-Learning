import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import gc
import tensorflow as tf

# Import optimized modules
from mobilenet_model import MobileNetFeatureExtractor
from hybrid_classifier import HybridClassifier

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# GPU Memory Configuration
def configure_gpu_memory():
    """Configure GPU to use only necessary memory"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Configured {len(gpus)} GPU(s) with memory growth")
        except RuntimeError as e:
            print(f"GPU configuration warning: {e}")

configure_gpu_memory()

def clear_gpu_memory():
    """Clear GPU memory cache"""
    tf.keras.backend.clear_session()
    gc.collect()
    print("✓ GPU memory cleared")

def plot_training_history(history, save_path='models/training_history.png'):
    """Plot training history with enhanced visualization"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot accuracy
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs, history.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2.5)
    ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2.5)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Accuracy', fontsize=13)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1])
    
    # Plot loss
    ax2.plot(epochs, history.history['loss'], 'b-', label='Train Loss', linewidth=2.5)
    ax2.plot(epochs, history.history['val_loss'], 'r-', label='Val Loss', linewidth=2.5)
    ax2.set_title('Model Loss Over Epochs', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('Loss', fontsize=13)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_path}")
    plt.close()

def plot_confusion_matrices(results, save_path='models/confusion_matrices.png'):
    """Plot confusion matrices for all classifiers"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    valid_results = {k: v for k, v in results.items() 
                    if v is not None and 'confusion_matrix' in v}
    
    if not valid_results:
        print("No confusion matrices to plot")
        return
    
    n_models = len(valid_results)
    cols = 4
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (name, result) in enumerate(valid_results.items()):
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'],
                   ax=axes[idx], cbar=True, annot_kws={'size': 12})
        
        axes[idx].set_title(f'{name}\nAcc: {result["accuracy"]:.4f} | F1: {result["f1_score"]:.4f}', 
                           fontweight='bold', fontsize=11)
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to {save_path}")
    plt.close()

def plot_model_comparison(results, cnn_accuracy, save_path='models/model_comparison.png'):
    """Plot comprehensive comparison of all models"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare data
    models = ['MobileNet CNN']
    accuracies = [cnn_accuracy]
    f1_scores = [0]  # CNN doesn't have F1 in this context
    
    for name, result in results.items():
        if result is not None and 'accuracy' in result:
            models.append(name)
            accuracies.append(result['accuracy'])
            f1_scores.append(result.get('f1_score', 0))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Accuracy comparison
    colors = ['#2ecc71' if acc == max(accuracies) else '#3498db' for acc in accuracies]
    bars = ax1.barh(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(acc + 0.005, i, f'{acc:.4f}', 
               va='center', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=15, fontweight='bold')
    ax1.set_xlim([0, 1.05])
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # F1-Score comparison (excluding CNN)
    f1_models = models[1:]  # Skip CNN
    f1_vals = f1_scores[1:]
    colors_f1 = ['#e74c3c' if f1 == max(f1_vals) else '#e67e22' for f1 in f1_vals]
    bars2 = ax2.barh(f1_models, f1_vals, color=colors_f1, edgecolor='black', linewidth=1.5)
    
    for i, (bar, f1) in enumerate(zip(bars2, f1_vals)):
        ax2.text(f1 + 0.005, i, f'{f1:.4f}', 
               va='center', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('F1-Score', fontsize=13, fontweight='bold')
    ax2.set_title('ML Classifier F1-Score Comparison', fontsize=15, fontweight='bold')
    ax2.set_xlim([0, 1.05])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Model comparison saved to {save_path}")
    plt.close()

def save_results_summary(cnn_results, ml_results, save_path='models/results_summary.txt'):
    """Save comprehensive results with all metrics"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IMAGE FORGERY DETECTION - OPTIMIZED MODEL EVALUATION\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU Optimized Training with Mixed Precision (FP16)\n")
        f.write("="*80 + "\n\n")
        
        # CNN Results
        f.write("="*80 + "\n")
        f.write("DEEP LEARNING MODEL (MobileNetV2 - Optimized)\n")
        f.write("="*80 + "\n")
        f.write(f"Test Accuracy: {cnn_results['accuracy']:.4f}\n")
        f.write(f"Test Loss: {cnn_results['loss']:.4f}\n")
        f.write(f"Architecture: MobileNetV2 (alpha=0.75) + Custom Head\n")
        f.write(f"Training: Mixed Precision (FP16), Early Stopping, LR Scheduling\n\n")
        
        # ML Classifier Results
        f.write("="*80 + "\n")
        f.write("MACHINE LEARNING CLASSIFIERS (CNN Feature-Based)\n")
        f.write("="*80 + "\n\n")
        
        # Sort by accuracy
        sorted_results = sorted(
            [(name, result) for name, result in ml_results.items() 
             if result is not None and 'accuracy' in result],
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            f.write(f"#{rank} {name}:\n")
            f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result.get('precision', 0):.4f}\n")
            f.write(f"  Recall:    {result.get('recall', 0):.4f}\n")
            f.write(f"  F1-Score:  {result.get('f1_score', 0):.4f}\n")
            f.write(f"  AUC-ROC:   {result.get('auc', 0):.4f}\n")
            f.write(f"\n{result['report']}\n")
            f.write("-"*80 + "\n\n")
        
        # Best Model Summary
        best_model = sorted_results[0] if sorted_results else None
        if best_model:
            f.write("="*80 + "\n")
            f.write("🏆 BEST PERFORMING MODEL\n")
            f.write("="*80 + "\n")
            f.write(f"Model: {best_model[0]}\n")
            f.write(f"Accuracy:  {best_model[1]['accuracy']:.4f}\n")
            f.write(f"Precision: {best_model[1].get('precision', 0):.4f}\n")
            f.write(f"Recall:    {best_model[1].get('recall', 0):.4f}\n")
            f.write(f"F1-Score:  {best_model[1].get('f1_score', 0):.4f}\n")
            f.write(f"AUC-ROC:   {best_model[1].get('auc', 0):.4f}\n")
            f.write("="*80 + "\n")
    
    print(f"✓ Results summary saved to {save_path}")

def save_training_config(config, save_path='models/training_config.json'):
    """Save training configuration"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Training config saved to {save_path}")

def load_preprocessed_data(preprocessed_dir):
    """Load preprocessed data efficiently"""
    print(f"\nLoading preprocessed data from: {preprocessed_dir}")
    
    if not os.path.exists(preprocessed_dir):
        raise FileNotFoundError(f"Directory not found: {preprocessed_dir}")
    
    # File paths
    paths = {
        'X_train': os.path.join(preprocessed_dir, 'X_train.npy'),
        'X_test': os.path.join(preprocessed_dir, 'X_test.npy'),
        'y_train': os.path.join(preprocessed_dir, 'y_train.npy'),
        'y_test': os.path.join(preprocessed_dir, 'y_test.npy')
    }
    
    # Verify all files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Load data with memory mapping for large files
    X_train = np.load(paths['X_train'], mmap_mode='r')
    X_test = np.load(paths['X_test'], mmap_mode='r')
    y_train = np.load(paths['y_train'])
    y_test = np.load(paths['y_test'])
    
    # Load into memory
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    print(f"✓ Data loaded successfully")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Image shape: {X_train.shape[1:]}")
    print(f"  Train: Authentic={np.sum(y_train==0)}, Forged={np.sum(y_train==1)}")
    print(f"  Test: Authentic={np.sum(y_test==0)}, Forged={np.sum(y_test==1)}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main training pipeline - OPTIMIZED FOR YOUR GPU"""
    
    # ========================================================================
    # CONFIGURATION - OPTIMIZED FOR 4-6GB GPU
    # ========================================================================
    config = {
        'preprocessed_dir': 'preprocessed_data',
        'models_dir': 'models',
        'resume_training': False,  # Set to True to resume from saved models
        
        # CNN Training (Optimized for limited GPU memory)
        'cnn_epochs': 30,  # Reduced epochs with early stopping
        'cnn_batch_size': 16,  # Reduced for 4GB GPU
        'use_data_augmentation': True,  # Better generalization
        
        # Fine-tuning (Optional)
        'enable_fine_tuning': False,  # Set True for even better accuracy
        'fine_tune_epochs': 10,
        'fine_tune_batch_size': 12,
        
        # Feature extraction
        'feature_batch_size': 16,
        
        # ML Training
        'use_cross_validation': False  # Set True for CV (slower but better)
    }
    
    save_training_config(config, os.path.join(config['models_dir'], 'training_config.json'))
    
    print("\n" + "="*80)
    print("🚀 OPTIMIZED IMAGE FORGERY DETECTION TRAINING PIPELINE")
    print("="*80)
    print(f"GPU Memory: 4-6GB Optimized")
    print(f"Mixed Precision: FP16 Enabled")
    print(f"Batch Size: {config['cnn_batch_size']} (Memory Optimized)")
    print(f"Data Augmentation: {'Enabled' if config['use_data_augmentation'] else 'Disabled'}")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING PREPROCESSED DATA")
    print("="*80)
    
    try:
        X_train, X_test, y_train, y_test = load_preprocessed_data(config['preprocessed_dir'])
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nRequired files in preprocessed_data/:")
        print("  - X_train.npy, X_test.npy, y_train.npy, y_test.npy")
        return
    
    # ========================================================================
    # STEP 2: TRAIN MOBILENET CNN
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: TRAINING MOBILENET CNN (GPU OPTIMIZED)")
    print("="*80)
    
    clear_gpu_memory()  # Clear before training
    
    mobilenet = MobileNetFeatureExtractor(input_shape=X_train.shape[1:])
    model_path = os.path.join(config['models_dir'], 'mobilenet_optimized.h5')
    
    if os.path.exists(model_path) and config['resume_training']:
        print("\n✓ Loading pre-trained model...")
        try:
            mobilenet.load_model(model_path)
        except Exception as e:
            print(f"✗ Loading failed: {e}")
            print("Training new model...")
            mobilenet.build_model()
    else:
        print("\n🔨 Building and training new model...")
        mobilenet.build_model()
        mobilenet.get_model_summary()
        
        # Train with optimized settings
        history = mobilenet.train(
            X_train, y_train,
            X_test, y_test,
            epochs=config['cnn_epochs'],
            batch_size=config['cnn_batch_size'],
            verbose=1,
            use_augmentation=config['use_data_augmentation']
        )
        
        # Save model and history
        mobilenet.save_model(model_path)
        plot_training_history(history, 
                             os.path.join(config['models_dir'], 'training_history.png'))
        
        # Optional fine-tuning for maximum accuracy
        if config['enable_fine_tuning']:
            print("\n🎯 Starting fine-tuning for better accuracy...")
            fine_tune_history = mobilenet.fine_tune(
                X_train, y_train,
                X_test, y_test,
                trainable_layers=30,
                epochs=config['fine_tune_epochs'],
                batch_size=config['fine_tune_batch_size']
            )
            mobilenet.save_model(os.path.join(config['models_dir'], 'mobilenet_finetuned.h5'))
    
    # Evaluate CNN
    print("\n📊 Evaluating MobileNet CNN...")
    cnn_loss, cnn_accuracy = mobilenet.evaluate(X_test, y_test, 
                                                batch_size=config['cnn_batch_size'])
    cnn_results = {'loss': cnn_loss, 'accuracy': cnn_accuracy}
    
    # ========================================================================
    # STEP 3: EXTRACT FEATURES
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING CNN FEATURES FOR ML CLASSIFIERS")
    print("="*80)
    
    features_dir = config['models_dir']
    features_train_path = os.path.join(features_dir, 'features_train.npy')
    features_test_path = os.path.join(features_dir, 'features_test.npy')
    
    if os.path.exists(features_train_path) and config['resume_training']:
        print("\n✓ Loading pre-extracted features...")
        features_train = np.load(features_train_path)
        features_test = np.load(features_test_path)
    else:
        print("\n🔍 Extracting features...")
        features_train = mobilenet.extract_features(X_train, 
                                                    batch_size=config['feature_batch_size'])
        features_test = mobilenet.extract_features(X_test, 
                                                   batch_size=config['feature_batch_size'])
        
        # Save features
        np.save(features_train_path, features_train)
        np.save(features_test_path, features_test)
        print(f"✓ Features saved to {features_dir}/")
    
    print(f"  Training features: {features_train.shape}")
    print(f"  Test features: {features_test.shape}")
    
    # Clear CNN from memory to free GPU
    clear_gpu_memory()
    del mobilenet, X_train, X_test
    gc.collect()
    
    # ========================================================================
    # STEP 4: TRAIN ML CLASSIFIERS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: TRAINING OPTIMIZED ML CLASSIFIERS (10 ALGORITHMS)")
    print("="*80)
    
    hybrid = HybridClassifier(use_scaler=True)
    
    voting_clf_path = os.path.join(config['models_dir'], 'voting_classifier.pkl')
    
    if os.path.exists(voting_clf_path) and config['resume_training']:
        print("\n✓ Loading pre-trained ML models...")
        hybrid.load_models(config['models_dir'])
    else:
        print("\n🤖 Training ML classifiers...")
        
        # Train individual classifiers
        hybrid.train_individual_classifiers(features_train, y_train,
                                           use_cv=config['use_cross_validation'])
        
        # Train ensemble
        hybrid.train_voting_classifier(features_train, y_train)
        
        # Save all models
        hybrid.save_models(config['models_dir'])
    
    # ========================================================================
    # STEP 5: EVALUATE ALL MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    ml_results = hybrid.evaluate_classifiers(features_test, y_test, detailed=True)
    
    # ========================================================================
    # STEP 6: SAVE RESULTS & VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: GENERATING REPORTS AND VISUALIZATIONS")
    print("="*80)
    
    save_results_summary(cnn_results, ml_results, 
                        os.path.join(config['models_dir'], 'results_summary.txt'))
    
    plot_confusion_matrices(ml_results,
                           os.path.join(config['models_dir'], 'confusion_matrices.png'))
    
    plot_model_comparison(ml_results, cnn_accuracy,
                         os.path.join(config['models_dir'], 'model_comparison.png'))
    
    # Get best model
    best_model, best_accuracy = hybrid.get_best_model(ml_results)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Performance Summary:")
    print(f"  ├─ MobileNet CNN:")
    print(f"  │  ├─ Accuracy: {cnn_accuracy:.4f}")
    print(f"  │  └─ Loss: {cnn_loss:.4f}")
    print(f"  │")
    print(f"  └─ Best ML Classifier: {best_model}")
    print(f"     └─ Accuracy: {best_accuracy:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  ├─ {config['models_dir']}/mobilenet_optimized.h5")
    print(f"  ├─ {config['models_dir']}/voting_classifier.pkl")
    print(f"  ├─ {config['models_dir']}/features_train.npy")
    print(f"  ├─ {config['models_dir']}/features_test.npy")
    print(f"  ├─ {config['models_dir']}/training_history.png")
    print(f"  ├─ {config['models_dir']}/confusion_matrices.png")
    print(f"  ├─ {config['models_dir']}/model_comparison.png")
    print(f"  └─ {config['models_dir']}/results_summary.txt")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Review results_summary.txt for detailed metrics")
    print(f"  2. Check confusion matrices for error analysis")
    print(f"  3. Enable fine-tuning for even better accuracy")
    print(f"  4. Use the models for inference/deployment")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
