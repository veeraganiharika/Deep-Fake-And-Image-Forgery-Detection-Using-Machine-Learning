"""
GPU Detection and Setup Script for Windows
This will help diagnose and fix GPU detection issues
"""

import sys
import os

def check_nvidia_gpu():
    """Check if NVIDIA GPU is detected by system"""
    print("\n" + "="*70)
    print("STEP 1: CHECKING NVIDIA GPU HARDWARE")
    print("="*70)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected by nvidia-smi")
            print("\nGPU Information:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi command failed")
            print("  Install NVIDIA GPU drivers from: https://www.nvidia.com/Download/index.aspx")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("  Install NVIDIA GPU drivers from: https://www.nvidia.com/Download/index.aspx")
        return False

def check_cuda():
    """Check CUDA installation"""
    print("\n" + "="*70)
    print("STEP 2: CHECKING CUDA INSTALLATION")
    print("="*70)
    
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ CUDA Toolkit installed")
            print(result.stdout)
            return True
        else:
            print("✗ CUDA not properly configured")
            return False
    except FileNotFoundError:
        print("⚠ CUDA Toolkit not found in PATH")
        print("  Options:")
        print("  1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("  2. Or use tensorflow>=2.10 which bundles CUDA")
        return False

def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print("\n" + "="*70)
    print("STEP 3: CHECKING TENSORFLOW GPU SUPPORT")
    print("="*70)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check if built with CUDA
        if tf.test.is_built_with_cuda():
            print("✓ TensorFlow is built with CUDA support")
        else:
            print("⚠ TensorFlow is NOT built with CUDA")
            print("  Install tensorflow-gpu or tensorflow>=2.10")
        
        # Check for GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"\n✓ TensorFlow detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                print(f"          {gpu.device_type}")
            
            # Try to enable memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("\n✓ GPU memory growth enabled")
            except RuntimeError as e:
                print(f"\n⚠ Memory growth error: {e}")
            
            # Test GPU computation
            print("\nTesting GPU computation...")
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                print("✓ GPU computation test PASSED")
                print(f"  Result: {c.numpy()}")
                return True
            except Exception as e:
                print(f"✗ GPU computation test FAILED: {e}")
                return False
        else:
            print("\n✗ No GPU detected by TensorFlow")
            print("\nTroubleshooting steps:")
            print("  1. Make sure NVIDIA GPU drivers are installed")
            print("  2. Install CUDA Toolkit 11.2+ or use TensorFlow 2.10+")
            print("  3. Reinstall TensorFlow:")
            print("     pip uninstall tensorflow tensorflow-gpu")
            print("     pip install tensorflow>=2.10")
            return False
            
    except ImportError:
        print("✗ TensorFlow not installed!")
        print("  Install: pip install tensorflow>=2.10")
        return False

def check_environment_variables():
    """Check CUDA environment variables"""
    print("\n" + "="*70)
    print("STEP 4: CHECKING ENVIRONMENT VARIABLES")
    print("="*70)
    
    cuda_vars = [
        'CUDA_PATH',
        'CUDA_HOME',
        'CUDNN_PATH',
        'PATH'
    ]
    
    found_any = False
    for var in cuda_vars:
        value = os.environ.get(var)
        if value and 'cuda' in value.lower():
            print(f"✓ {var}: {value}")
            found_any = True
    
    if not found_any:
        print("⚠ No CUDA environment variables found")
        print("  This might be OK if using TensorFlow 2.10+")

def get_recommendation():
    """Provide recommendations based on system"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\nFor best GPU support on Windows:")
    print("\n1. EASIEST - Use TensorFlow 2.10+ (bundles CUDA):")
    print("   pip uninstall tensorflow tensorflow-gpu")
    print("   pip install tensorflow==2.12.0")
    print("   This includes CUDA support automatically!")
    
    print("\n2. If GPU still not detected:")
    print("   - Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
    print("   - Restart your computer")
    print("   - Run this script again")
    
    print("\n3. For CPU-only training (slower but works):")
    print("   - Your code will still work, just slower")
    print("   - Reduce batch size to 8 in train_models.py")
    print("   - Expect 2-4 hours training time")

def main():
    """Run all checks"""
    print("\n" + "="*70)
    print("GPU SETUP DIAGNOSTIC - NVIDIA GeForce MX")
    print("="*70)
    
    results = {
        'nvidia': check_nvidia_gpu(),
        'cuda': check_cuda(),
        'tensorflow': check_tensorflow_gpu()
    }
    
    check_environment_variables()
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    for check, status in results.items():
        status_icon = "✓" if status else "✗"
        print(f"{status_icon} {check.upper()}: {'PASS' if status else 'FAIL'}")
    
    print("="*70)
    
    if results['tensorflow']:
        print("\n🎉 SUCCESS! GPU is working with TensorFlow!")
        print("\nYou can now run:")
        print("  python train_models.py")
        return 0
    else:
        print("\n⚠ GPU NOT WORKING")
        get_recommendation()
        
        print("\n💡 You can still train on CPU (slower):")
        print("  python train_models.py")
        print("  (Training will take 2-4 hours instead of 30-50 minutes)")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)