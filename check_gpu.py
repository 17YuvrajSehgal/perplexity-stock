#!/usr/bin/env python3
"""
GPU Verification Script
Checks if GPU is available and properly configured for training
"""

import sys

def check_gpu():
    """Check GPU availability and configuration"""
    print("="*60)
    print("GPU CONFIGURATION CHECK")
    print("="*60)
    
    # Check PyTorch installation
    try:
        import torch
        print("✓ PyTorch installed")
        print(f"  Version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch")
        return False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Test GPU tensor operations
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("\n✓ GPU tensor operations working correctly")
        except Exception as e:
            print(f"\n✗ GPU tensor operations failed: {e}")
            return False
    else:
        print("\n⚠ GPU not available")
        print("  Reasons could be:")
        print("  - No NVIDIA GPU installed")
        print("  - CUDA drivers not installed")
        print("  - PyTorch CPU-only version installed")
        print("\n  To install GPU support:")
        print("  1. Install NVIDIA drivers")
        print("  2. Install CUDA toolkit")
        print("  3. Install PyTorch with CUDA:")
        print("     Visit: https://pytorch.org/get-started/locally/")
    
    # Check Stable-Baselines3
    try:
        import stable_baselines3
        print(f"\n✓ Stable-Baselines3 installed")
        print(f"  Version: {stable_baselines3.__version__}")
    except ImportError:
        print("\n✗ Stable-Baselines3 not installed")
        print("  Install with: pip install stable-baselines3")
        return False
    
    # Check Gymnasium
    try:
        import gymnasium
        print(f"✓ Gymnasium installed")
        print(f"  Version: {gymnasium.__version__}")
    except ImportError:
        print("\n✗ Gymnasium not installed")
        print("  Install with: pip install gymnasium")
        return False
    
    print("\n" + "="*60)
    if cuda_available:
        print("✓ SYSTEM READY FOR GPU TRAINING")
    else:
        print("⚠ SYSTEM WILL USE CPU (slower but functional)")
    print("="*60)
    
    return cuda_available


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)

