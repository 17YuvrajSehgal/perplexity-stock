# GPU Setup Guide for Deep RL Trading System

This guide will help you configure GPU support for faster training.

## Prerequisites

1. **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
2. **NVIDIA GPU Drivers** installed
3. **CUDA Toolkit** (version 11.8 or 12.1 recommended)

## Step 1: Check Your GPU

### Windows
```powershell
nvidia-smi
```

### Linux/Mac
```bash
nvidia-smi
```

You should see your GPU information. If not, install NVIDIA drivers first.

## Step 2: Install CUDA Toolkit

### Windows
1. Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
2. Install the toolkit (default settings are fine)
3. Verify installation:
   ```powershell
   nvcc --version
   ```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit

# Verify
nvcc --version
```

## Step 3: Install PyTorch with CUDA Support

**IMPORTANT:** Install the correct PyTorch version for your CUDA version.

### Check Your CUDA Version
```bash
nvidia-smi
```
Look for "CUDA Version" in the top right (e.g., 11.8, 12.1)

### Install PyTorch

Visit https://pytorch.org/get-started/locally/ and select:
- Your OS
- Package: pip
- CUDA version

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (if no GPU):**
```bash
pip install torch torchvision torchaudio
```

## Step 4: Verify GPU Setup

Run the GPU check script:
```bash
python check_gpu.py
```

You should see:
```
✓ GPU detected: [Your GPU Name]
✓ CUDA version: [Version]
✓ GPU tensor operations working correctly
✓ SYSTEM READY FOR GPU TRAINING
```

## Step 5: Install Other Dependencies

```bash
pip install -r requirements.txt
```

Note: The requirements.txt file has been updated with GPU installation notes.

## Step 6: Run Training with GPU

The system will automatically detect and use GPU if available:

```bash
python main.py --ticker AAPL --total_timesteps 100000
```

To explicitly specify device:
```bash
# Use GPU
python main.py --ticker AAPL --device cuda

# Use CPU (force)
python main.py --ticker AAPL --device cpu

# Auto-detect (default)
python main.py --ticker AAPL --device auto
```

## Performance Comparison

### CPU Training
- 100k timesteps: ~30-60 minutes
- Suitable for: Testing, small models

### GPU Training
- 100k timesteps: ~5-15 minutes
- Suitable: Production training, large models
- Speedup: 3-5x faster

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size: `--batch_size 32`
- Reduce n_steps: `--n_steps 1024`
- Close other GPU applications

### Issue: "CUDA not available" but GPU is installed
**Solutions:**
1. Verify PyTorch CUDA version matches your CUDA toolkit:
   ```python
   import torch
   print(torch.version.cuda)  # Should match nvidia-smi CUDA version
   ```
2. Reinstall PyTorch with correct CUDA version
3. Restart your terminal/IDE

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Slow training even with GPU
**Solutions:**
1. Check GPU utilization: `nvidia-smi` (should show high %)
2. Ensure data is on GPU (handled automatically by Stable-Baselines3)
3. Increase batch size if memory allows
4. Check for CPU bottlenecks in data loading

## GPU Memory Management

The system automatically manages GPU memory. For large models:

1. **Monitor GPU memory:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Adjust batch size** if needed:
   ```bash
   python main.py --ticker AAPL --batch_size 32
   ```

3. **Use mixed precision** (future enhancement):
   - Can be added to PPO agent for 2x memory savings

## Multi-GPU Support

Currently, the system uses single GPU. For multi-GPU training:

1. Use `CUDA_VISIBLE_DEVICES` to select GPU:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python main.py --ticker AAPL
   ```

2. Future enhancement: Add multi-GPU support via DataParallel

## Testing GPU Performance

Run a quick test:
```bash
python main.py --ticker AAPL --total_timesteps 10000 --device cuda
```

Compare with CPU:
```bash
python main.py --ticker AAPL --total_timesteps 10000 --device cpu
```

## Additional Resources

- PyTorch CUDA Installation: https://pytorch.org/get-started/locally/
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- Stable-Baselines3 GPU Guide: https://stable-baselines3.readthedocs.io/en/master/guide/install.html

---

**Ready to train?** Run:
```bash
python check_gpu.py  # Verify setup
python main.py --ticker AAPL --total_timesteps 100000  # Start training
```

