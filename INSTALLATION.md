# Installation and Setup Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   # Edit path.sh to set PATH_TO_REPOSITORY to your actual path
   export PATH_TO_REPOSITORY="/path/to/td-speakerbeam-pytorch"
   source path.sh
   ```

3. **Test installation:**
   ```bash
   python test_installation.py
   ```

## Remote Server Setup

For testing on the remote Linux server (ssh.dingshiqi.me):

1. **Connect to server:**
   ```bash
   ssh shiqi@ssh.dingshiqi.me
   # Password: Ding090508
   ```

2. **Navigate to testing directory:**
   ```bash
   cd /home/ubuntu/
   ```

3. **Clone/copy the project:**
   ```bash
   # Copy your td-speakerbeam-pytorch directory here
   ```

4. **Install dependencies:**
   ```bash
   pip install -r td-speakerbeam-pytorch/requirements.txt
   ```

5. **Set up environment:**
   ```bash
   cd td-speakerbeam-pytorch
   export PATH_TO_REPOSITORY="/home/ubuntu/td-speakerbeam-pytorch"
   export PYTHONPATH=${PATH_TO_REPOSITORY}/src:$PYTHONPATH
   ```

6. **Test installation:**
   ```bash
   python test_installation.py
   ```

## Training and Evaluation

1. **Prepare data:**
   ```bash
   cd egs/libri2mix
   bash local/prepare_data.sh /path/to/libri2mix/data
   ```

2. **Train model:**
   ```bash
   python train.py --exp_dir exp/speakerbeam
   ```

3. **Evaluate model:**
   ```bash
   python eval.py --test_dir data/wav8k/min/test --task sep_noisy \
                  --model_path exp/speakerbeam/best_model.pth \
                  --out_dir exp/speakerbeam/out_best \
                  --exp_dir exp/speakerbeam --use_gpu=1
   ```

## Key Differences from Original

- **No Asteroid dependency:** Pure PyTorch implementation
- **Self-contained:** All required components implemented from scratch
- **Compatible:** Maintains same API and functionality
- **Portable:** Works on remote servers without complex dependencies

## Troubleshooting

### Common Issues

1. **Missing map_mixture2enrollment files**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'data/wav8k/min/test/map_mixture2enrollment'
   ```
   **Solution**: These files are now included in the project. They should be located at:
   - `egs/libri2mix/data/wav8k/min/test/map_mixture2enrollment`
   - `egs/libri2mix/data/wav8k/min/dev/map_mixture2enrollment`

2. **Import errors**
   ```
   ModuleNotFoundError: No module named 'models'
   ```
   **Solution**: Ensure PYTHONPATH is set correctly:
   ```bash
   export PYTHONPATH=/path/to/td-speakerbeam-pytorch/src:$PYTHONPATH
   ```

3. **CUDA errors**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use CPU-only mode or reduce batch size:
   ```bash
   python train.py --exp_dir exp/test --use_gpu=0
   # or
   python train.py --exp_dir exp/test --batch_size=2
   ```

4. **Permission issues**
   ```
   Permission denied: './local/prepare_data.sh'
   ```
   **Solution**: Make scripts executable:
   ```bash
   chmod +x local/prepare_data.sh
   chmod +x ../../path.sh
   ```

5. **LibriMix data path issues**
   ```
   FileNotFoundError: CSV file not found
   ```
   **Solution**: Ensure your LibriMix data path contains the correct structure:
   ```
   /path/to/libri2mix/
   ├── wav8k/min/
   │   ├── train-100/
   │   ├── train-360/
   │   ├── dev/
   │   └── test/
   ```

### Testing the Installation

Run the test script to verify everything is working:
```bash
python test_installation.py
```

This will check:
- All imports work correctly
- Model can be created
- Forward pass works
- Loss computation works
- Metrics computation works