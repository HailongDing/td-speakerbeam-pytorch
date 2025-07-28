# Project Structure

```
td-speakerbeam-pytorch/
├── README.md                           # Main project documentation
├── INSTALLATION.md                     # Installation and setup guide
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                    # Python dependencies
├── path.sh                            # Environment setup script
├── test_installation.py               # Installation test script
│
├── src/                               # Source code
│   ├── models/                        # Model implementations
│   │   ├── __init__.py
│   │   ├── base_models.py             # Base encoder-masker-decoder classes
│   │   ├── convolutional.py           # TCN and convolutional layers
│   │   ├── adapt_layers.py            # Speaker adaptation layers
│   │   ├── td_speakerbeam.py          # Main TD-SpeakerBeam model
│   │   └── system.py                  # PyTorch Lightning system
│   │
│   ├── datasets/                      # Dataset handling
│   │   ├── __init__.py
│   │   ├── librimix.py                # Base LibriMix dataset
│   │   └── librimix_informed.py       # LibriMix with enrollment
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── torch_utils.py             # PyTorch utilities
│       ├── filterbanks.py             # Encoder/decoder filterbanks
│       ├── losses.py                  # Loss functions (SI-SDR)
│       ├── metrics.py                 # Evaluation metrics
│       ├── optimizers.py              # Optimizer utilities
│       └── parser_utils.py            # Argument parsing
│
├── egs/                               # Example recipes
│   └── libri2mix/                     # LibriMix recipe
│       ├── train.py                   # Training script
│       ├── eval.py                    # Evaluation script
│       └── local/                     # Local utilities
│           ├── conf.yml               # Configuration file
│           ├── prepare_data.sh        # Data preparation script
│           ├── create_local_metadata.py
│           ├── create_enrollment_csv_all.py
│           └── create_enrollment_csv_fixed.py
│
├── example/                           # Example files
│   └── conf.yml                       # Example configuration
│
└── notebooks/                         # Jupyter notebooks
    └── SpeakerBeam_demo_notebook.ipynb # Demo notebook
```

## Key Components

### Models (`src/models/`)
- **td_speakerbeam.py**: Main TD-SpeakerBeam model implementation
- **convolutional.py**: Temporal Convolutional Network (TCN) layers
- **adapt_layers.py**: Speaker adaptation mechanisms
- **base_models.py**: Base classes for encoder-masker-decoder architecture
- **system.py**: PyTorch Lightning training system

### Utilities (`src/utils/`)
- **filterbanks.py**: Learnable encoder/decoder filterbanks (replaces asteroid_filterbanks)
- **losses.py**: SI-SDR loss function (replaces asteroid.losses)
- **metrics.py**: Evaluation metrics (replaces asteroid.metrics)
- **torch_utils.py**: PyTorch utilities (replaces asteroid.utils)

### Datasets (`src/datasets/`)
- **librimix.py**: Base LibriMix dataset loader
- **librimix_informed.py**: LibriMix with speaker enrollment information

### Training Recipe (`egs/libri2mix/`)
- **train.py**: Main training script
- **eval.py**: Evaluation script
- **local/**: Data preparation and configuration files

## Removed Dependencies

The following Asteroid components have been replaced with pure PyTorch implementations:

- `asteroid.engine.System` → `models.system.SystemInformed`
- `asteroid_filterbanks.make_enc_dec` → `utils.filterbanks.make_enc_dec`
- `asteroid.masknn.convolutional.TDConvNet` → `models.convolutional.TDConvNet`
- `asteroid.losses.singlesrc_neg_sisdr` → `utils.losses.singlesrc_neg_sisdr`
- `asteroid.metrics.get_metrics` → `utils.metrics.get_metrics`
- `asteroid.utils.torch_utils` → `utils.torch_utils`
- `asteroid.engine.optimizers` → `utils.optimizers`
- `asteroid.data.LibriMix` → `datasets.librimix.LibriMix`

## Usage

1. **Setup environment**: `source path.sh`
2. **Test installation**: `python test_installation.py`
3. **Prepare data**: `bash egs/libri2mix/local/prepare_data.sh /path/to/data`
4. **Train model**: `python egs/libri2mix/train.py --exp_dir exp/test`
5. **Evaluate model**: `python egs/libri2mix/eval.py --model_path exp/test/best_model.pth --test_dir data/test --task sep_noisy --out_dir exp/test/results`