# TD-SpeakerBeam PyTorch Implementation

This is a PyTorch-only implementation of the TD-SpeakerBeam model for target speech extraction, ported from the original Asteroid-based implementation.

## Key Changes from Original
- Removed dependency on Asteroid toolkit
- Pure PyTorch implementation
- Maintained all original functionality and scripts
- Compatible with remote Linux server testing

## Requirements

To install requirements:
```
pip install -r requirements.txt
```

## Project Structure
```
td-speakerbeam-pytorch/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── datasets/          # Dataset handling
│   └── utils/             # Utility functions
├── egs/                   # Example recipes
│   └── libri2mix/         # Libri2Mix recipe
├── example/               # Example files
└── notebooks/             # Demo notebooks
```

## Running the experiments

The directory `egs` contains a recipe for [Libri2mix dataset](https://github.com/JorisCos/LibriMix). 

### Prerequisites
1. **LibriMix Dataset**: Download the LibriMix dataset from [LibriMix repository](https://github.com/JorisCos/LibriMix)
2. **Environment Setup**: Set up the Python environment and install dependencies

### Setup Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   # Edit path.sh to set your actual repository path
   export PATH_TO_REPOSITORY="/path/to/td-speakerbeam-pytorch"
   source path.sh
   ```

3. **Navigate to recipe directory:**
   ```bash
   cd egs/libri2mix
   ```

### Preparing data
```bash
bash local/prepare_data.sh <path-to-libri2mix-data>
```

**Note**: The `<path-to-libri2mix-data>` should contain `wav8k/min` subdirectories with the LibriMix dataset.

### Training SpeakerBeam
```bash
source ../../path.sh
python train.py --exp_dir exp/speakerbeam
```

### Evaluation
```bash
python eval.py --test_dir data/wav8k/min/test --task sep_noisy --model_path exp/speakerbeam/best_model.pth --out_dir exp/speakerbeam/out_best --exp_dir exp/speakerbeam --use_gpu=1
```

## Remote Testing
This implementation is designed to work on remote Linux servers. Test environment:
- Server: your_ssh_server
- Testing directory: /home/ubuntu/

## Reference
Please cite the original works when using this code:
```
@ARTICLE{Zmolikova_Spkbeam_STSP19,
  author={Žmolíková, Kateřina and Delcroix, Marc and Kinoshita, Keisuke and Ochiai, Tsubasa and Nakatani, Tomohiro and Burget, Lukáš and Černocký, Jan},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={SpeakerBeam: Speaker Aware Neural Network for Target Speaker Extraction in Speech Mixtures}, 
  year={2019},
  volume={13},
  number={4},
  pages={800-814},
  doi={10.1109/JSTSP.2019.2922820}}

@INPROCEEDINGS{delcroix_tdSpkBeam_ICASSP20,
  author={Delcroix, Marc and Ochiai, Tsubasa and Zmolikova, Katerina and Kinoshita, Keisuke and Tawara, Naohiro and Nakatani, Tomohiro and Araki, Shoko},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Improving Speaker Discrimination of Target Speech Extraction With Time-Domain Speakerbeam}, 
  year={2020},
  volume={},
  number={},
  pages={691-695},
  doi={10.1109/ICASSP40776.2020.9054683}}
```
