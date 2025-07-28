# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import os
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset


class LibriMix(Dataset):
    """LibriMix dataset for speech separation."""
    
    def __init__(self, csv_dir, task="sep_clean", sample_rate=16000, n_src=2, segment=3):
        self.csv_dir = csv_dir
        self.task = task
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.segment = segment
        
        # Load CSV file
        csv_path = os.path.join(csv_dir, f"mixture_{task}_mix_both.csv")
        if not os.path.exists(csv_path):
            # Try alternative naming
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and 'mixture' in f]
            if csv_files:
                csv_path = os.path.join(csv_dir, csv_files[0])
            else:
                raise FileNotFoundError(f"No CSV file found in {csv_dir}")
                
        self.df = pd.read_csv(csv_path)
        
        if segment is not None:
            self.seg_len = int(segment * sample_rate)
            # Filter out short utterances
            self.df = self.df[self.df['length'] >= self.seg_len]
        else:
            self.seg_len = None
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Read mixture
        mixture_path = row['mixture_path']
        if self.seg_len is not None:
            start = torch.randint(0, int(row['length']) - self.seg_len + 1, (1,)).item()
            mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=start + self.seg_len)
        else:
            mixture, _ = sf.read(mixture_path, dtype="float32")
        mixture = torch.from_numpy(mixture)
        
        # Read sources
        sources = []
        for i in range(self.n_src):
            source_path = row[f'source_{i+1}_path']
            if self.seg_len is not None:
                source, _ = sf.read(source_path, dtype="float32", start=start, stop=start + self.seg_len)
            else:
                source, _ = sf.read(source_path, dtype="float32")
            sources.append(torch.from_numpy(source))
        
        sources = torch.stack(sources)
        return mixture, sources
    
    def get_infos(self):
        """Get dataset information."""
        return {
            'dataset': 'LibriMix',
            'task': self.task,
            'sample_rate': self.sample_rate,
            'n_src': self.n_src,
            'segment': self.segment
        }