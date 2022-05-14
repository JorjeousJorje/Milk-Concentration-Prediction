import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data import Dataset

class MilkSignalDataset(Dataset):
    def __init__(self, file_path: str, transforms = None, signals_label: str = '0', marked: bool=True, compression: str=None):
        # compression may be "zip"
        self.df: pd.DataFrame = pd.read_csv(file_path, compression=compression)
        
        self.signals: pd.DataFrame = self.df.loc[:, signals_label:]
        self.features: pd.DataFrame = self.df.loc[:, :signals_label]
        self.transforms = transforms
        self.marked = marked
        self.use_transforms = True
        
    def __len__(self):
        return len(self.signals.values)

    def signal_len(self):
            return len(self.signals.values[0])
    
    def __getitem__(self, index) -> tuple[Tensor, float]:        
        signal: np.ndarray = self.signals.values[index]
        signal: Tensor = torch.tensor(signal, dtype=torch.float32)
        
        if self.use_transforms and self.transforms is not None:
            signal = self.transforms(signal)
        
        if self.marked:
            label = self.features["conc"][index]
            label: Tensor = torch.tensor(label, dtype=torch.float32)
            return  signal, label.item()
    
        return signal
    
    def visualize_samples(self, indices, count=10, title=None, use_transform=True):
        # visualize random 10 samples
        plt.figure(figsize=(count*3,3))
        
        display_indices = indices[:count]
        self.use_transforms = use_transform
        if title:
            plt.suptitle(f"{title} {len(display_indices)}/{len(indices)}")
                    
        for i, index in enumerate(display_indices):
            plt.subplot(1, count, i + 1)
            x = None
            
            if self.marked:    
                x, numPeaks = self.__getitem__(index)
                plt.title(f"milk conc: {numPeaks:.2f}, max: {x.max():.2f}")
            else:
                x = self.__getitem__(index)
            
            plt.plot(x.squeeze())
            plt.grid(False)
        
        self.use_transforms = True