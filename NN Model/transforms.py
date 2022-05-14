import torch
import numpy as np

from sklearn.preprocessing import minmax_scale
from torch import Tensor
from scipy.signal import butter, filtfilt

class AddGaussianNoiseInPersents(object):
    def __init__(self, persent: float, mean: float=0.0) -> None:
        super().__init__()
        self.persent = persent / 100.0
        self.mean = mean
    
    def __call__(self, tensor: Tensor) -> Tensor:
        sigma = tensor.std() * self.persent    
        return tensor + torch.randn(tensor.size()) * sigma + self.mean
    
class Normalize(object):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, tensor: Tensor) -> Tensor:  
        return tensor / tensor.sum()
    
    
class CenterSignal(object):
    def __init__(self) -> None:
        super().__init__()
       
    def __call__(self, tensor: Tensor) -> Tensor:
        centralized = torch.zeros_like(tensor) 
        signal =  tensor[tensor > 0.0]
        offset = centralized.size(0) - signal.size(0)
        centralized[offset // 2: offset // 2 + signal.size(0)] = signal
        
        return centralized
    
class SmoothSignal(object):
    def __init__(self, filter_order: int, critical_freq: float=8e-3) -> None:
        super().__init__()
        self.critical_freq = critical_freq
        self.filter_order = filter_order
        self.std: float = 0.00367
        self.b, self.a = butter(self.filter_order, self.critical_freq)
       
    def __call__(self, tensor: Tensor) -> Tensor:
        to_numpy: np.ndarray = tensor.numpy()
        smoothed: np.ndarray = filtfilt(self.b, self.a, to_numpy)
        
        # some torch issues
        smoothed = smoothed.copy()
        # smoothed -= 2.0 * self.std
        return torch.tensor(smoothed, dtype=tensor.dtype)
    
class IntensityFilter(object):
    def __init__(self, critical_intensity: float, substruct_min: bool) -> None:
        self.critical_intensity = critical_intensity
        self.substruct_min = substruct_min
    
    def __call__(self, tensor: Tensor) -> Tensor:
        mask = tensor > self.critical_intensity
        leftmost_value = tensor[mask][0]
        rightmost_value = tensor[mask][-1]
        value_to_fill = leftmost_value if leftmost_value < rightmost_value else rightmost_value
        
        filtered = torch.full_like(tensor, value_to_fill)
        filtered[mask] = tensor[mask]
        
        # TODO: substract minimum!
        if self.substruct_min:
            filtered -= filtered.min()
        
        return filtered
    
class ScaleSignal(object):
    
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, tensor: Tensor) -> Tensor:
        return torch.tensor(minmax_scale(tensor), dtype=torch.float)