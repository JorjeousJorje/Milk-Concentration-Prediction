import torch
import torch.nn as nn

from abc import abstractmethod
from torch import Tensor

class AbstractRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    @abstractmethod   
    def forward(self, x: Tensor) -> Tensor:
        pass
      
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            self.eval()
            return self.forward(x)
        
class BasicBlock(nn.Module):
    
    def __init__(self, basic_block: nn.Sequential):
        super().__init__()
        
        assert isinstance(basic_block, nn.Sequential)
        self.basic_block = basic_block
        
    def __getitem__(self, index: int):
        return self.basic_block[index]
    
    def forward(self, x: Tensor) -> Tensor:
        return self.basic_block(x)