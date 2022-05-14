import torch.nn as nn
import torch

from random import randint
from torch import Tensor
from numpy import ndarray



from AbstractRegressionModel import AbstractRegressionModel, BasicBlock

class LinearBlock(BasicBlock):
    def __init__(self, in_size, out_size):
        basic_block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(num_features=out_size),
            nn.ReLU(inplace=True)
        )
        super().__init__(basic_block)
        
class LinearRegressor(AbstractRegressionModel):
    def __init__(self, input_features, predictions_count):
        super().__init__()
        
        self.predictions_count = predictions_count
        self.lin_blocks = nn.Sequential(
                LinearBlock(input_features, 256),
                LinearBlock(256, 128),
                LinearBlock(128, 64),
                LinearBlock(64, 32),
            )
        self.set_input_features = True
        
        self.prediction = nn.Sequential(
            nn.Linear(1, self.predictions_count),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor):
        device = self.dummy_param.device
        x = x.squeeze()
        x = x.to(device, dtype=torch.float)
        x = self.lin_blocks(x)
        
        if self.set_input_features:
            self.set_input_features = False
            self.prediction[0] = nn.Linear(x.shape[1], self.predictions_count).to(device)
            
        x = self.prediction(x)
        return x