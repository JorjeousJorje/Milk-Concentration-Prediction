import torch.nn as nn
import torch

from random import randint
from torch import Tensor
from numpy import ndarray



from AbstractRegressionModel import AbstractRegressionModel, BasicBlock

class ConvBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=2, dilation=1, stride=1, add_bn=True):
        if add_bn:
            basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride),
                nn.ReLU(inplace=True)
            )
        super().__init__(basic_block)
        
class ConvRegressor(AbstractRegressionModel):
    def __init__(self, in_channels, predictions_count):
        super().__init__()
        
        self.predictions_count = predictions_count
        self.conv_blocks = nn.Sequential(
                ConvBlock(in_channels, 64),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(64, 64),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(64, 64),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(64, 64),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(64, 64),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(64, 64),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(64, 128),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(128, 256),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(256, 512),
                nn.MaxPool1d(kernel_size=2),
                ConvBlock(512, 512),

                nn.Flatten(),
                nn.Dropout(p=0.5)
            )
        self.set_input_features = True
        
        self.prediction = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor):
        device = self.dummy_param.device
        x = x.reshape(x.shape[0], 1,  -1)
        x = x.to(device, dtype=torch.float)
        x = self.conv_blocks(x)
        
        if self.set_input_features:
            self.set_input_features = False
            self.prediction[0] = nn.Linear(x.shape[1], 1).to(device)
        x = self.prediction(x)
        return x