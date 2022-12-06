import torch
from torch import nn
from torchinfo import summary

class AlexNet(nn.Module):
    def __init__(self,
                 intput_shape:int = 3,
                 output_shape:int = 1):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=intput_shape,
                      out_channels=96,
                      kernel_size=11,
                      stride=4,
                      padding=0),
            nn.MaxPool2d(kernel_size=3,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.MaxPool2d(kernel_size=3,
                         stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=3,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 6 * 6,
                      out_features=4096),
            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.Linear(in_features=4096,
                      out_features=output_shape)
        )
        
    def forward(self, x):
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))
    

if __name__ == '__main__':
    model = AlexNet(3, 3)
    print(summary(model=model,
                  input_size=(1, 3, 227, 227),
                  col_names=['input_size', 'output_size', 'num_params', 'trainable'],
                  col_width=20))