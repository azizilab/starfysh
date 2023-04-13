# define the resNet block
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F


class net(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels,stride=1):
        super().__init__()

        self.res_fun = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            #nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels,out_channels * net.expansion),
            #nn.BatchNorm1d(out_channels * net.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != net.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels * net.expansion),
                #nn.BatchNorm1d(out_channels * net.expansion)
            )
    
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.res_fun(x) + self.shortcut(x))
    
# define the resNet
class ResNet(nn.Module):

    def __init__(self, block, num_block):

        super().__init__()
        self.in_channels = 4096
        

        self.conv1 = nn.Sequential(
            nn.Linear(6000, 4096),
            #nn.MaxPool1d(kernel_size=2,stride=2),
            #nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)
            )
        self.conv2_x = self._make_layer(block, 4096, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 2048, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 1024, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        #self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512 * block.expansion, 60)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):

        output = self.conv1(x)
        #print(output.shape)
        output = self.conv2_x(output)
        #print(output.shape)
        output = self.conv3_x(output)
        #print(output.shape)
        output = self.conv4_x(output)
        #print(output.shape)
        output = self.conv5_x(output)
        #print(output.shape)
        #output = self.avg_pool(output)
        #print(output.shape)
        output = output.view(output.size(0), -1)

        #print(output.shape)
        output1 = self.fc1(output)

 
        return output1

    def resnet18():

        return ResNet(net, [2, 2, 2, 2])

    def resnet50():

        return ResNet(net, [3, 4, 6, 3])
