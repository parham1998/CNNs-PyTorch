import torch
import torch.nn as nn

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class MobileNetV1(nn.Module):
    mobileNetV1_st = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]
    
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(in_features=1024, out_features=10)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self):
        layers = []
        in_channels = 32
        #
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, stride=2, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace=True))
        
        for i in self.mobileNetV1_st:
            layers.append(Block(in_planes=in_channels, out_planes=i[0], stride=i[1]))
            in_channels = i[0]
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)