import torch
import torch.nn as nn
import torch.nn.functional as F

class bottleneck(nn.Module):
    
    def __init__(self, expantion, in_channels, out_channels, stride):
        super(bottleneck, self).__init__()
        self.identical = True if stride == 1 and in_channels == out_channels else False
        
        self.inC = in_channels        
        self.expantion = int(expantion * in_channels)
        # 1*1 pointwise exapntion
        if self.inC != self.expantion:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.expantion, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(self.expantion)
        # 3*3 depthwise 
        self.conv2 = nn.Conv2d(in_channels=self.expantion, out_channels=self.expantion, kernel_size=3, stride=stride, padding=1, groups=self.expantion, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expantion)
        # 1*1 pointwise compression
        self.conv3 = nn.Conv2d(in_channels=self.expantion, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x))) if self.inC != self.expantion else x
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + x if self.identical else out
        return out
        

class MobileNetV2(nn.Module):
    # (exapntion factor, num of output channels, repeating num, stride)
    mobileNetV2_st = [
        (1, 16, 1, 1),
        (6, 24, 2, 1), # change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1)]
    
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=10),
        )
        
        
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self):
        layers = []
        in_channels = 32
        layers.append(nn.Conv2d(in_channels=3, out_channels=32, stride=1, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU6(inplace=True))
        
        for param in self.mobileNetV2_st:
            for repeat in range(0, param[2]):
                stride = param[3] if repeat == 0 else 1
                layers.append(bottleneck(param[0], in_channels, param[1], stride))
                in_channels = param[1]
        
        layers.append(nn.Conv2d(in_channels=320, out_channels=1280, stride=1, kernel_size=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(1280))
        layers.append(nn.ReLU6(inplace=True))
        return nn.Sequential(*layers)