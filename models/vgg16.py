import torch
import torch.nn as nn

# short form
class VGG16(nn.Module):
    vgg16_st = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']
    
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=10))
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self):
        layers = []
        in_channels = 3
        for i in self.vgg16_st:
            if i == 'P':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(i))
                layers.append(nn.ReLU(inplace=True))
                in_channels = i
            
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)

"""
# [(image - kernel + 2p)/s + 1] * [(image - kernel + 2p)/s + 1]
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            # Conv Layer block 1
            # 32 * 32 * 3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32 * 32 * 64 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # pooling
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2 
            # 16 * 16 * 64 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16 * 16 * 128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # pooling
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 3 
            # 8 * 8 * 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8 * 8 * 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8 * 8 * 256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # pooling
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Layer block 4
            # 4 * 4 * 256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 4 * 4 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 4 * 4 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # pooling
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 5
            # 2 * 2 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 2 * 2 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 2 * 2 * 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # pooling
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 1 * 1 * 512
            nn.AvgPool2d(kernel_size=1, stride=1)
        )

        self.classifier = nn.Sequential(
            #nn.Linear(in_features=1 * 1 * 512, out_features=512),
            #nn.ReLU(True),
            #nn.Dropout(0.5),
            #nn.Linear(in_features=512, out_features=512),
            #nn.ReLU(True),
            #nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x):        
        # conv layers
        x = self.features(x)
        
        # flatten // such as global pooling
        #x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.classifier(x)
        return x
"""