# =============================================================================
# Import all libraries
# =============================================================================
import torchvision
from torch import nn


# =============================================================================
# loading pretrained vgg16
# =============================================================================
def VGG16_pretrained(freezing=False):

    net = torchvision.models.vgg16(pretrained=True)   

    # change the number of classes 
    net.classifier = nn.Linear(in_features=512, out_features=10)
    net.avgpool.output_size = (1, 1)

    # freeze features(convolutions) weights
    if freezing == True:
        for param in net.features.parameters():
            param.requires_grad = False
        
    return net