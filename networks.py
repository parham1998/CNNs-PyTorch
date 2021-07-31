# =============================================================================
# Import required libraries
# =============================================================================
from models import *

# =============================================================================
# CNN models
# =============================================================================
def networks(model_name, freezing = False):

    if model_name == 'VGG16_pretrained':
        return './checkpoints/cifar10_VGG16_pretrained.pth', VGG16_pretrained(freezing = False)
       
    if model_name == 'VGG16':
        return './checkpoints/cifar10_VGG16.pth', VGG16()
    
    if model_name == 'ResNet34':
        return './checkpoints/cifar10_ResNet34.pth', ResNet34()
    
    if model_name == 'ResNet101':
        return './checkpoints/cifar10_ResNet101.pth', ResNet101()