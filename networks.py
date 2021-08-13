# =============================================================================
# Import required libraries
# =============================================================================
from models import *

# =============================================================================
# CNN models
# =============================================================================
def networks(model_name, freezing=False):

    if model_name == 'VGG16_pretrained':
        return './checkpoints/cifar10_VGG16_pretrained.pth', VGG16_pretrained(freezing)
       
    if model_name == 'VGG16':
        return './checkpoints/cifar10_VGG16.pth', VGG16()
    
    if model_name == 'ResNet34':
        return './checkpoints/cifar10_ResNet34.pth', ResNet34()
    
    if model_name == 'ResNet101':
        return './checkpoints/cifar10_ResNet101.pth', ResNet101()
    
    if model_name == 'MobileNetV1':
        return './checkpoints/cifar10_MobileNetV1.pth', MobileNetV1()
    
    if model_name == 'MobileNetV2':
        return './checkpoints/cifar10_MobileNetV2.pth', MobileNetV2()