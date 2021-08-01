# =============================================================================
# Import required libraries
# =============================================================================
import torch
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from networks import networks

# =============================================================================
# Test models on one image
# =============================================================================
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

img = Image.open("test.jpg")
plt.imshow(img)

# np.array(img).shape => (H, W, 3)
# transforms.ToTensor()(img).shape => (3, H, W)

mean = [0.4914058 , 0.48216087, 0.4465181]
std = [0.24668495, 0.24312325, 0.26110893]

my_transforms = transforms.Compose([
                transforms.Resize((32,32)),
                #transforms.CenterCrop((32,32)),
                #transforms.ColorJitter(1, 1),
                transforms.RandomRotation((90, 90)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), padding=2),
                #transforms.RandomGrayscale(p = 1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean, 
                    std=std
                ),
             ])
tensor = my_transforms(img)

def imshow(img):
    img[0] = (img[0] * std[0]) + mean[0]
    img[1] = (img[1] * std[1]) + mean[1]
    img[2] = (img[2] * std[2]) + mean[2]
    # img shape => (3, h, w), img shape after transpose => (h, w, 3)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0))) 
imshow(tensor)

# img_model.shape => [1, 3, 32, 32]
img_model = tensor.unsqueeze(0)

# =============================================================================
# Predict test image label
# =============================================================================
PATH, net = networks('VGG16')
net.load_state_dict(torch.load(PATH))

output = net(img_model)

_, predicted = torch.max(output, 1)
print(classes[predicted])