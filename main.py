# =============================================================================
# Import required libraries
# =============================================================================
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

import numpy as np
import matplotlib.pyplot as plt
import timeit

from networks import networks

# =============================================================================
# Check if CUDA is available
# =============================================================================
train_on_GPU = torch.cuda.is_available()
if not train_on_GPU:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...') 
    print(torch.cuda.get_device_properties('cuda'))

# =============================================================================
# Load data & data preprocessing
# =============================================================================
# number of subprocesses to use for data loading
num_workers = 2
batch_size = 128

def get_mean_and_std(trainset): 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    means = []
    stds = [] 
    for data, targets in trainloader:                
        batch_mean = np.mean(data.numpy(), axis=(0, 2, 3)) 
        batch_std = np.std(data.numpy(), axis=(0, 2, 3)) 
        means.append(batch_mean)
        stds.append(batch_std)
    return np.mean(means, axis=0), np.mean(stds, axis=0)

mean, std = get_mean_and_std(torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor()))

transform_train = transforms.Compose([
                      #transforms.Resize((32,32)),
                      #transforms.CenterCrop((32,32)),
                      #transforms.ColorJitter(1, 1),
                      #transforms.RandomRotation((90, 90)),
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

transform_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(
                         mean=mean, 
                         std=std
                     ),
                 ])

trainset = torchvision.datasets.CIFAR10(
    root="./data", 
    train=True, 
    download=True, 
    transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True,
    transform=transform_test)

# show one image
plt.imshow(trainset.data[6]) 

# prepare data loaders
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers)

# =============================================================================
# Show one batch of images
# =============================================================================
classes = trainset.classes

def imshow(img):
    img[0] = (img[0] * std[0]) + mean[0]
    img[1] = (img[1] * std[1]) + mean[1]
    img[2] = (img[2] * std[2]) + mean[2]
    # img shape => (3, h, w), img shape after transpose => (h, w, 3)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0))) 
       
# get one batch of images
images, labels = iter(trainloader).next()

# plot the images with corresponding labels
fig = plt.figure(figsize=(32, 16))
for i in np.arange(batch_size):
    ax = fig.add_subplot(8, 16, i+1)
    imshow(images[i])
    ax.set_title(classes[labels[i]])
    
# =============================================================================
# CNN models
# =============================================================================
PATH, net = networks('MobileNetV2')

print(net)    

if train_on_GPU:
    net.cuda()
    print('\n net can be trained on gpu') 
    
# =============================================================================
# Load model
# =============================================================================
net.load_state_dict(torch.load(PATH))

# =============================================================================
# Specify loss function and optimizer
# =============================================================================
lr = 0.01
momentum = 0.9
weight_decay = 5e-4
epochs = 200

criterion = nn.CrossEntropyLoss()
params = [p for p in net.parameters() if p.requires_grad == True]
optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# =============================================================================
# training
# =============================================================================
best_accuracy = 0

# losses per epoch
train_losses = []
test_losses = []

# ===========
# train model
# ===========
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(trainloader):
        
        if train_on_GPU:
            data, targets = data.cuda(), targets.cuda()

        # zero the gradients parameter
        optimizer.zero_grad()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = net(data)

        # calculate the batch loss
        loss = criterion(outputs, targets)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
    
        # parameters update
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    train_losses.append(train_loss/(batch_idx+1))
    print('Epoch: {} \t Training Loss: {:.3f} \t Training Accuracy: {:.3f}'.format(epoch+1, train_loss/(batch_idx+1), 100.*correct/total))

# ==============
# test model
# ==============
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(testloader):
        
            if train_on_GPU:
                data, targets = data.cuda(), targets.cuda()

            outputs = net(data)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    acc = 100.*correct/total
    test_losses.append(test_loss/(batch_idx+1))
    print('Epoch: {} \t Test Loss: {:.3f} \t Test Accuracy: {:.3f}'.format(epoch+1, test_loss/(batch_idx+1), acc))
    
    # save model if test accuracy has increased 
    global best_accuracy
    if acc > best_accuracy:
        print('Test accuracy increased ({:.3f} --> {:.3f}). saving model ...'.format(best_accuracy, acc))
        torch.save(net.state_dict(), PATH)
        best_accuracy = acc

print('==> Start Training ...')
for epoch in range(epochs):
    start = timeit.default_timer()
    train(epoch)
    test(epoch)
    scheduler.step()
    stop = timeit.default_timer()
    print('time: {:.3f}'.format(stop - start))
print('==> End of training ...')

# =============================================================================
# Plot train & test loss
# =============================================================================
plt.subplots(figsize=(16, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, 'r')
plt.xlabel('Training loss')
plt.subplot(1, 2, 2)
plt.plot(range(epochs), test_losses, 'b')
plt.xlabel('Test loss')
plt.show()

# =============================================================================
# Test model on test data & Confusion matrix
# =============================================================================
confusion_matrix = np.zeros(shape=(len(classes), len(classes)))

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (data, targets) in enumerate(testloader):
        
        if train_on_GPU:
            data, targets = data.cuda(), targets.cuda()

        outputs = net(data)

        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
print('Accuracy of the network on the 10000 test images: {:.3f}'.format(100. * correct/total))

# plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(confusion_matrix)
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)              
# rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")                
# loop over data dimensions and create text annotations.
for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")
fig.tight_layout()
plt.show()
