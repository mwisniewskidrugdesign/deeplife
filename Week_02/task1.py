import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision  #to get the MNIST dataset

import numpy as np
import matplotlib.pyplot as plt

#get and format the training set
from torchvision.transforms import ToTensor, Normalize, Compose
## defining a pre-processing tranformation
# ToTensor() converts images or numpy.ndarray to tensors
# Normalization with mean 0 and std 1
my_preprocess = Compose([ToTensor(),
                         Normalize((0,), (1,))])

mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=my_preprocess)

mnist_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=my_preprocess)

#In case we need higher precision,
#  DoubleTensor is 64-bit floating point and FloatTensor is 32-bit floating point number
#x_train=mnist_trainset.data.type(torch.DoubleTensor)
print(mnist_trainset.data.size()) # train data
print(mnist_trainset.targets.size()) # labels

print("\nNumber of Targets :",len(np.unique(mnist_trainset.targets)))
print("Train targets Values :", np.unique(mnist_trainset.targets),"\n")

print(mnist_testset.data.size()) # test data
print("Test targets Values :", np.unique(mnist_testset.targets),"\n")

#  print(f"Image {0} is a {mnist_trainset.targets[0]}")
#  plt.imshow(mnist_trainset.data[0], cmap='gray')
#  plt.show()

# we'll use a batch size of 128 for training our network
batch_size = 128

# initialize a DataLoader object for each dataset
train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_data,
                                               batch_size=batch_size,
                                               num_workers=2)
test_dataloader = torch.utils.data.DataLoader(mnist_testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)



# grab the first batch from one of our DataLoader objects
example_batch_img, example_batch_label = next(iter(train_dataloader))

# inputs and labels are batched together as tensor objects
print(f"Batch inputs shape: {example_batch_img.shape}, Batch labels shape: {example_batch_label.shape}")