import pickle as pk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import math

#  Set fixed random number seed
torch.manual_seed(42)
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
print(dev)
device = torch.device(dev)

class CustomImageDataset(Dataset):
    def __init__(self, images_list,c1s,c2s,transform=None):
        """
        Custom dataset initialization.
        :param images_list: List of images
        :param c1s: List of labels for characteristic C1.
        :param c2s: List of labels for characteristic C1.
        :param transform: Transformations to apply to images.
        """
        self.datas = images_list
        self.c1s = c1s
        self.c2s = c2s
        self.targets, self.stratified_labels = self.transform_targets(c1s,c2s)
        self.transform = transform
    def __len__(self):
        """
        :return: The number of samples in the dataset.
        """
        return len(self.datas)

    def __getitem__(self, idx):
        """
        gets the image and corresponding labels for a given index.
        :param idx: Index of the sample
        :return: Tuple containing the image and label.
        """
        data = self.datas[idx]
        target = self.targets[idx]

        if self.transform:
            data = self.transform(data)

        return data, target
    def transform_targets(self,c1s,c2s):
        """
        Encodes the labels into one-hot representation
        :param c1s:  List of labels for characteristic C1.
        :param c2s:  List of labels for characteristic C2.
        :return: Tensor containing the one-hot encoded labels.
        """
        assert len(c1s) == len(c2s), f'c1 and c2 Shape mismatch:'
        num_samples = len(c1s)
        unique_c1s = sorted(set(c1s))
        unique_c2s = sorted(set(c2s))

        num_classes_c1 = len(unique_c1s)
        num_classes_c2 = len(unique_c2s)
        labels_one_hot = []
        stratified_labels = []
        for c1,c2 in zip(c1s,c2s):
            c1_idx = unique_c1s.index(c1)
            c2_idx = unique_c2s.index(c2)
            combined_index = c1_idx * num_classes_c2 + c2_idx
            one_hot = np.zeros((num_classes_c1*num_classes_c2))
            one_hot[combined_index] = 1
            labels_one_hot.append(one_hot)
            stratified_label = c1_idx * num_classes_c2 + c2_idx
            stratified_labels.append(stratified_label)

        labels_one_hot = np.vstack(labels_one_hot)

        return torch.tensor(labels_one_hot, dtype=torch.float32), stratified_labels
class CustomModel(nn.Module):
    """
    Explanation:
    Since our data is not complicated, I did not feel the need
    to create a larger number of neurons in the convolutional layers.
    Utilizing a Dropout layer in the model can aid in regularization, especially when dealing
    with limited data. Dropout randomly deactivates certain neurons during training,
    which can enhance the model's ability to generalize and reduce the risk of overfitting.
    right? idk, I wrote it at 4 in the morning.
    """
    def __init__(self,conv1_out_channels,filter_size,padding,pooling_function,dropout_rate):
        super(CustomModel, self).__init__()
        self.conv1 =  nn.Conv2d(3,conv1_out_channels, kernel_size=filter_size,padding=padding)
        self.conv2 = nn.Conv2d(conv1_out_channels,conv1_out_channels*2, kernel_size=filter_size,padding=padding)
        self.maxpool2d = pooling_function(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)

        linear_input = (conv1_out_channels*2)*((math.floor((math.floor(110 + (2*padding) - filter_size) + 1) + (2*padding) - filter_size) + 1)/2)**2
        linear_input = int(linear_input)
        self.fc1 = nn.Linear(linear_input,256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256,30)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool2d(x)
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
def evaluate(model, test_loader, error):
    """
    Function to evaluate the model on the test dataset.
    :param model: Trained model to be evaluated.
    :param test_loader: DataLoader for the test data.
    :param error: Loss function used to compute the model's error.
    :return: A tuple containing the error value and accuracy of the model on the test dataset.
    """
    correct = 0
    cur_loss = 0;

    # Iterating through the test data
    for test_datas, test_targets in test_loader:
        # Moving data to the GPU (if available)
        #  test_datas, test_targets = test_imgs.cuda(), test_targets.cuda()
        # Computing the model's output
        output = model(test_datas)
        # Calculating the loss value
        loss = error(output, test_targets)
        cur_loss+=loss.item()
        # Selecting the predicted class based on the model's output
        predicted = torch.max(output,1)[1]
        # Calculating the number of correct predictions
        correct += (predicted == test_targets).sum()

    # Computing the average loss and accuracy
    avg_loss = cur_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return avg_loss, accuracy
def fit(model, train_loader, error, val_loader, epochs):
    """
    Trains the model using the provided training data and evaluates it on the test data for the specified number of epochs.
    :param model: The neural network model to be trained.
    :param train_loader: DataLoader for the training data.
    :param error: Loss function used for training the model.
    :param test_loader: DataLoader for the test data
    :param epochs: Number of epochs for training
    :return: A tuple containing lists of training and test losses for each epoch.
    """

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

    model.train()
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        train_datas, train_targets = next(iter(train_loader))
        optimizer.zero_grad()
        output = model(train_datas)
        loss = error(output, train_targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        valid_loss, _ = evaluate(model,val_loader, error)
        valid_losses.append(valid_loss)

        #print(f'Epoch : {epoch},  train loss:{train_losses[-1]}, valid loss:{valid_losses[-1]}')

    return train_losses, valid_losses




#  Define transformation
my_preprocess = Compose([ToTensor(),
                        Normalize((0.,0.,0.,), (1.,1.,1.))])


#  Load our data
with open('input.pkl', 'rb') as f:
    data = pk.load(f)

images = data['imgs']
C1s = data['C1']
C2s = data['C2']

#  Create and instance of the custom dataset
homework_dataset = CustomImageDataset(images,C1s,C2s,transform=my_preprocess)

#  Sample dataset call
# idx = 0
# image, target = homework_dataset[idx]

#  Splitting into training and testing sets using StratifiedShuffleSplit
test_size = 0.2
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idxs, test_idxs = next(sss.split(homework_dataset.datas, homework_dataset.stratified_labels))

#  Creating training and testing datasets
train_dataset = Subset(homework_dataset, train_idxs)
test_dataset = Subset(homework_dataset, test_idxs)

#  Creating training and validation datasets
train_size = int(len(train_dataset) * 0.85)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset,[train_size,val_size])

# Batches were omitted due to the small number of records, and
# computational resources were not a significant issue in this case.

train_loader = DataLoader(train_dataset)
test_loader = DataLoader(test_dataset)
val_loader = DataLoader(val_dataset)

#  example_img, example_label = next(iter(train_loader))
#  print(f"Batch inputs shape: {example_batch_img.shape}, Batch labels shape: {example_batch_label.shape}")


#  Specify parameters
epochs = 15
error = nn.CrossEntropyLoss()
conv1_out_channels_s = [16,32,64]
filter_sizes = [3,5]
paddings = [0,1]
pooling_functions = [nn.MaxPool2d, nn.AvgPool2d]
dropout_rates = [0,0.1,0.25,0.4,0.5]

for a,conv1_out_channels in enumerate(conv1_out_channels_s):

    for b, filter_size in enumerate(filter_sizes):

        for c, padding in enumerate(paddings):

            for d, pooling_function in enumerate(pooling_functions):
                for e, dropout_rate in enumerate(dropout_rates):
                   model = CustomModel(conv1_out_channels,filter_size,padding,pooling_function,dropout_rate)
                   print(model)
                   a=1
                   train_losses, valid_losses = fit(model,train_loader, error, val_loader, epochs)
                   print(f'Specification:\nNumber of channels of first Conv2D layer: {conv1_out_channels},\nFilter Size: {filter_size},\nPadding: {padding},\nPooling Function: {pooling_function},\nDropout Rate: {dropout_rate}.')
                   loss, acc = evaluate(model, train_loader, error)
                   print(f"Train Accuracy:{acc}, Train loss:{loss}")
                   loss, acc = evaluate(model, test_loader, error)
                   print(f"Test Accuracy:{acc}, Test loss:{loss}")

                   loss, acc = evaluate(model, val_loader, error)
                   print(f"Validation Accuracy:{acc}, Validation loss:{loss}")
                   print('\n\n\n')