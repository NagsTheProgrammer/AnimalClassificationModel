# Tutorial
# Import libraries

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import glob
from sklearn.model_selection import StratifiedShuffleSplit
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import argparse
from pedestrian_detection_utils import *  #
import matplotlib.pyplot as plt
import os

# -- ARGUMENTS --
# Define arguments, check system

# ***placeholders for now
learningRate = 100
batchSize = 100
epochs = 100
nepochs = 20  # from garbage classifier
trainSplit = 0.7  # training split is 70%
valSplit = 0.15  # validation split is 15%
testSplit = 0.15  # testing split is 15%
modelBenchmarkPATH = '\\bestmodelpath'
imagesPATH = '\\images'
labelsPATH = '\\labels'
verboseFLAG = False
transferLearningFLAG = True
numClasses = 13
learning_rate = 1e-5
gamma_val = 0.9
best_loss = 1e+20


# Util section
class TorchVisionDataset(Dataset):
    def __init__(self, data_dic, transform=None):
        self.file_paths = data_dic["X"]
        self.labels = data_dic["Y"]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)
        return image, label

class ObjectModel(nn.Module):
    def __init__(self, num_classes, input_shape, transfer=False):
        super().__init__()

        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape

        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=transfer)

        if self.transfer:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        n_features = self._get_conv_output(self.input_shape)
        self.classifier = nn.Linear(n_features, num_classes)

    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.feature_extractor(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # will be used during inference
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if verboseFLAG:
    print(device)

# print(args) # printed the values from each argument? can delete, this was a test

# -- DATA PREP --
# Instantiate arguments, prepare data

# Getting dataset
rootDir = os.path.dirname(os.path.abspath(__file__))
imagesPATH = os.path.join(rootDir, f'dataset\smallDataset\\images\\')
labelsPATH = os.path.join(rootDir, f'dataset\smallDataset\\annotations\\')

# Getting annotations from XML files
# *** to delete
# args.images_path="path to images" #path to root folder
# images = glob.glob(imagesPATH)

# -- LABELS --
labels = ['cereal', 'chocolate_milk', 'heineken', 'iron_man', 'medicine', 'milk_bottle', 'milk_box', 'monster',
          'purple_juice', 'red_juice', 'shampoo', 'tea_box', 'yellow_juice']

# Experimental setup train val test split
# Data augmentation
# Data loaders
trainloader, valloader, testloader = \
    get_data_loaders(imagesPATH, valSplit, testSplit, \
                     batch_size=batchSize, verbose=verboseFLAG)

# Create our model
net = ObjectModel(4, (3, 224, 224), transferLearningFLAG)
net.to(device)

train_validate(net, trainloader, valloader, epochs, batchSize, \
               learningRate, modelBenchmarkPATH, device, verboseFLAG)

net.load_state_dict(torch.load(modelBenchmarkPATH))

test(net, testloader, device)

# The default image size used by the authors of YOLOv5 during training is 640x640 pixels.
# So, they recommend that the trained YOLOv5 model be used on new input images that are resized to a multiple
# of 32 like 320x320, 352x352, 384x384,…. for efficiency.
# So, the required dimensions depend on the input image size used during training and the input images used
# for detection should be resized to the same size used during training which is preferably a multiple of 32.

# Transfer learning Michael will do this
# Add and train new top/predictor
# Fine-tune learning layers
# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer to output the correct number of classes
model.model[-1].c = numClasses

# Loss and metrics, Callbacks and tracking
# Loss: categorical cross-entropy
# Metrics: accuracy, sensitivity, specificity, confusion matrix, training and inference time
# Early stopping/patience
# Model benchmark
# Learning rate scheduler
# Weights and biases (train/val loss)
criterion = nn.CrossEntropyLoss()  # 1e-5 and 1e-1
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=gamma_val)  # ExponentialLR decays the learning rate of each parameter group


# by gamma every epoch, this is from the garbage classifier


# Hyperparameters
# Batch size
# Number of epochs
# Learning rate


def find_optimal_lr(model, criterion, optimizer, dataloader, num_iter=100, start_lr=1e-7, end_lr=10, device='cpu'):
    # Set up the learning rate scheduler to gradually increase the learning rate over time
    lr_lambda = lambda x: 10 ** (x / (num_iter - 1) * (end_lr - start_lr) + start_lr)
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    # Set the model to training mode
    model.train()

    # Run the learning rate range test
    losses = []
    lrs = []
    for i, (inputs, labels) in enumerate(dataloader):
        if i >= num_iter:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        losses.append(loss.item())
        lrs.append(lr_scheduler.get_lr())

    # Find the optimal learning rate
    optimal_lr = lrs[losses.index(min(losses))]

    plt.xscale('log')
    plt.plot(lrs, losses)
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')
    plt.show()

    return optimal_lr, min(losses)


lr, loss = find_optimal_lr(model, criterion, optimizer, dataloader)


# Train


# Test
# Run prediction on your test set
# Extract relevant metrics
# Measure inference time

# train_validate
def train_validate(net, trainloader, valloader, epochs, batch_size,
                   learning_rate, best_model_path, device, verbose):
    best_loss = 1e+20
    for epoch in range(epochs):  # loop over the dataset multiple times

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        # Training Loop
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end=' ')
        scheduler.step()

        val_loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
            print(f'val loss: {val_loss / i:.3f}')

            # Save best model
            if val_loss < best_loss:
                print("Saving model")
                torch.save(net.state_dict(), best_model_path)
                best_loss = val_loss

    print('Finished Training')
