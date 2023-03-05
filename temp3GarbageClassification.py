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
import xml.etree.cElementTree as et

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

# import torch.nn as nn
# import torch.nn.functional as F
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))

class AnimalModel(nn.Module):
    def __init__(self, num_classes, input_shape, transfer=False):
        super().__init__()

        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape

        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=transfer)

        # Load the pre-trained YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Freeze all layers in the model
        for param in model.parameters():
            param.requires_grad = False

        # Replace the final layer to output the correct number of classes
        model.model[-1].c = num_classes

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

# Main function call
if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    # parser.add_argument('--batch_size', type=int, default=2,
    #                     help='batch size,  number of images in each iteration during training')
    # parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    # parser.add_argument('--val_split', type=float, default=0.2, help='val split')
    # parser.add_argument('--test_split', type=float, default=0.2, help='test split')
    # parser.add_argument('--best_model_path', type=str, help='best model path')
    # parser.add_argument('--images_path', type=str, help='path to images')
    # parser.add_argument('--verbose', type=bool, default=True, help='verbose debugging flag')
    # parser.add_argument('--transfer_learning', type=bool, default=False, help='transfer learning flag')
    #
    # args = parser.parse_args()

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    if verboseFLAG:
        print(device)

    trainloader, valloader, testloader = \
        get_data_loaders(imagesPATH, valSplit, testSplit, \
                         batch_size=batchSize, verbose=verboseFLAG)

    # Create our model
    net = AnimalModel(3, (3, 224, 224), transferLearningFLAG) # **change input model shape to match input dataset
    net.to(device)

    train_validate(net, trainloader, valloader, epochs, batchSize, \
                   learningRate, modelBenchmarkPATH, device, verboseFLAG)

    net.load_state_dict(torch.load(modelBenchmarkPATH))

    test(net, testloader, device)


# Function to get the statistics of a dataset
def get_dataset_stats(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in data_loader:
        data = data[0]  # Get the images to compute the stgatistics
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def get_data_loaders(images_path, val_split, test_split, batch_size=32, verbose=True):
    """
    These function generates the data loaders for our problem. It assumes paths are
    defined by "/" and image files are jpg. Each subfolder in the images_path
    represents a different class.

    Args:
        images_path (_type_): Path to folders containing images of each class.
        val_split (_type_): percentage of data to be used in the val set
        test_split (_type_): percentage of data to be used in the val set
        verbose (_type_): debug flag

    Returns:
        DataLoader: Train, validation and test data laoders.
    """

    # Listing the data
    images = glob.glob(images_path + "*/*.jpg")
    images = np.array(images)
    labels = np.array([f.split("/")[-2] for f in images])

    # Formatting the labs as ints
    classes = np.unique(labels).flatten()
    labels_int = np.zeros(labels.size, dtype=np.int64)

    # Convert string labels to integers
    for ii, jj in enumerate(classes):
        labels_int[labels == jj] = ii

    if verbose:
        print("Number of images in the dataset:", images.size)
        for ii, jj in enumerate(classes):
            print("Number of images in class ", jj,
                  ":", (labels_int == ii).sum())

    # Splitting the data in dev and test sets
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split, random_state=10)
    sss.get_n_splits(images, labels_int)
    dev_index, test_index = next(sss.split(images, labels_int))

    dev_images = images[dev_index]
    dev_labels = labels_int[dev_index]

    test_images = images[test_index]
    test_labels = labels_int[test_index]

    # Splitting the data in train and val sets
    val_size = int(val_split * images.size)
    val_split = val_size / dev_images.size
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_split, random_state=10)
    sss2.get_n_splits(dev_images, dev_labels)
    train_index, val_index = next(sss2.split(dev_images, dev_labels))

    train_images = images[train_index]
    train_labels = labels_int[train_index]

    val_images = images[val_index]
    val_labels = labels_int[val_index]

    if verbose:
        print("Train set:", train_images.size)
        print("Val set:", val_images.size)
        print("Test set:", test_images.size)

    # Representing the sets as dictionaries
    train_set = {"X": train_images, "Y": train_labels}
    val_set = {"X": val_images, "Y": val_labels}
    test_set = {"X": test_images, "Y": test_labels}

    # Transforms
    torchvision_transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomVerticalFlip(),
                                                      transforms.ToTensor()])

    # Datasets
    train_dataset_unorm = TorchVisionDataset(
        train_set, transform=torchvision_transform_train)

    # Get training set stats
    trainloader_unorm = torch.utils.data.DataLoader(
        train_dataset_unorm, batch_size=batch_size, shuffle=True, num_workers=0)
    mean_train, std_train = get_dataset_stats(trainloader_unorm)

    if verbose:
        print("Statistics of training set")
        print("Mean:", mean_train)
        print("Std:", std_train)

    torchvision_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=mean_train, std=std_train)])

    torchvision_transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=mean_train, std=std_train)])

    # Get the train/val/test loaders
    train_dataset = TorchVisionDataset(
        train_set, transform=torchvision_transform)
    val_dataset = TorchVisionDataset(val_set, transform=torchvision_transform)
    test_dataset = TorchVisionDataset(
        test_set, transform=torchvision_transform_test)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0)

    return trainloader, valloader, testloader

# for performing transfer learning steps:
# 1. load pre-traine model
# 2. freeze top layers
# 3. add top layer (prediction layer)
def transfer_model(net):
    net.load_state_dict(torch.load(r'/yoloV3/yolov3.weights')) # ** check path to make sure this works properly
    net.eval()

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


def test(net, testloader, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the test images: {100 * correct / total} %')
