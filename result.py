import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset
from torchvision import transforms, models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Vision Dataset
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
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


# Custom nn Module
class AnimalModel(nn.Module):
    def __init__(self, num_classes, input_shape, transfer=False):
        super().__init__()

        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape

        # transfer learning if pretrained=True
        self.feature_extractor = models.densenet161(pretrained=transfer)

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


# Dataloader generation convenience function

def get_data_loaders(val_split, test_split, batch_size=32, verbose=True):
    num_workers = 0
    random_state = 10
    n_splits = 1

    # Listing the data
    # Cats
    print("LISTING DATA")
    input_dir = "dataset/temp/cats"
    images = [os.path.join(input_dir, image) for image in os.listdir(input_dir)]
    cat_images = np.array(images)  # transform to numpy
    cat_labels = ['cat'] * len(cat_images)

    # Dogs
    input_dir2 = "dataset/temp/dogs"
    images2 = [os.path.join(input_dir2, image) for image in os.listdir(input_dir2)]
    dog_images = np.array(images2)  # transform to numpy
    dog_labels = ['dog'] * len(dog_images)

    # Panda
    input_dir3 = "dataset/temp/panda"
    images3 = [os.path.join(input_dir3, image) for image in os.listdir(input_dir3)]
    panda_images = np.array(images3)  # transform to numpy
    panda_labels = ['panda'] * len(panda_images)

    # Appending lists
    images = np.append(np.append(cat_images, dog_images), panda_images)
    labels = cat_labels + dog_labels + panda_labels
    labels = np.array(labels)

    # Formatting the labs as ints
    classes = np.unique(labels).flatten()
    labels_int = np.zeros(labels.size, dtype=np.int64)

    # Convert string labels to integers
    for index, class_name in enumerate(classes):
        labels_int[labels == class_name] = index

    if verbose:
        print("Number of images in the dataset:", len(images))
        for index, class_name in enumerate(classes):
            print("Number of images in class ", class_name,
                  ":", (labels_int == index).sum())

    # Splitting the data in dev and test sets
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_split, random_state=random_state)
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
        n_splits=n_splits, test_size=val_split, random_state=random_state)
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

    # Transforms
    torchvision_transform_train = transforms.Compose(
        [transforms.Resize((args.unified_image_width, args.unified_image_height)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.ToTensor()])

    # Datasets
    train_dataset_unorm = CustomDataset(
        train_images, train_labels, transform=torchvision_transform_train)

    # Get training set stats
    trainloader_unorm = torch.utils.data.DataLoader(
        train_dataset_unorm, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    mean_train, std_train = get_dataset_stats(trainloader_unorm)

    if verbose:
        print("Statistics of training set")
        print("Mean:", mean_train)
        print("Std:", std_train)

    torchvision_transform = transforms.Compose(
        [transforms.Resize((args.unified_image_width, args.unified_image_height)),
         transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean_train, std=std_train)])

    torchvision_transform_test = transforms.Compose(
        [transforms.Resize((args.unified_image_width, args.unified_image_height)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean_train, std=std_train)])

    # Get the train/val/test loaders
    train_dataset = CustomDataset(
        train_images, train_labels, transform=torchvision_transform)
    val_dataset = CustomDataset(val_images, val_labels, transform=torchvision_transform)
    test_dataset = CustomDataset(
        test_images, test_labels, transform=torchvision_transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# Stats generation convenience function

def get_dataset_stats(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in data_loader:
        samples = data[0]
        batch_samples = samples.size(0)
        samples = samples.view(batch_samples, samples.size(1), -1)
        mean += samples.mean(2).sum(0)
        std += samples.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


# Model training convenience function

def define_hyperparameters_and_train_model(best_model_path, device, verbose, patience):
    epochs_range = [10, 20, 30, 40, 50]
    batch_size_range = [2, 4, 8, 16, 32, 48, 64]
    learning_rate_range = [0.0001, 0.001, 0.01, 0.1, 1]

    # Perform the grid search
    best_val_loss = float('inf')
    best_hyperparameters = None

    for epochs in epochs_range:
        for batch_size in batch_size_range:
            for learning_rate in learning_rate_range:
                train_loader, val_loader, test_loader = get_data_loaders(args.val_split, args.test_split)
                current_model = AnimalModel(args.num_classes,
                                            (args.num_classes, args.unified_image_width, args.unified_image_height))
                current_model.to(device)
                val_loss = train_validate_with_hyperparameters(current_model, train_loader, val_loader, epochs,
                                                               learning_rate, best_model_path, device, patience,
                                                               verbose)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_hyperparameters = (epochs, batch_size, learning_rate)
                    torch.save(current_model.state_dict(), best_model_path)

    if verbose:
        print("Best nbr epochs:", best_hyperparameters[0])
        print("Best batch size:", best_hyperparameters[1])
        print("Best learning rate:", best_hyperparameters[2])

    return best_hyperparameters


def train_validate_with_hyperparameters(observed_model, train_loader, val_loader, epochs,
                                        learning_rate, best_model_path, device, patience, verbose):
    best_loss = float("inf")
    gamma = 0.9
    counter = 0

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = torch.optim.AdamW(observed_model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(epochs):  # loop over the dataset multiple times

        # Training Loop
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = observed_model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        val_loss = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = observed_model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

            # Save best model
            if val_loss < best_loss:
                torch.save(observed_model.state_dict(), best_model_path)
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print(f'Validation loss has not improved for {patience} epochs. Stopping training.')
                    break

    return val_loss / (i + 1)


# Test convenience function

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


# Main

if __name__ == "__main__":
    # Constants definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size,  number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=10, help='total epochs')
    parser.add_argument('--val_split', type=float, default=0.2, help='val split')
    parser.add_argument('--test_split', type=float, default=0.2, help='test split')
    parser.add_argument('--best_model_path', type=str, default="best_model", help='best model path')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose debugging flag')
    parser.add_argument('--transfer_learning', type=bool, default=True, help='transfer learning flag')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes in dataset')
    parser.add_argument('--unified_image_height', type=int, default=224, help='transfer learning flag')
    parser.add_argument('--unified_image_width', type=int, default=224, help='transfer learning flag')
    parser.add_argument('--patience', type=int, default=5, help='transfer learning flag')

    args = parser.parse_args()

    # Device choice
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.verbose:
        print("Chosen device:", device)

    # Model training and saving
    best_batch_size = define_hyperparameters_and_train_model(args.best_model_path, device, args.verbose, args.patience)[
        1]

    # Get test dataloader
    test_loader = get_data_loaders(args.val_split, args.test_split, batch_size=best_batch_size, verbose=args.verbose)[2]

    # Loading best model
    model = AnimalModel(args.num_classes, (args.num_classes, args.unified_image_width, args.unified_image_height))
    model.load_state_dict(torch.load(args.best_model_path))

    # Best model testing
    test(model, test_loader, device)
