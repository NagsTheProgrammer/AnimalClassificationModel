import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import glob

verboseFLAG = True


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


class AnimalModel(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        self.num_classes = num_classes
        self.input_shape = input_shape

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


# Define hyperparameters (Example)
batch_size = 2
epochs = 20
patience = 5
start_lr = 1e-5
end_lr = 1
num_classes = 3

# Define device
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
if verboseFLAG:
    print(device)

# Define transforms (Resizing and data augmentation)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])


# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer to output the correct number of classes
model.model[-1].c = num_classes

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=start_lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)

# Define the dataset and dataloaders
paths = {'path/to/train/images', 'path/to/train/labels'}
train_dataset = TorchVisionDataset(paths, transform=transform)
val_dataset = TorchVisionDataset(paths, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define learning rate
learning_rate, loss = find_optimal_lr(model, criterion, optimizer, train_loader, num_iter=100, start_lr=start_lr, end_lr=end_lr, device=device_name)

# Train the model
best_loss = float('inf')
patience_count = 0
for epoch in range(epochs):
    train_loss = 0
    for images, targets in train_loader:
        # Move data to GPU if available
        images = images.to(device)
        targets = targets.to(device)

        # Zero the gradients and forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute the loss and backward pass
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update the training loss
        train_loss += loss.item()

    # Update the scheduler and print the loss
    scheduler.step()
    print(f'Epoch {epoch + 1}: Train loss={train_loss / len(train_loader)}')

    # Save the best model based on validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_count = 0
    else:
        patience_count += 1
        if patience_count == patience:
            print('Stopping early due to lack of improvement')
            break
