import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision import models
from PIL import Image


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


class ObjectModel(nn.Module):
    def __init__(self, num_classes, input_shape, transfer=False):
        super().__init__()

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

# Define hyperparameters (Example)
batch_size = 2
epochs = 20
patience = 5
start_lr = 1e-6
end_lr = 10
num_classes = 13

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if verboseFLAG:
    print(device)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
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
train_dataset = CustomDataset(images_dir='path/to/train/images', labels_dir='path/to/train/labels', transform=transform)
val_dataset = CustomDataset(images_dir='path/to/val/images', labels_dir='path/to/val/labels', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

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
