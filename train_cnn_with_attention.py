import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
from cnn_with_attention import FlexibleCNNWithAttention  # Assume your attention model is in attention_cnn.py

class CustomPKMNDataset(Dataset):
    def __init__(self, img_dir, data_file, transform=None, class_to_idx=None):
        """
        Initialize the dataset.
        Args:
            img_dir (str): Path to the directory containing images.
            data_file (str): Path to the CSV file containing metadata (image, type, caption).
            transform (callable, optional): Transform to be applied to images.
            class_to_idx (dict, optional): Mapping from class name (type) to integer index.
        """
        self.img_dir = img_dir
        self.data = pd.read_csv(data_file)
        self.transform = transform

        # Generate or validate class_to_idx
        if class_to_idx is None:
            class_names = self.data['type'].unique().tolist()
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        else:
            self.class_to_idx = class_to_idx

        # Ensure all types in the dataset have a valid mapping
        invalid_types = set(self.data['type']) - set(self.class_to_idx.keys())
        if invalid_types:
            raise ValueError(f"The following types in the dataset have no mapping in class_to_idx: {invalid_types}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx (int): Index of the sample.
        Returns:
            A tuple (image_tensor, caption_tensor, type_idx).
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        type_str = row['type']
        caption = row['caption']

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert type to class index
        type_idx = self.class_to_idx[type_str]

        # Tokenize the caption (example: character-based tokenization)
        caption_tensor = torch.tensor([ord(c) for c in caption], dtype=torch.long)

        return image, caption_tensor, type_idx

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    """
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    type_indices = [item[2] for item in batch]

    images = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    type_indices = torch.tensor(type_indices, dtype=torch.long)

    return images, captions, type_indices

def calculate_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total * 100

def train_one_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, _, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct_predictions += (torch.max(outputs, dim=1)[1] == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples * 100
    return epoch_loss, epoch_accuracy

def validate(model, dataloader, criterion, device="cpu"):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, _, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            correct_predictions += (torch.max(outputs, dim=1)[1] == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = val_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples * 100
    return epoch_loss, epoch_accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10, patience=3, device="cpu"):
    print(f"Training on {device}")
    model = model.to(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device=device)

        if scheduler:
            scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_cnn_attention.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model_cnn_attention.pth"))

# Parameters
img_dir = "data/images"
data_file = "data/data.csv"
batch_size = 32
num_epochs = 100
learning_rate = 0.001
val_split = 0.2
patience = 10

# Define your classes
class_names = [
    "normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground",
    "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
]
num_classes = len(class_names)
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

# Transforms
transform = transforms.Compose([
    transforms.Resize((272, 272)),
    transforms.ToTensor()
])

# Create dataset
dataset = CustomPKMNDataset(img_dir, data_file, transform=transform, class_to_idx=class_to_idx)

# Split dataset
val_size = int(val_split * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Model, criterion, optimizer
model = FlexibleCNNWithAttention(input_channels=3, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler, num_epochs=num_epochs, patience=patience, device="cuda" if torch.cuda.is_available() else "cpu")
