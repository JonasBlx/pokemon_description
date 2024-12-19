import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence

from cnn import FlexibleCNN

class CustomPKMNDataset(Dataset):
    def __init__(self, img_dir, data_file, transform=None, class_to_idx=None):
        """
        Initialise le dataset.
        """
        self.img_dir = img_dir
        self.data = pd.read_csv(data_file)
        self.transform = transform

        # Générer ou vérifier le mapping class_to_idx
        if class_to_idx is None:
            class_names = self.data['type'].unique().tolist()
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        else:
            self.class_to_idx = class_to_idx

        # Vérifier que tous les types ont une correspondance
        invalid_types = set(self.data['type']) - set(self.class_to_idx.keys())
        if invalid_types:
            raise ValueError(f"Types invalides trouvés : {invalid_types}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retourne (image_tensor, caption_tensor, type_idx).
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        type_str = row['type']
        caption = row['caption']

        # Chargement de l'image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Type -> index
        type_idx = self.class_to_idx[type_str]

        # Tokenisation simple du caption
        caption_tensor = torch.tensor([ord(c) for c in caption], dtype=torch.long)

        return image, caption_tensor, type_idx

def custom_collate_fn(batch):
    """
    Collate function pour gérer les séquences de longueurs variables.
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

    print("d")
    for batch in dataloader:
        # Ensure proper unpacking based on the dataloader's output
        if len(batch) == 2:
            images, labels = batch
        elif len(batch) == 3:
            images, _, labels = batch
        else:
            raise ValueError("Unexpected number of elements in dataloader output")
        print("e")
        images, labels = images.to(device), labels.to(device)
        print("f")
        # Forward pass
        print(f"Image shape: {images.shape}")
        outputs = model(images)
        print("g")
        loss = criterion(outputs, labels)
        print("h")
        # Backward pass and optimization
        optimizer.zero_grad()
        print("i")
        loss.backward()
        print("j")
        optimizer.step()
        print("k")
        
        # Update metrics
        running_loss += loss.item()
        correct_predictions += (torch.max(outputs, dim=1)[1] == labels).sum().item()
        total_samples += labels.size(0)

    # Compute average loss and accuracy
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
        print("a")
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        print("b")
        val_loss, val_accuracy = validate(model, val_loader, criterion, device=device)
        print("c")
        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_cnn.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model_cnn.pth"))


if __name__ == "__main__":
    # Paramètres
    img_dir = "data/images"
    data_file = "data/data.csv"
    batch_size = 64  # Augmentation de la taille du batch
    num_epochs = 100
    learning_rate = 0.001
    val_split = 0.2
    patience = 10

    class_names = [
        "normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground",
        "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
    ]
    num_classes = len(class_names)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    # Ajout d'augmentations de données
    transform = transforms.Compose([
        transforms.Resize((272, 272)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    # Création du dataset
    dataset = CustomPKMNDataset(img_dir, data_file, transform=transform, class_to_idx=class_to_idx)

    # Split dataset
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Modèle
    model = FlexibleCNN(input_channels=3, num_classes=num_classes, dropout_rate=0.3)

    # Si déséquilibre des classes, on peut utiliser des poids de classe
    # Exemple (à adapter) :
    # class_weights = torch.tensor([1.0, 2.0, 1.5, ...], dtype=torch.float32)
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # Pour l'exemple, on utilise juste CrossEntropyLoss standard
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler cosinus
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler, num_epochs=num_epochs, patience=patience, device=device)
