import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Un bloc résiduel simple :
    Conv2d -> BN -> GELU -> Conv2d -> BN
    Ajout d'une connexion skip. Les dimensions d'entrée et de sortie sont identiques.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Si in_channels != out_channels, on projette pour adapter les dimensions
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = F.gelu(out)
        return out

class FlexibleCNN(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_classes=18,
        dropout_rate=0.3
    ):
        super(FlexibleCNN, self).__init__()
        # On peut utiliser plus de couches convolutives, par exemple [64, 128, 256, 512]
        # Première couche avec un kernel plus large (7x7) pour capturer des motifs plus globaux
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Blocs convolutifs avec résidus
        self.block1 = ResidualBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.block2 = ResidualBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.block3 = ResidualBlock(256, 512, kernel_size=3, stride=1, padding=1)

        # Pooling adaptatif pour réduire la dimension spatiale à 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Couches fully connected supplémentaires pour réduire la sur-optimisation
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        print("1")
        x = self.initial_conv(x)
        print("2")
        x = self.block1(x)
        print("3")
        x = self.block2(x)
        print("4")
        print(f"Shape before block3: {x.shape}")
        x = self.block3(x)
        print("5")
        x = self.global_pool(x)
        print("6")
        x = self.fc(x)
        print("7")
        return x

# Exemple d'instanciation du modèle
if __name__ == "__main__":
    model = FlexibleCNN(
        input_channels=3,  # RGB
        num_classes=18,    # 18 types de Pokémon
        dropout_rate=0.3   # Taux de dropout un peu plus faible
    )
    print(model)
