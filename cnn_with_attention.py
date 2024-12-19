import torch.nn as nn
from attention_mechanisms.cbam import CBAM # Check : https://github.com/changzy00/pytorch-attention from changzy00

class FlexibleCNNWithAttention(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_classes=18,
        conv_layers=[64, 128, 256],
        kernel_size=3,
        dropout_rate=0.5
    ):
        super(FlexibleCNNWithAttention, self).__init__()
        self.layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()  # Attention modules for each conv layer
        self.num_classes = num_classes

        # Define convolutional layers with CBAM attention
        prev_channels = input_channels
        for out_channels in conv_layers:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=prev_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            # Add CBAM for each convolutional block
            self.attention_layers.append(CBAM(out_channels))
            prev_channels = out_channels

        # Fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(prev_channels, num_classes)
        )

    def forward(self, x):
        for layer, attention in zip(self.layers, self.attention_layers):
            x = layer(x)
            x = attention(x)  # Apply attention after each convolutional block
        x = self.global_pool(x)
        x = self.fc(x)
        return x

# Example instantiation of the model
model = FlexibleCNNWithAttention(
    input_channels=3,  # RGB images
    num_classes=18,  # Assuming 18 Pokémon types
    conv_layers=[64, 128, 256],  # Number of filters for each layer
    kernel_size=3,  # Size of the convolutional kernel
    dropout_rate=0.5  # Dropout rate
)

# Print model summary
print(model)
