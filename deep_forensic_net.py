import torch
import torch.nn as nn
import torchvision.models as models


class DeepForensicClassifier(nn.Module):
    """
    Deep Convolutional Feature Extractor using a ResNet-50 backbone
    modified for single-channel high-frequency sensor noise classification.
    """

    def __init__(self, num_classes=7):
        super(DeepForensicClassifier, self).__init__()

        # Initialize a production-grade ResNet-50 model with default weights
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 1 input channel (grayscale)
        # instead of 3 (RGB), preserving the native 512x512 spatial dimensions.
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )

        # Isolate the feature extraction layers, dropping the standard pooling and FC layers
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # Extract the dimensional depth of the latent space projection
        self.latent_dim = backbone.fc.in_features

        # Construct a regularized classification head to prevent patch overfitting
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Executes the forward pass to map input tensors to class logits.
        """
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits