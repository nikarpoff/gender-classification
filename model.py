import torch

from torch import nn


class GenderClassifier(nn.Module):
    def __init__(self, pretrained_model: nn.Module) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model
        self.flatten = nn.Flatten()
        self.perceptron = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1, bias=True),
            nn.Sigmoid(),
        )
        # self.perceptron = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=64, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=1, bias=True),
        #     nn.Sigmoid(),
        # )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Extracted by ViT image features with shape [1, 512].
        embeddings = self.pretrained_model.encode_image(image).float()

        # Squeeze features to vector
        x = self.flatten(embeddings)

        return self.perceptron(x)
