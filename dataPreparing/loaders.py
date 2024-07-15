import torch
import clip
import pandas as pd

from PIL import Image
from torchvision import datasets

from .dataset import CustomFacesDataset
from .dataset import CelebAWithFilename


class DatasetsLoader:
    def __init__(self, device, images_preprocess):
        self.device = device
        self.preprocess = images_preprocess

    def load_celeba_dataset(self, root) -> (datasets, datasets):
        # Download CelebA as PIL images with attributes
        # Download training data from open datasets.
        training_data = datasets.CelebA(
            root=root,
            split="train",
            download=True,
            transform=self.preprocess_celeba_image,
            target_transform=self.preprocess_celeba_target,
        )

        return training_data, self.load_test_celeba_dataset(root)

    def load_test_celeba_dataset(self, root) -> datasets:
        # Download test data from open datasets.
        return datasets.CelebA(
            root=root,
            split="test",
            download=True,
            transform=self.preprocess_celeba_image,
            target_transform=self.preprocess_celeba_target,
        )

    def load_test_celeba_images(self, root) -> datasets:
        # Download test data from open datasets.
        return datasets.CelebA(
            root=root,
            split="test",
            download=True,
            target_transform=self.preprocess_celeba_target,
        )

    def load_test_celeba_with_path(self, root) -> datasets:
        return CelebAWithFilename(
            root=root,
            split="test",
            download=True,
            transform=self.preprocess_celeba_image,
        )

    def preprocess_celeba_image(self, image: Image.Image):
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def preprocess_celeba_target(self, target: torch.Tensor):
        # Get from target field "Male".
        return target[20].unsqueeze(0).to(self.device).float()

    def preprocess_custom_target(self, target: torch.Tensor):
        # Get from target field "Male"
        return target[0].to(self.device).float()

    def load_custom_dataset(self) -> datasets:
        # Read csv.
        base_path = './data/faces-dataset/'
        df = pd.read_csv(f'{base_path}/data.csv')

        # return dataset
        return CustomFacesDataset(df, transform=self.preprocess, base_path=base_path)

    def load_cafe_dataset(self) -> datasets:
        # Read csv.
        base_path = './data/cafe-faces'
        df = pd.read_csv(f'{base_path}/data.csv')

        # return dataset
        return CustomFacesDataset(df, transform=self.preprocess, base_path=base_path)

    def load_custom_dataset_without_path(self) -> datasets:
        # Read csv.
        base_path = './data/faces-dataset/'
        df = pd.read_csv(f'{base_path}/data.csv')

        # return dataset
        return CustomFacesDataset(df, transform=self.preprocess,
                                  target_transform=self.preprocess_custom_target, base_path=base_path)


class ModelsLoader:
    def __init__(self, device):
        self.device = device

    def load_vit_b_16(self):
        return clip.load(name="ViT-B/16", device=self.device)

    def load_local(self, path):
        return torch.load(path, map_location=self.device)
