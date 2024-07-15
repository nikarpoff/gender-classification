import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image


class CustomFacesDataset(Dataset):
    def __init__(self, dataframe, base_path: str, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.target_transform = target_transform
        self.base_path = base_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get data from dataframe.
        row = self.dataframe.iloc[idx]
        image_path = row['path']
        gender = int(row['gender'])

        # Open image.
        image = Image.open(f"{self.base_path}/{image_path}").convert('RGB')

        # Cast gender to tensor
        gender = torch.tensor(gender, dtype=torch.long)

        # Apply transform.
        if self.transform:
            image = self.transform(image)

        target = (gender.unsqueeze(-1).type(torch.float), image_path)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target


class CelebAWithFilename(datasets.CelebA):
    def __getitem__(self, index):
        # Get image and target labels.
        img, target = super().__getitem__(index)
        # Get filename.
        filename = self.filename[index]
        return img, (target[20].unsqueeze(0).float(), filename)
