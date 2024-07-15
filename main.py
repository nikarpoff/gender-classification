import torch

from torch import nn
from teacher import Teacher
from model import GenderClassifier
from utils import define_device

from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold

from dataPreparing.loaders import ModelsLoader, DatasetsLoader


# Define computation device.
device = define_device()

# load ViT (from CLIP).
modelsLoader = ModelsLoader(device=device)
pretrained_model, preprocess = modelsLoader.load_vit_b_16()

# Make pretrained models not trainable.
for param in pretrained_model.parameters():
    param.requires_grad = False

# Init models.
model = GenderClassifier(pretrained_model)

# Load data.
datasets_loader = DatasetsLoader(device, preprocess)

# dataset_name = "custom"
dataset_name = "celeba"

if dataset_name == "celeba":
    train_data, test_data = datasets_loader.load_celeba_dataset(root="data")
    dataset = ConcatDataset([train_data, test_data])
elif dataset_name == "adience":
    pass
else:
    dataset = datasets_loader.load_custom_dataset_without_path()

# Use k-fold with splitting in 5 folds.
splits = 5
k_fold = KFold(n_splits=splits, shuffle=True)

# Choose optimizers, hyperparams and loss function
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
loss_fn = nn.BCELoss()

batch_size = 64
epochs = 1

model_name = 'gc-simple-dnn-2'

teacher = Teacher(device=device,
                  model_classifier=model,
                  model_name=model_name,
                  batch_size=batch_size,
                  loss_function=loss_fn,
                  optimizer=optimizer,
                  )

print("\nStart training...\n")

for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):
    # Print
    print(f'\nFold -> {fold} ---------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, sampler=train_subsampler)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, sampler=test_subsampler)

    for epoch in range(epochs):
        # teacher.test(test_dataloader)
        print(f"\nEpoch {epoch + 1}/{epochs} -----------------")
        teacher.train(train_dataloader)
        torch.save(model, f'models/{model_name}-{epoch}.pth')
        # print("\nStart testing...\n")
        # teacher.test(test_dataloader)
    teacher.test(test_dataloader)
torch.save(model, f'models/{model_name}.pth')
