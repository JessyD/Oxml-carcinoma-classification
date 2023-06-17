# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
# ---

from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.transforms import transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = self.data_dir, f"img_{self.labels_df.iloc[idx]['id']}.png"
        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[idx]['malignant'] + 1

        if self.transform:
            image = self.transform(image)

        return image, label

# Set random seed for reproducibility
torch.manual_seed(2)

# read in the data
data_dir = Path("../data/oxml-carinoma-classification")

data_df = pd.read_csv(data_dir / 'labels.cvs')

# Define any image transformations if needed
# transform = torchvision.transforms.Compose([...])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the custom dataset
dataset = CustomDataset(image_dir, labels_file, transform)

# Create the data loader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
