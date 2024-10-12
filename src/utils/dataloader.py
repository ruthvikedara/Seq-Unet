import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
import numpy as np
import random
from PIL import Image

"""
Data was stored in the following format:

/data/patient_x/
    ├── images/
        ├── 1.jpg
        ├── 2.jpg
        └── ...
    └── masks/
        ├── 1.tif
        ├── 2.tif
        └── ...
"""

class SequentialData(Dataset):
    def __init__(self, home_directory, block_rate):
        self.home_directory = home_directory
        self.block = block_rate
        self.image_files = []
        self.mask_files = []
        self.prior_files = []
        self.transform_image = v2.Compose([
            v2.Grayscale(),
            v2.ToTensor(),
            v2.Normalize((0.0,), (1.0,)),
        ])
        self.transform_mask = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.0,), (1.0,)),
            v2.GaussianBlur(11, sigma=5),
            v2.Lambda(lambda x: (x > 0.5).float())
        ])
        self.transform_blur = v2.GaussianBlur(31, sigma=15)

        for patient_folder in os.listdir(self.home_directory):
            image_folder = os.path.join(self.home_directory, patient_folder, 'images')
            mask_folder = os.path.join(self.home_directory, patient_folder, 'masks')

            if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
                print(f"Skipping {patient_folder} due to missing image or mask folder.")
                continue

            # Sorted lists of image and mask files
            image_files_for_patient = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
            mask_files_for_patient = sorted(os.listdir(mask_folder), key=lambda x: int(os.path.splitext(x)[0]))

            # Check if the number of images and masks match
            if len(image_files_for_patient) != len(mask_files_for_patient):
                print(f"Warning: Mismatch in images and masks for {patient_folder}. Skipping this patient.")
                continue

            # Generate full paths for each image and mask, and store them in the respective lists
            self.image_files.extend([os.path.join(patient_folder, 'images', img) for img in image_files_for_patient])
            self.mask_files.extend([os.path.join(patient_folder, 'masks', mask) for mask in mask_files_for_patient])

            # Consider previous mask as prior file
            if len(mask_files_for_patient) > 1:
                self.prior_files.extend([os.path.join(patient_folder, 'masks', mask) for mask in mask_files_for_patient[:-1]])

        assert len(self.image_files) == len(self.mask_files), "Number of images and masks do not match."
        if self.prior_files:
            assert len(self.prior_files) == len(self.image_files) - 1, "Number of prior files is inconsistent."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mask_file = self.mask_files[idx]

        image_path = os.path.join(self.home_directory, image_file)
        mask_path = os.path.join(self.home_directory, mask_file)

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image_tensor = self.transform_image(image)
        mask_tensor = self.transform_mask(mask).squeeze()
        
        rand = random.random()
        if rand <= self.block:
            prior_file = self.prior_files[idx]
            prior_path = os.path.join(self.home_directory, prior_file)
            prior = np.array(Image.open(prior_path))

            prior_tensor = self.ransform_mask(prior)
            gmask = self.transform_blur(torch.clamp(prior_tensor + 0.1, 0,1))
            final_image = torch.stack((image_tensor, gmask*image_tensor), dim=0).squeeze()
        else:
            final_image = torch.stack((image_tensor, torch.zeros_like(image_tensor)), dim=0).squeeze()


        return final_image, mask_tensor, patient_id, image_number


def get_dataloaders(DATAPATH, batch_size, block_rate=0.01, train=80, val=10, test=10):
    dataset = SequentialData(DATAPATH, block_rate=block_rate)

    train_size = train*300
    val_size = val*300
    test_size = test*300

    train_dataset = Subset(dataset, range(train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, len(dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader
