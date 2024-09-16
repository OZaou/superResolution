
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomCrop

class SRDataset(Dataset):
    def __init__(self, hr_dir, patch_size=32, scale_factor=2, transform=None):
        super(SRDataset, self).__init__()
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.transform = transform
        self.hr_image_filenames = sorted([
            os.path.join(hr_dir, x) for x in os.listdir(hr_dir) 
            if x.endswith(('png', 'jpg', 'jpeg'))
        ])

    def __len__(self):
        return len(self.hr_image_filenames)

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_image_filenames[idx]).convert('RGB')
        min_size = self.patch_size * self.scale_factor
        if hr_image.width < min_size or hr_image.height < min_size:
            hr_image = hr_image.resize((min_size, min_size), Image.BICUBIC)
        
        left = torch.randint(0, hr_image.width - self.patch_size * self.scale_factor + 1, (1,)).item()
        upper = torch.randint(0, hr_image.height - self.patch_size * self.scale_factor + 1, (1,)).item()
        hr_patch = hr_image.crop((
            left,
            upper,
            left + self.patch_size * self.scale_factor,
            upper + self.patch_size * self.scale_factor
        ))
        
        lr_patch = hr_patch.resize(
            (self.patch_size, self.patch_size),
            Image.BICUBIC
        )
        
 
        lr_patch_upsampled = lr_patch.resize(
            (self.patch_size * self.scale_factor, self.patch_size * self.scale_factor),
            Image.BICUBIC
        )
        
        if self.transform:
            lr_tensor = self.transform(lr_patch_upsampled)
            hr_tensor = self.transform(hr_patch)
        else:
            lr_tensor = ToTensor()(lr_patch_upsampled)
            hr_tensor = ToTensor()(hr_patch)
        
        return lr_tensor, hr_tensor
