import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        self.h5_path = h5_file_path
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.transform = transform
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def close(self):
        self.h5_file.close()