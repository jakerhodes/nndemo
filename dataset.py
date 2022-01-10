import pandas as pd
from torch.utils.data import Dataset
import PIL

class CSVLoader(Dataset):
    def __init__(self, csv, transforms):

        self.csv = csv
        self.df = pd.read_csv(csv)
        self.label = self.df['label']
        self.path  = self.df['path']
        self.transforms = transforms 


    def __getitem__(self, index):
        label = self.label[index] 
        path  = self.path[index]
        img   = PIL.Image.open(path).convert('RGB')
        img   = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.label)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
