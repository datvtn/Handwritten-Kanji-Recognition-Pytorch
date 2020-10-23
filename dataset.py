import torch
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset

import cv2
from tqdm import tqdm

with open("ETL9G_dataset/classes.txt") as f:
    classes= f.read().strip().split(' ')

class ETL9GDataset(Dataset):
    def __init__(self, path, transform= None):
        super().__init__()
        self.path= path
        with open(self.path) as f:
            self.tmp= f.read().strip().split('\n')
        self.img_paths= []
        self.labels= []
        print ("[INFO] Loading ", self.path)
        for v in tqdm(self.tmp):
            temp= v.split(' ')
            self.img_paths.append(temp[0])
            label= classes.index(temp[1])
            self.labels.append(label)
        self.transform= transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path= self.img_paths[idx]
        label= self.labels[idx]
        img= cv2.imread(img_path)
        if self.transform is not None:
            img= self.transform(img)
        return img, label
