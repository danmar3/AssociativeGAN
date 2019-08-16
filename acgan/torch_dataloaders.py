import os
import random
import attr
import torch
import pandas as pd

from torch.utils import data
from torchvision import transforms as trfs
from torchvision.datasets import ImageFolder

from PIL import Image

from glob import glob


from os.path import split, join
from tqdm.auto import tqdm
import numpy as np

@attr.attrs
class CelebA(data.Dataset):
    dataroot = attr.ib()
    attr_path = attr.ib()
    labels = attr.ib(['Chubby'])
    transforms = attr.ib(None)
    num_samples = attr.ib(5000)


    def __attrs_post_init__(self):
        self.prep_attributes()
        self.files = glob(join(self.dataroot, "img_align_celeba", "*"))
        #self.num_files = len(self.files)
        #self.num_images = self.num_files
        self.num_images = self.num_files = self.num_samples

        assert self.num_files <= len(self.attr_df)

        self.fid_to_path = {split(f)[-1]: f
                            for f in self.files}


        self.train_ixes = np.random.choice(self.attr_df.index, self.num_samples,
                                           replace=True)
        self.test_ixes = np.random.choice([i for i in self.attr_df.index
                                           if i not in self.train_ixes],
                                        1000, replace=False)

        self.transforms = trfs.Compose([
                                        trfs.Resize(64),
                                        trfs.CenterCrop(64),
                                        trfs.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])


    def prep_attributes(self):
        self.attr_df = pd.read_csv(self.attr_path)
        self.attr_df.replace(-1, 0, inplace=True)
        self.attr_df = self.attr_df.set_index('image_id')


    def __getitem__(self, item):
        ix = self.train_ixes[item]
        path = self.fid_to_path[ix]
        image = Image.open(path)

        labels = self.attr_df.loc[ix][self.labels].values
        labels = torch.from_numpy(labels).float()
        return self.transforms(image), torch.FloatTensor(labels)

    def __len__(self):
        return self.num_images

