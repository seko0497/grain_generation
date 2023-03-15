from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import einops


class GrainDataset(Dataset):

    def __init__(self, root_dir, img_size, image_idxs,
                 mask_one_hot=False, num=1000, train=True):

        self.root_dir = root_dir
        self.num = num
        self.train = train
        self.image_idxs = image_idxs
        self.image_size = img_size
        self.in_channels = ["intensity", "depth"]

        self.crop_scale = transforms.Compose([
            transforms.RandomCrop(img_size[0]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        self.images = [torch.cat(
            tuple([
                transforms.ToTensor()(
                    Image.open(f"{self.root_dir}/{i}_{feature}.png")
                ) for feature in self.in_channels
            ])
            + tuple(
                [transforms.ToTensor()(
                    Image.open(f"{self.root_dir}/{i}_target.png")
                )]
            ))
            for i in image_idxs]

    def __len__(self):

        if self.train:
            return self.num
        else:
            return len(self.image_idxs)

    def __getitem__(self, index):

        if self.train:

            index = np.random.randint(0, len(self.images))

            image = self.crop_scale(self.images[index])

        else:

            image = self.images[index]
            image = image[
                :,
                :image.shape[1] // self.image_size[0] * self.image_size[0],
                :image.shape[2] // self.image_size[0] * self.image_size[0]]

            p1 = image.shape[1] // self.image_size[0]
            p2 = image.shape[2] // self.image_size[0]

            image = einops.rearrange(
                    image,
                    "c (p1 h) (p2 w) -> (p1 p2) c h w",
                    p1=p1,
                    p2=p2
                )
            image = image * 2 - 1

        return {"I": image}


# DEBUG
grain_dataset = GrainDataset("data/grains_txt", (256, 256), [1, 4, 5], num=16,
                             train=True)
grain_dataloader = DataLoader(grain_dataset, batch_size=1)

for batch in grain_dataloader:
    # pass
    print(batch["I"].shape)
    print(torch.unique(batch["I"][0][0]))
    __, ax = plt.subplots(3)

    ax[0].imshow(batch["I"][0][0], cmap="viridis", vmin=-1, vmax=1)
    ax[1].imshow(batch["I"][0][1], cmap="viridis", vmin=-1, vmax=1)
    ax[2].imshow(batch["I"][0][2], cmap="viridis", vmin=-1, vmax=1)
    plt.show()
