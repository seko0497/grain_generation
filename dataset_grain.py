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

    def __init__(self, root_dir, channel_names, image_idxs,
                 patch_size, img_size, mask_one_hot=False,
                 num_classes=2, num=500, train=True):

        self.num = num
        self.train = train
        self.image_idxs = image_idxs
        self.patch_size = patch_size
        self.image_size = img_size
        self.channel_names = channel_names
        self.mask_one_hot = mask_one_hot
        self.num_classes = num_classes

        self.crop_scale = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        self.images = [torch.cat(
            tuple([
                transforms.ToTensor()(
                    Image.open(f"{root_dir}/{i}_{feature}.png")
                ) for feature in channel_names
            ])
            + tuple(
                [transforms.ToTensor()(
                    Image.open(f"{root_dir}/{i}_target.png")
                )]
            ))
            for i in image_idxs]

        if not train:
            image = self.images[0]
            image = image[
                :,
                :image.shape[1] // patch_size * patch_size,
                :image.shape[2] // patch_size * patch_size]

            p1 = image.shape[1] // patch_size
            p2 = image.shape[2] // patch_size

            image = einops.rearrange(
                    image,
                    "c (p1 h) (p2 w) -> (p1 p2) c h w",
                    p1=p1,
                    p2=p2)
            self.images = image

    def __len__(self):

        if self.train:
            return self.num
        else:
            return len(self.images)

    def __getitem__(self, index):

        if self.train:

            index = np.random.randint(0, len(self.images))
            image = self.crop_scale(self.images[index])

        else:

            image = self.images[index]
            image = image * 2 - 1

        if self.patch_size != self.image_size[0]:
            image = torch.nn.functional.interpolate(
                image[None], self.image_size, mode="area")[0]
            image[2] = (image[2] == 1.0)
            image[2] = image[2] * 2 - 1

        if self.mask_one_hot:
            inp = image[:len(self.channel_names)]
            trg = image[len(self.channel_names):]
            trg = (trg + 1) / 2
            trg = torch.nn.functional.one_hot(trg[0].long(),
                                              self.num_classes)
            trg = trg * 2 - 1
            trg = torch.moveaxis(trg, -1, 0)
            image = torch.cat((inp, trg))

        return {"I": image}


# # DEBUG
grain_dataset = GrainDataset(
    "data/grains_txt", channel_names=["intensity", "depth"], image_idxs=[9],
    patch_size=256, img_size=(256, 256), mask_one_hot=False, train=True)


grain_dataloader = DataLoader(grain_dataset, batch_size=1)

for i, batch in enumerate(grain_dataloader):

    # low_res = down_upsample(batch["I"], 2)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(batch["I"][0, 0], vmax=1, vmin=-1)
    axes[1].imshow(batch["I"][0, 1], vmax=1, vmin=-1)
    axes[2].imshow(batch["I"][0, 2], vmax=1, vmin=-1)
    # axes[0].imshow(low_res[0, 0], vmax=1, vmin=-1)
    # axes[1].imshow(low_res[0, 1], vmax=1, vmin=-1)
    # axes[2].imshow(low_res[0, 2], vmax=1, vmin=-1)
    plt.show()
