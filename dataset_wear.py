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


class WearDataset(Dataset):

    def __init__(self, root_dir, raw_img_size, img_size, mask_one_hot=False,
                 num_classes=3, label_dist=False):

        self.root_dir = root_dir
        self.one_hot = mask_one_hot
        self.num_classes = num_classes
        self.label_dist = label_dist
        self.label_dist_scaler_fitted = False

        self.files = [
            os.path.splitext(
                os.path.split(f"{self.root_dir}/target/1/{filename}")[-1])[0]
            for filename in os.listdir(f"{self.root_dir}/target/1")]

        self.inputs = [f"{self.root_dir}/features/1/{filename}.png"
                       for filename in self.files]

        self.targets = [f"{self.root_dir}/target/1/{filename}.npy"
                        for filename in self.files]

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(raw_img_size[0]),
            transforms.Resize(img_size),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

        self.target_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(raw_img_size[0]),
            transforms.Resize(
                img_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])

        # if self.label_dist:
        #     self.label_dist_scaler = MinMaxScaler()
        #     self.fit_scaler()
        #     self.label_dist_scaler_fitted = True

    def fit_scaler(self):

        for i in range(self.__len__()):

            self.label_dist_scaler.partial_fit(self.__getitem__(i)["L"][None])

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        inp = Image.open(f"{self.inputs[index]}")
        inp = self.transforms(inp)

        trg = np.load(self.targets[index])

        trg = self.target_transforms(trg)

        if self.label_dist:
            counter = Counter({cl: 0 for cl in range(self.num_classes)})
            counter.update(np.array(trg).flatten())
            label_dist = np.array(
                [dict(counter)[cl + 1] for cl in range(self.num_classes - 1)],
                dtype=float)
            label_dist /= label_dist.sum()
            # if self.label_dist_scaler_fitted:
            #     label_dist = self.label_dist_scaler.transform(
            #         np.array(label_dist)[None])[0]

        if self.one_hot:
            trg = torch.nn.functional.one_hot(trg[0].long(),
                                              num_classes=self.num_classes)
            trg = torch.moveaxis(trg, -1, 0)
        else:
            trg = torch.Tensor(trg / (self.num_classes - 1))
        trg = trg * 2 - 1

        if self.label_dist:
            return {"I": torch.cat((inp, trg)).float(),
                    "L": torch.Tensor(label_dist)}
        else:
            return {"I": torch.cat((inp, trg)).float()}


# def calc_min_max():

#     wear_dataset = WearDataset(
#         "data/RT100U_processed/train", (448, 576), (256, 256), norm=None)
#     wear_dataloader = DataLoader(wear_dataset, batch_size=128)
#     scaler = MinMaxScaler()
#     for batch in wear_dataloader:
#         scaler.partial_fit(batch["L"])

#     print(scaler.data_max_, scaler.data_min_)


# calc_min_max()

# # DEBUG

# norm = ([0.97099304, 0., 0.], [0.99629211, 0.0241394, 0.02012634])

# wear_dataset = WearDataset(
#     "data/RT100U_processed/train", (448, 576), (256, 256),
#     mask_one_hot=False, norm=norm)
# wear_dataloader = DataLoader(wear_dataset, batch_size=4)
# for batch in wear_dataloader:
#     print(batch["L"])
#     # image = torch.moveaxis(batch["I"][0, :3], 0, -1).numpy()
#     # trg = torch.moveaxis(batch["I"][0, 3:], 0, -1).numpy()
#     # image = (image + 1) / 2
#     # plt.imshow(trg)
#     # plt.show()
