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
            transforms.CenterCrop(raw_img_size[0]),
            transforms.Resize(img_size),
        ])

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        inp = Image.open(f"{self.inputs[index]}")
        inp = self.transforms(inp)

        trg = np.load(self.targets[index])

        trg = torch.nn.functional.one_hot(
            torch.LongTensor(trg), self.num_classes)

        trg = self.target_transforms(torch.moveaxis(trg, -1, 0))

        if self.label_dist:
            label_dist = torch.zeros((self.num_classes,))
            trg_cls = torch.argmax(trg, dim=0)
            if torch.unique(trg_cls).shape[0] == self.num_classes:
                label_dist[-1] = 1.0
            else:
                label_dist[torch.unique(trg_cls).int()[-1].item() - 1] = 1.0

        if not self.one_hot:
            trg = torch.argmax(trg, dim=0, keepdim=True)
            trg = torch.Tensor(trg / (self.num_classes - 1))
        trg = trg * 2 - 1

        if self.label_dist:
            return {"I": torch.cat((inp, trg)).float(),
                    "L": torch.Tensor(label_dist).long()}
        else:
            return {"I": torch.cat((inp, trg)).float()}


# DEBUG
# wear_dataset = WearDataset(
#     "data/RT100U_processed/train", (448, 576), (256, 256),
#     mask_one_hot=False, label_dist=True)
# wear_dataloader = DataLoader(wear_dataset, batch_size=4)
# for batch in wear_dataloader:
#     # pass
#     # print(batch["L"])
#     for image, label_dist in zip(batch["I"], batch["L"]):
#         trg = torch.moveaxis(image[-1:], 0, -1).numpy()
#         trg = (trg + 1) / 2
#         print(np.unique(trg))
#         print(label_dist)
#         plt.imshow(trg)
#         plt.show()
