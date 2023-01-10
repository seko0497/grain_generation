from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class WearDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.files = [
            os.path.splitext(
                os.path.split(f"{self.root_dir}/target/1/{filename}")[-1])[0]
            for filename in os.listdir(f"{self.root_dir}/target/1")]

        self.inputs = [f"{self.root_dir}/features/1/{filename}.png"
                       for filename in self.files]

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        inp = Image.open(f"{self.inputs[index]}")
        inp = self.transforms(inp)

        return {"I": inp}


# DEBUG
# wear_dataset = WearDataset(
#     "data/RT100U_processed/train")
# wear_dataloader = DataLoader(wear_dataset, batch_size=4)
# for batch in wear_dataloader:
#     # pass
#     image = torch.moveaxis(batch["I"][0], 0, -1).numpy()
#     image = (image + 1) / 2
#     plt.imshow(image)
#     plt.show()
