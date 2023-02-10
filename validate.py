import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from dataset_wear import WearDataset
import wandb


class Validation():

    def __init__(
            self, real_samples_dir, raw_img_size, image_size, num_workers,
            use_wandb=False):

        self.fid = FrechetInceptionDistance(
            normalize=True, reset_real_features=False)
        self.real_samples_dir = real_samples_dir
        self.raw_img_size = raw_img_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.use_wandb = use_wandb

        self.fit_real_samples()

    def fit_real_samples(self):

        fid_loader = DataLoader(WearDataset(
            self.real_samples_dir,
            raw_img_size=self.raw_img_size,
            img_size=self.image_size
        ), batch_size=115,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            shuffle=False)
        for batch in fid_loader:
            real_samples = (batch["I"] + 1) / 2

        self.fid.update(real_samples, real=True)

    def validate(self, samples):

        self.fid.reset()
        self.fid.update(samples.cpu(), real=False)
        fid = self.fid.compute()

        return fid
