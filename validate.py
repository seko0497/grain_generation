import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_wear import WearDataset
import wandb

from losses import HybridLoss, normal_kl
from diffusion import Diffusion


class Validation():

    def __init__(self, img_channels=3):

        self.fid = FrechetInceptionDistance(
            normalize=True, reset_real_features=False)
        self.img_channels = img_channels

    def fit_real_samples(self, dataloader):

        for batch in dataloader:
            real_samples = (batch["I"][:, :self.img_channels] + 1) / 2
            self.fid.update(real_samples, real=True)

    def valid_hybrid_loss(
            self, model, data_loader, device, diffusion, timesteps,
            pred_type="all", img_channels=3, guidance_scale=0.2):

        epoch_loss = 0
        model.eval()

        mask = None

        for batch_idx, batch in enumerate(tqdm(data_loader)):

            if pred_type == "all":
                x_0 = batch["I"].to(device)
            elif pred_type == "mask":
                x_0 = batch["I"][:, img_channels:].to(device)
            else:
                x_0 = batch["I"][:, :img_channels].to(device)
                mask = batch["I"][:, img_channels:].to(device)

            t = torch.randint(
                0, timesteps, (x_0.shape[0],), dtype=torch.int64)
            noisy_image, noise = diffusion.forward_process(x_0, t)

            noisy_image = noisy_image.to(device)
            noise = noise.to(device)

            with torch.no_grad():
                output = model(noisy_image, t.to(device), mask=mask)

            # classifier free guidance
            if mask is not None:
                zeros = torch.zeros_like(mask).to(device)
                output_zero = model(noisy_image, t.to(device), zeros)

                output[:, :img_channels] = (output_zero[:, :img_channels] +
                                            guidance_scale * (
                                                output[:, :img_channels] -
                                                output_zero[:, :img_channels]))

            true_mean, true_log_var_clipped = diffusion.q_posterior(
                noisy_image, x_0, t)
            out_mean, out_var = diffusion.p(
                output[:, :x_0.shape[1]], output[:, x_0.shape[1]:],
                noisy_image, t, learned_var=True)

            loss_fn = HybridLoss()
            loss = loss_fn(
                noise, output[:, :x_0.shape[1]], x_0, t.to(device), true_mean,
                true_log_var_clipped, out_mean, out_var)

            epoch_loss += loss.item()

        model.train()
        return epoch_loss / len(data_loader)

    def valid_fid(self, samples):

        self.fid.reset()
        self.fid.update(samples.cpu(), real=False)
        fid = self.fid.compute()

        return fid

    def generate_samples(self, condition, pred_type, img_channels, num_classes,
                         diffusion, sampling_steps, batch_size, model, device,
                         valid_loader=None):

        samples = []
        sample_masks = []

        # sample
        if condition == "mask":

            for batch in valid_loader:

                sample_mask = batch["I"][:, 3:]
                sample_masks.append(sample_mask)
                sample_batch = diffusion.sample(model,
                                                sample_mask.shape[0],
                                                mask=sample_mask.to(
                                                    device),
                                                sampling_steps=sampling_steps)
                samples.append(sample_batch)
            sample_masks = torch.cat(sample_masks)

        else:

            for _ in range(2 // batch_size):
                samples.append(diffusion.sample(model, batch_size,
                                                sampling_steps=sampling_steps))
        samples = torch.cat(samples)

        # split images and masks
        if pred_type == "all":
            sample_masks = samples[:, img_channels:]
            samples = samples[:, :img_channels]

        # convert masks to have one channel
        if sample_masks.shape[1] == num_classes:
            sample_masks = torch.argmax(
                            sample_masks, dim=1).float()
            sample_masks = sample_masks / (num_classes - 1)
        else:
            sample_masks = sample_masks[:, 0]

        return samples, sample_masks
