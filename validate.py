from collections import Counter
from matplotlib import cm
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

    def fit_real_samples(self, dataloader, fit_masks=False):

        for batch in dataloader:
            if fit_masks:
                masks = batch["I"][:, self.img_channels:]
                if masks.shape[1] != 1:
                    masks_argmax = torch.argmax(
                            masks, dim=1, keepdim=True).float()
                    masks = masks_argmax / (masks.shape[1] - 1)
                else:
                    masks = (masks + 1) / 2
                masks = masks[:, 0]
                cmap = cm.get_cmap("viridis")
                cm_masks = []
                for mask in masks:
                    cm_mask = cmap(np.array(mask))[:, :, :3]
                    cm_masks.append(
                        torch.moveaxis(torch.Tensor(cm_mask), -1, 0))
                real_samples = torch.stack(cm_masks)

            else:
                real_samples = (batch["I"][:, :self.img_channels] + 1) / 2

            self.fid.update(real_samples, real=True)

    def valid_hybrid_loss(
            self, model, data_loader, device, diffusion, timesteps,
            pred_type="all", img_channels=3, drop_rate=0.2):

        epoch_loss = 0
        mask = None

        with torch.no_grad():
            model.eval()

            for batch_idx, batch in enumerate(tqdm(data_loader)):

                if pred_type == "all":
                    x_0 = batch["I"].to(device)
                elif pred_type == "mask":
                    x_0 = batch["I"][:, img_channels:].to(device)
                else:
                    x_0 = batch["I"][:, :img_channels].to(device)
                    mask = batch["I"][:, img_channels:].to(device)
                    drop_mask = (
                        torch.rand(
                            (mask.shape[0], 1, 1, 1)) > drop_rate).float()
                    mask *= drop_mask.to(device)

                if "L" in batch.keys():
                    label_dist = batch["L"].to(device)
                    if torch.rand(1) <= drop_rate:
                        label_dist = None
                else:
                    label_dist = None

                t = torch.randint(
                    0, timesteps, (x_0.shape[0],), dtype=torch.int64)
                noisy_image, noise = diffusion.forward_process(x_0, t)

                noisy_image = noisy_image.to(device)
                noise = noise.to(device)

                output = model(noisy_image, t.to(device), mask=mask,
                               label_dist=label_dist)

                true_mean, true_log_var_clipped = diffusion.q_posterior(
                    noisy_image, x_0, t)
                out_mean, out_var = diffusion.p(
                    output[:, :x_0.shape[1]], output[:, x_0.shape[1]:],
                    noisy_image, t, learned_var=True, pred_type=pred_type,
                    img_channels=img_channels)

                loss_fn = HybridLoss()
                loss = loss_fn(
                    noise, output[:, :x_0.shape[1]], x_0, t.to(device),
                    true_mean, true_log_var_clipped, out_mean, out_var)

                epoch_loss += loss.item()

        model.train()
        return epoch_loss / len(data_loader)

    def valid_fid(self, samples, masks=False):

        if masks:
            fid_masks = []
            cmap = cm.get_cmap("viridis")
            for mask in samples:
                mask = cmap(np.array(mask))[:, :, :3]
                fid_masks.append(torch.moveaxis(torch.Tensor(mask), -1, 0))
            fid_masks = torch.stack(fid_masks)

            samples = fid_masks

        self.fid.reset()
        self.fid.update(samples.cpu(), real=False)
        fid = self.fid.compute()

        return fid

    def generate_samples(self, condition, pred_type, img_channels, num_classes,
                         diffusion, sampling_steps, batch_size, model, device,
                         valid_loader=None, guidance_scale=0.2):

        samples = []
        sample_masks = []
        label_dists = []

        # sample
        if condition == "mask":

            for batch in valid_loader:

                sample_mask = batch["I"][:, img_channels:]
                sample_masks.append(sample_mask)
                sample_batch = diffusion.sample(model,
                                                sample_mask.shape[0],
                                                mask=sample_mask.to(
                                                    device),
                                                label_dist=None,
                                                sampling_steps=sampling_steps,
                                                pred_type=pred_type,
                                                img_channels=img_channels,
                                                guidance_scale=guidance_scale)
                samples.append(sample_batch)
            sample_masks = torch.cat(sample_masks)

        else:

            for _ in range(128 // batch_size):
                if condition == "label_dist":
                    label_dist = torch.rand(
                        batch_size, num_classes - 1).to(device)
                    label_dists.append(label_dist)
                else:
                    label_dist = None
                samples.append(diffusion.sample(model, batch_size,
                                                mask=None,
                                                label_dist=label_dist,
                                                sampling_steps=sampling_steps,
                                                pred_type=pred_type,
                                                img_channels=img_channels,
                                                guidance_scale=guidance_scale))
        samples = torch.cat(samples)

        # split images and masks
        if pred_type == "all":
            sample_masks = samples[:, img_channels:]
            samples = samples[:, :img_channels]
        elif pred_type == "mask":
            sample_masks = samples
            samples = []

        # convert masks to have one channel
        if sample_masks.shape[1] == num_classes:
            sample_masks = torch.argmax(
                            sample_masks, dim=1).float()
            sample_masks = sample_masks / (num_classes - 1)
        else:
            sample_masks = sample_masks[:, 0]

        if label_dists != []:
            label_dists = torch.cat(label_dists)

        return samples, sample_masks, label_dists

    def label_dist_rmse(self, pred_masks, label_dists, scaler):

        num_classes = label_dists.shape[1]
        pred_label_dists = []
        for mask in pred_masks:

            counter = Counter({cl: 0 for cl in range(num_classes)})
            counter.update(np.array(mask.cpu().detach()).flatten())
            label_dist = np.array(
                [dict(counter)[cl] for cl in range(num_classes)],
                dtype=float)
            label_dist /= label_dist.sum()
            label_dist = scaler.transform(label_dist[None])[0]
            pred_label_dists.append(torch.Tensor(label_dist))
        pred_label_dists = torch.stack(pred_label_dists)

        return torch.sqrt(
            torch.nn.functional.mse_loss(
                pred_label_dists.to(label_dists.device), label_dists))
