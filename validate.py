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

    def valid_nll(self, valid_loader: DataLoader, model, diffusion: Diffusion,
                  steps, device):

        timesteps = torch.round(
                        torch.linspace(
                            0, diffusion.timesteps - 1, steps)).int()
        loss_fn = HybridLoss()

        all_bpd = []

        for batch_idx, batch in enumerate(tqdm(valid_loader)):

            vb = []

            x_0 = batch["I"].to(device)

            for t in tqdm(reversed(timesteps), total=len(timesteps)):

                t = torch.full((x_0.shape[0],), t)

                noisy_image, noise = diffusion.forward_process(x_0, t)

                noisy_image = noisy_image.to(device)
                noise = noise.to(device)

                with torch.no_grad():

                    output = model(noisy_image, t.to(device))

                    true_mean, true_log_var_clipped = diffusion.q_posterior(
                        noisy_image, x_0, t)
                    out_mean, out_var = diffusion.p(
                        output[:, :3], output[:, 3:], noisy_image, t,
                        learned_var=True)

                    vlb = loss_fn.vlb_loss(true_mean, true_log_var_clipped,
                                           out_mean, out_var, x_0,
                                           t.to(device))
                vb.append(vlb)

            q_mean, _, q_log_variance = diffusion.q(
                x_0, torch.full((x_0.shape[0],), diffusion.timesteps - 1))
            kl_prior = normal_kl(
                q_mean, q_log_variance,
                torch.Tensor([0.0]).to(device), torch.Tensor([0.0]).to(device))
            prior_bpd = kl_prior.mean(
                dim=list(range(1, len(kl_prior.shape)))) / np.log(2.0)

            vb = torch.stack(vb, dim=1)
            total_bpd = vb.sum(dim=1) + prior_bpd
            total_bpd = total_bpd.mean()
            all_bpd.append(total_bpd.item())

        return np.mean(all_bpd)

    def valid_hybrid_loss(
            self, model, data_loader, device, diffusion, timesteps):

        epoch_loss = 0
        model.eval()

        for batch_idx, batch in enumerate(tqdm(data_loader)):

            x_0 = batch["I"].to(device)

            t = torch.randint(
                0, timesteps, (x_0.shape[0],), dtype=torch.int64)
            noisy_image, noise = diffusion.forward_process(x_0, t)

            noisy_image = noisy_image.to(device)
            noise = noise.to(device)

            output = model(noisy_image, t.to(device))

            true_mean, true_log_var_clipped = diffusion.q_posterior(
                noisy_image, x_0, t)
            out_mean, out_var = diffusion.p(
                output[:, :3], output[:, 3:], noisy_image, t, learned_var=True)

            loss_fn = HybridLoss()
            loss = loss_fn(
                noise, output[:, :3], x_0, t.to(device), true_mean,
                true_log_var_clipped, out_mean, out_var)

            epoch_loss += loss.item()

        model.train()
        return epoch_loss / len(data_loader)

    def valid_fid(self, samples):

        self.fid.reset()
        self.fid.update(samples.cpu(), real=False)
        fid = self.fid.compute()

        return fid
