from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

from diffusion import Diffusion
from image_transforms import down_upsample


def train(
    model, diffusion: Diffusion, timesteps, device, data_loader, optimizer,
        epoch, loss_fn, ema=None, pred_type="all", condition="None",
        super_res=False, img_channels=3, drop_rate=0.2):

    model.train()
    epoch_loss = 0

    mask = None

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        if pred_type == "all":
            x_0 = batch["I"].to(device)
        elif pred_type == "mask":
            x_0 = batch["I"][:, img_channels:].to(device)
            if x_0.shape[1] != 1:  # one hot encoding
                x_0 = x_0 * 2 - 1
        else:
            x_0 = batch["I"][:, :img_channels].to(device)
            if condition == "mask":
                mask = batch["I"][:, img_channels:].to(device)
                drop_mask = (
                    torch.rand((mask.shape[0], 1, 1, 1)) > drop_rate).float()
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

        # Forward pass
        optimizer.zero_grad()

        if not super_res:
            output = model(noisy_image, t.to(device), mask=mask,
                           label_dist=label_dist)
        else:

            # downsampling
            low_res = down_upsample(x_0, img_channels)

            output = model(noisy_image, t.to(device), low_res=low_res)

        # Backward pass

        if loss_fn.__class__.__name__ == "HybridLoss":

            true_mean, true_log_var_clipped = diffusion.q_posterior(
                noisy_image, x_0, t)

            out_mean, out_var = diffusion.p(
                output[:, :x_0.shape[1]], output[:, x_0.shape[1]:],
                noisy_image, t, learned_var=True)

            loss = loss_fn(
                noise, output[:, :x_0.shape[1]], x_0, t.to(device), true_mean,
                true_log_var_clipped, out_mean, out_var)

        else:
            loss = loss_fn(output, noise)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        if ema:
            ema.step(epoch, model)

    epoch_loss /= len(data_loader)

    return epoch_loss
