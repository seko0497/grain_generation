from tqdm import tqdm
import torch

from diffusion import Diffusion


def train(
    model, diffusion: Diffusion, timesteps, device, data_loader, optimizer,
        epoch, loss_fn, ema=None, pred_type="all", img_channels=3,
        guidance_scale=0.2):

    model.train()
    epoch_loss = 0

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

        # Forward pass
        optimizer.zero_grad()
        output = model(noisy_image, t.to(device), mask=mask)

        # classifier free guidance
        if mask is not None:
            zeros = torch.zeros_like(mask).to(device)
            output_zero = model(noisy_image, t.to(device), zeros)

            output[:, :img_channels] = (output_zero[:, :img_channels] +
                                        guidance_scale * (
                                            output[:, :img_channels] -
                                            output_zero[:, :img_channels]))

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
