import wandb
from tqdm import tqdm
import torch

from diffusion import Diffusion


def train(
    model, diffusion: Diffusion, timesteps, device, data_loader, optimizer,
        epoch, loss_fn, use_wandb=False, ema=None):

    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        x_0 = batch["I"].to(device)

        t = torch.randint(
            0, timesteps, (x_0.shape[0],), dtype=torch.int64)
        noisy_image, noise = diffusion.forward_process(x_0, t)

        noisy_image = noisy_image.to(device)
        noise = noise.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(noisy_image, t.to(device))

        if use_wandb:

            if batch_idx == 0 and epoch % 5 == 0:
                wandb.log({"Train": [
                    wandb.Image(
                        torch.moveaxis(
                            noisy_image[0], 0, -1).cpu().detach().numpy()),
                    wandb.Image(
                        torch.moveaxis(
                            noise[0], 0, -1).cpu().detach().numpy()),
                    wandb.Image(
                        torch.moveaxis(
                            output[0, :3], 0, -1).cpu().detach().numpy())]},
                          step=epoch, commit=False)

        # Backward pass

        if loss_fn.__class__.__name__ == "HybridLoss":

            true_mean, true_log_var_clipped = diffusion.q_posterior(
                noisy_image, x_0, t)
            out_mean, out_var = diffusion.p(
                output[:, :3], output[:, 3:], noisy_image, t, learned_var=True)

            loss = loss_fn(
                noise, output[:, :3], x_0, t.to(device), true_mean,
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
