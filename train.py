import wandb
from tqdm import tqdm
import torch


def train(
    model, diffusion, timesteps, device, data_loader, optimizer,
        epoch, loss_fn, use_wandb=False, ema=None):

    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        x_0 = batch["I"]

        t = torch.randint(
            0, timesteps, (x_0.shape[0],), dtype=torch.int64).to(device)
        noisy_image, noise = diffusion.forward_process(x_0, t)

        noisy_image = noisy_image.to(device)
        noise = noise.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(noisy_image, t)

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
                            output[0], 0, -1).cpu().detach().numpy())]},
                          step=epoch, commit=False)

        # Backward pass
        loss = loss_fn(output, noise)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        if ema:
            ema.step()

    epoch_loss /= len(data_loader)

    return epoch_loss
