import torch
import wandb
from config import get_config
from torch.utils.data import DataLoader
from unet import UNet
from train import train

from wear_generation.dataset_wear import WearDataset
from diffusion import Diffusion


def main():

    config = get_config()

    use_wandb = config.get("use_wandb", False)

    if use_wandb:
        wandb.init(config=config, entity="seko97", project="wear_generation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.get("random_seed", 1234))
    torch.backends.cudnn.determerministic = True

    persistent_workers = True if config["num_workers"] > 0 else False

    if use_wandb:
        train_loader = DataLoader(WearDataset(
            f"{config['train_dataset']}/train",
        ), batch_size=wandb.config.batch_size,
           num_workers=config.get("num_workers", 1),
           persistent_workers=persistent_workers,
           pin_memory=True,
           shuffle=True)

    else:

        train_loader = DataLoader(WearDataset(
            f"{config['train_dataset']}/train",
        ), batch_size=config.get("batch_size", 4),
           num_workers=config.get("num_workers", 1),
           persistent_workers=persistent_workers,
           pin_memory=True,
           shuffle=True)

    model = UNet(3, 3, config["time_emb_dim"], device)
    diffusion = Diffusion(
        config["beta_0"],
        config["beta_t"],
        config["timesteps"],
        config["img_size"],
        device
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)

    model.to(device)

    if use_wandb:

        optimizer = getattr(torch.optim, config.get("optimizer", "Adam"))(
            model.parameters(),
            lr=wandb.config.learning_rate,
            betas=[0.0, 0.999]
        )

    else:
        optimizer = getattr(torch.optim, config.get("optimizer", "Adam"))(
            model.parameters(),
            lr=config["learning_rate"],
            betas=[0.0, 0.999]
        )

    loss = getattr(torch.nn, config.get("loss", "CrossEntropyLoss"))()

    if use_wandb:
        wandb.watch(model, log="all")

    for epoch in range(1, config["epochs"] + 1):
        epoch_loss = train(model, diffusion, config["timesteps"], device,
                           train_loader, optimizer, epoch, loss, use_wandb)
        if epoch % config["evaluate_every"] == 0:
            samples = diffusion.sample(model, 1)
            wandb.log({"Sample": wandb.Image(
                torch.moveaxis(samples[0], 0, -1).cpu().detach().numpy())},
                step=epoch, commit=False)

        wandb.log({"train_loss": epoch_loss})


if __name__ == '__main__':
    main()
