import torch
import wandb
from config import get_config
from torch.utils.data import DataLoader
from unet import Unet
from train import train

from wear_generation.dataset_wear import WearDataset
from diffusion import Diffusion

from torchmetrics.image.fid import FrechetInceptionDistance


def main():

    config = get_config()

    use_wandb = config.get("use_wandb", False)

    if use_wandb:
        wandb.init(config=config, entity="seko97", project="wear_generation")

    wandb.config.update(
        {"beta_0": config["beta_0"] / (wandb.config.timesteps / 1000),
         "beta_t": config["beta_t"] / (wandb.config.timesteps / 1000)},
        allow_val_change=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.get("random_seed", 1234))
    torch.backends.cudnn.determerministic = True

    persistent_workers = True if config["num_workers"] > 0 else False

    if use_wandb:
        train_loader = DataLoader(WearDataset(
            f"{config['train_dataset']}/train",
            raw_img_size=config['raw_img_size'],
            img_size=config['img_size']
        ), batch_size=wandb.config.batch_size,
           num_workers=config.get("num_workers", 1),
           persistent_workers=persistent_workers,
           pin_memory=True,
           shuffle=True)

    else:

        train_loader = DataLoader(WearDataset(
            f"{config['train_dataset']}/train",
            raw_img_size=config['raw_img_size'],
            img_size=config['img_size']
        ), batch_size=config.get("batch_size", 4),
           num_workers=config.get("num_workers", 1),
           persistent_workers=persistent_workers,
           pin_memory=True,
           shuffle=True)

    fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)

    img_size = wandb.config.img_size if use_wandb else config["img_size"]
    fid_loader = DataLoader(WearDataset(
        f"{config['train_dataset']}/train",
        raw_img_size=config['raw_img_size'],
        img_size=wandb.config.img_size if use_wandb else config["img_size"]
    ), batch_size=115,
        num_workers=config.get("num_workers", 1),
        persistent_workers=persistent_workers,
        pin_memory=True,
        shuffle=False)
    for batch in fid_loader:
        real_samples = (batch["I"] + 1) / 2

    fid.update(real_samples, real=True)

    if use_wandb:
        model = Unet(config["model_dim"], device)
    else:
        model = Unet(wandb.config.model_dim, device)

    if use_wandb:

        diffusion = Diffusion(
            wandb.config.beta_0,
            wandb.config.beta_t,
            wandb.config.timesteps,
            wandb.config.img_size,
            device,
            wandb.config.schedule,
            use_wandb=use_wandb)
    else:
        diffusion = Diffusion(
            config["beta_0"],
            config["beta_t"],
            config["timesteps"],
            config["img_size"],
            device,
            config['schedule'],
            use_wandb=use_wandb)

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

    best = {"epoch": 0, "fid": torch.inf}

    timesteps = wandb.config.timesteps if use_wandb else config["timesteps"]

    for epoch in range(1, config["epochs"] + 1):
        epoch_loss = train(model, diffusion, timesteps, device,
                           train_loader, optimizer, epoch, loss, use_wandb)
        if epoch % config["evaluate_every"] == 0:
            samples = diffusion.sample(model, 4, epoch)
            if use_wandb:
                wandb.log({"Sample": wandb.Image(
                    torch.moveaxis(samples[0], 0, -1).cpu().detach().numpy())},
                    step=epoch, commit=False)
            fid.reset()
            fid.update(samples, real=False)
            current_fid = fid.compute()
            if current_fid <= best["fid"]:
                best["epoch"] = epoch
                best["fid"] = current_fid

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, "wear_generation/best.pth")
                wandb.save("wear_generation/best.pth")

            if use_wandb:
                wandb.log({"FID": current_fid,
                           "best_epoch": best["epoch"],
                           "best_fid": best["fid"]},
                          step=epoch, commit=False)

        if use_wandb:
            wandb.log({"train_loss": epoch_loss})


if __name__ == '__main__':
    main()
