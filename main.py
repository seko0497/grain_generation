from matplotlib import cm
import numpy as np
import torch
import wandb
from config import config as config_dict
from torch.utils.data import DataLoader
from unet import Unet
from train import train
from losses import HybridLoss

from dataset_wear import WearDataset
from diffusion import Diffusion, get_schedule
from ema import ExponentialMovingAverage

from validate import Validation


def main():

    if config_dict["use_wandb"]:
        wandb.init(
            config=config_dict, entity="seko97", project="wear_generation")

    config = Config(config=config_dict, wandb=config_dict["use_wandb"])

    if config.get("use_wandb"):
        wandb.config.update(
            {"beta_0": config.get("beta_0") / (wandb.config.timesteps / 1000),
             "beta_t": config.get("beta_t") / (wandb.config.timesteps / 1000)},
            allow_val_change=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.get("random_seed"))
    torch.backends.cudnn.determerministic = True

    persistent_workers = True if config.get("num_workers") > 0 else False

    train_loader = DataLoader(WearDataset(
        f"{config.get('train_dataset')}/train",
        raw_img_size=config.get('raw_img_size'),
        img_size=config.get('img_size'),
        mask=True if config.get("pred_mask") is not None else False
    ), batch_size=config.get("batch_size"),
        num_workers=config.get("num_workers"),
        persistent_workers=persistent_workers,
        pin_memory=True,
        shuffle=True)

    valid_loader = DataLoader(WearDataset(
        f"{config.get('train_dataset')}/valid",
        raw_img_size=config.get('raw_img_size'),
        img_size=config.get('img_size'),
        mask=True if config.get("pred_mask") is not None else False
    ), batch_size=config.get("batch_size"),
        num_workers=config.get("num_workers"),
        persistent_workers=persistent_workers,
        pin_memory=True,
        shuffle=True)

    validation = Validation(
        config.get('train_dataset'),
        config.get("raw_img_size"),
        config.get("img_size"),
        num_workers=config.get("num_workers"))

    in_channels = 3 if config.get("pred_mask") is None else 4
    out_channels = (in_channels * 2 if config.get("loss") == "hybrid"
                    else in_channels)

    checkpoint = None
    if config.get("checkpoint"):
        checkpoint = torch.load(config.get("checkpoint"))
        print(f"loaded checkpoint {config.get('checkpoint')}")

    model = Unet(config.get("model_dim"),
                 device,
                 in_channels=in_channels,
                 out_channels=out_channels,
                 dim_mults=config.get("dim_mults"),
                 num_resnet_blocks=config.get("num_resnet_blocks"),
                 dropout=config.get("dropout"))

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if config.get("ema"):
        ema = ExponentialMovingAverage(model)
    else:
        ema = None

    betas = get_schedule(
        config.get("schedule"),
        config.get("beta_0"),
        config.get("beta_t"),
        config.get("timesteps"))
    diffusion = Diffusion(
        betas,
        config.get("timesteps"),
        config.get("img_size"),
        in_channels,
        device,
        predict_mask=config.get("pred_mask"),
        use_wandb=config.get("use_wandb"))

    optimizer = getattr(torch.optim, config.get("optimizer"))(
        model.parameters(),
        lr=config.get("learning_rate"),
        betas=[0.0, 0.999]
    )

    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if config.get("loss") == "simple":
        loss = torch.nn.MSELoss()
    elif config.get("loss") == "hybrid":
        loss = HybridLoss()

    if config.get("use_wandb"):
        wandb.watch(model, log="all")

    best = {"epoch": 0, "fid": torch.inf}
    if checkpoint:
        best["epoch"] = checkpoint["epoch"]

    for epoch in range(1, config.get("epochs") + 1):

        if checkpoint:
            epoch += checkpoint["epoch"]

        epoch_loss = train(model,
                           diffusion,
                           config.get("timesteps"),
                           device,
                           train_loader,
                           optimizer,
                           epoch,
                           loss,
                           config.get("use_wandb"),
                           ema)

        if (epoch % config.get("evaluate_every") == 0 and
                epoch >= config.get("start_eval_epoch")):

            if ema is not None:
                eval_model = ema.ema_model
            else:
                eval_model = model

            eval_model.eval()

            samples = []

            for _ in range(128 // config.get("batch_size")):
                samples.append(diffusion.sample(
                    eval_model, config.get("batch_size"), epoch,
                    sampling_steps=config.get("sampling_steps")))

            samples = torch.cat(samples)
            current_fid = validation.valid_fid(samples[:, :3])
            eval_model.train()

            if current_fid <= best["fid"]:
                best["epoch"] = epoch
                best["fid"] = current_fid

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': eval_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'FID': current_fid,
                    'loss': loss}, f"wear_generation/best.pth")

                if config.get("use_wandb"):
                    wandb.save("wear_generation/best.pth")

                    sample = samples[0].cpu().detach().numpy()
                    sample_image = np.moveaxis(sample[:3], 0, -1)

                    sample_mask = sample[-1]
                    cmap = cm.get_cmap("viridis")
                    sample_mask = cmap(sample_mask)[:, :, :3]

                    sample = np.vstack((sample_image, sample_mask))
                    wandb.log({"Sample": wandb.Image(sample)},
                              step=epoch, commit=False)

            wandb.log({"FID": current_fid,
                       "best_epoch": best["epoch"],
                       "best_fid": best["fid"]},
                      step=epoch, commit=False)

        valid_loss = validation.valid_hybrid_loss(
                model, valid_loader, device, diffusion, config.get("timesteps")
            )

        if config.get("use_wandb"):
            wandb.log({"train_loss": epoch_loss,
                       "valid_loss": valid_loss}, step=epoch)


class Config():

    def __init__(self, config, wandb=True):

        self.use_wand = wandb
        self.config = config

    def get(self, key):

        if self.use_wand:

            return getattr(wandb.config, key)

        else:
            return self.config[key]


if __name__ == '__main__':
    main()
