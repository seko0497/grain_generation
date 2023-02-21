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
        img_size=config.get('img_size')
    ), batch_size=config.get("batch_size"),
        num_workers=config.get("num_workers"),
        persistent_workers=persistent_workers,
        pin_memory=True,
        shuffle=True)

    validation = Validation(
        f"{config.get('train_dataset')}/train",
        config.get("raw_img_size"),
        config.get("img_size"),
        num_workers=config.get("num_workers"))

    out_channels = (config.get("out_size") if config.get("loss") == "simple"
                    else config.get("out_size") * 2)

    model = Unet(config.get("model_dim"),
                 device,
                 out_channels=out_channels,
                 in_channels=config.get("out_size"),
                 dim_mults=config.get("dim_mults"),
                 num_resnet_blocks=config.get("num_resnet_blocks"))

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)

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
        config.get("out_size"),
        device,
        use_wandb=config.get("use_wandb"))

    optimizer = getattr(torch.optim, config.get("optimizer"))(
        model.parameters(),
        lr=config.get("learning_rate"),
        betas=[0.0, 0.999]
    )

    if config.get("loss") == "simple":
        loss = torch.nn.MSELoss()
    elif config.get("loss") == "hybrid":
        loss = HybridLoss()

    if config.get("use_wandb"):
        wandb.watch(model, log="all")

    best = {"epoch": 0, "fid": torch.inf}

    for epoch in range(1, config.get("epochs") + 1):
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

            samples = diffusion.sample(
                eval_model, 4, epoch,
                sampling_steps=config.get("sampling_steps"))

            current_fid = validation.validate(samples[:, :3])

            if current_fid <= best["fid"]:
                best["epoch"] = epoch
                best["fid"] = current_fid

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': eval_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, "wear_generation/best.pth")

            if config.get("use_wandb"):
                # wandb.save("wear_generation/best.pth")

                sample = samples[0].cpu().detach().numpy()
                sample_image = np.moveaxis(sample[:3], 0, -1)

                sample_mask = sample[-1]
                cmap = cm.get_cmap("viridis")
                sample_mask = cmap(sample_mask)[:, :, :3]

                sample = np.vstack((sample_image, sample_mask))

                wandb.log({"FID": current_fid,
                           "best_epoch": best["epoch"],
                           "best_fid": best["fid"],
                           "Sample": wandb.Image(sample)},
                          step=epoch, commit=False)

        if config.get("use_wandb"):
            wandb.log({"train_loss": epoch_loss})


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
