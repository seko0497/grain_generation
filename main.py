import torch
import wandb
import copy
from config import Config
from torch.utils.data import DataLoader
from unet import Unet
from train import train

from wear_generation.dataset_wear import WearDataset
from diffusion import Diffusion
from ema import ExponentialMovingAverage

from validate import Validation


def main():

    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.get("random_seed", 1234))
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
        config.get("image_size"),
        num_workers=config.get("num_workers"))

    model = Unet(config.get("model_dim"), device)

    if config.get("ema"):
        ema = ExponentialMovingAverage(model)
    else:
        ema = None

    diffusion = Diffusion(
        config.get("beta_0"),
        config.get("beta_t"),
        config.get("timesteps"),
        config.get("img_size"),
        device,
        config.get('schedule'),
        use_wandb=config.get("use_wandb"))

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)

    model.to(device)

    optimizer = getattr(torch.optim, config.get("optimizer"))(
        model.parameters(),
        lr=config["learning_rate"],
        betas=[0.0, 0.999]
    )

    loss = getattr(torch.nn, config.get("loss", "CrossEntropyLoss"))()

    if config.get("use_wandb"):
        wandb.watch(model, log="all")

    best = {"epoch": 0, "fid": torch.inf}

    for epoch in range(1, config["epochs"] + 1):
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

        if epoch % config["evaluate_every"] == 0:

            if ema is not None:
                eval_model = ema.ema_model
            else:
                eval_model = model

            samples = diffusion.sample(eval_model, 4, epoch)

            current_fid = validation.validate(samples, epoch)

            if current_fid <= best["fid"]:
                best["epoch"] = epoch
                best["fid"] = current_fid

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': eval_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, "wear_generation/best.pth")
                wandb.save("wear_generation/best.pth")

            if config.get("use_wandb"):
                wandb.log({"FID": current_fid,
                           "best_epoch": best["epoch"],
                           "best_fid": best["fid"],
                           "Sample": wandb.Image(
                                torch.moveaxis(
                                 samples[0], 0, -1).cpu().detach().numpy())},
                          step=epoch, commit=False)

        if config.get("use_wandb"):
            wandb.log({"train_loss": epoch_loss})


if __name__ == '__main__':
    main()
