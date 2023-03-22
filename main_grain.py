from matplotlib import cm, pyplot as plt
import numpy as np
import torch
import wandb
from config import config as config_dict
from torch.utils.data import DataLoader
from unet import Unet, SuperResUnet
from train import train
from losses import HybridLoss

from dataset_grain import GrainDataset
from diffusion import Diffusion, get_schedule
from ema import ExponentialMovingAverage

from validate import Validation


def main():

    if config_dict["use_wandb"]:
        wandb.init(
            config=config_dict, entity="vm-ml", project="grain_generation")

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

    train_loader = DataLoader(GrainDataset(
        f"{config.get('train_dataset')}",
        img_size=config.get('img_size'),
        image_idxs=[7, 2, 5, 8, 6, 10, 1, 4],
        mask_one_hot=config.get("mask_one_hot"),
    ), batch_size=config.get("batch_size"),
        num_workers=config.get("num_workers"),
        persistent_workers=persistent_workers,
        pin_memory=True,
        shuffle=True)

    valid_loader = DataLoader(GrainDataset(
        f"{config.get('train_dataset')}",
        img_size=config.get('img_size'),
        image_idxs=[9],
        mask_one_hot=config.get("mask_one_hot"),
        train=False,
    ), batch_size=1,
        num_workers=config.get("num_workers"),
        persistent_workers=persistent_workers,
        pin_memory=True,
        shuffle=True)

    intensity_validation = Validation(config.get("img_channels"))
    depth_validation = Validation(config.get("img_channels"))

    if config.get("pred_type") != "mask":
        intensity_validation.fit_real_samples(train_loader, channel=0)
        depth_validation.fit_real_samples(train_loader, channel=1)

    mask_validation = None
    if config.get("pred_type") == "all" or config.get("pred_type") == "mask":
        mask_validation = Validation(config.get("img_channels"))
        mask_validation.fit_real_samples(train_loader, channel=2)

    if config.get("pred_type") == "all":
        if config.get("mask_one_hot"):
            in_channels = (config.get("img_channels") +
                           config.get("num_classes"))
        else:
            in_channels = config.get("img_channels") + 1
    elif config.get("pred_type") == "mask":
        if config.get("mask_one_hot"):
            in_channels = config.get("num_classes")
        else:
            in_channels = 1
    else:
        in_channels = config.get("img_channels")
    out_channels = (in_channels * 2 if config.get("loss") == "hybrid"
                    else in_channels)

    checkpoint = None
    if config.get("checkpoint"):
        checkpoint = torch.load(config.get("checkpoint"))

    if not config.get("super_res"):
        model = Unet(config.get("model_dim"),
                     device,
                     in_channels=in_channels,
                     out_channels=out_channels,
                     dim_mults=config.get("dim_mults"),
                     num_resnet_blocks=config.get("num_resnet_blocks"),
                     dropout=config.get("dropout"),
                     spade=config.get("condition") == "mask",
                     num_classes=config.get("num_classes"))
    else:
        model = SuperResUnet(config.get("model_dim"),
                             device,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             dim_mults=config.get("dim_mults"),
                             num_resnet_blocks=config.get("num_resnet_blocks"),
                             dropout=config.get("dropout"),
                             spade=config.get("condition") == "mask",
                             num_classes=config.get("num_classes"))

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
        use_wandb=config.get("use_wandb"),
        num_classes=config.get("num_classes"))

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

    best = {"epoch": 0, "fid_image": torch.inf, "fid_mask": torch.inf}
    start_epoch = 1
    if checkpoint:
        best["epoch"] = checkpoint["epoch"]
        start_epoch = checkpoint["epoch"] + 1
    del checkpoint

    for epoch in range(start_epoch, config.get("epochs") + 1):

        epoch_loss = train(model,
                           diffusion,
                           config.get("timesteps"),
                           device,
                           train_loader,
                           optimizer,
                           epoch,
                           loss,
                           ema,
                           img_channels=config.get("img_channels"),
                           pred_type=config.get("pred_type"),
                           condition=config.get("condition"),
                           super_res=config.get("super_res"),
                           drop_rate=config.get("drop_condition_rate"))

        if (epoch % config.get("evaluate_every") == 0 and
                epoch >= config.get("start_eval_epoch")):

            if ema is not None:
                eval_model = ema.ema_model
            else:
                eval_model = model

            eval_model.eval()

            validation = Validation(config.get("img_channels"))
            samples, sample_masks, label_dists, low_res = validation.generate_samples(
                config.get("condition"),
                config.get("pred_type"),
                config.get("img_channels"),
                config.get("num_classes"),
                diffusion,
                config.get("sampling_steps"),
                config.get("batch_size"),
                eval_model,
                device,
                config.get("super_res"),
                valid_loader,
                guidance_scale=config.get("guidance_scale"))
            sample_masks = torch.round(sample_masks)

            if config.get("condition") == "label_dist":
                label_dist_rmse = validation.label_dist_rmse(
                    sample_masks, label_dists,
                    train_loader.dataset.label_dist_scaler)
            else:
                label_dist_rmse = None

            current_mask_fid = None
            current_intensity_fid = None
            current_depth_fid = None
            current_image_fid = None
            if config.get("pred_type") != "mask":
                current_intensity_fid = intensity_validation.valid_fid(
                    samples[:, 0].cpu().detach())
                current_depth_fid = depth_validation.valid_fid(
                    samples[:, 1].cpu().detach())
                # current_image_fid = (
                #     current_intensity_fid + current_depth_fid) / 2
                # current_image_fid = current_intensity_fid
                best_metric = "fid_image"
            else:
                best_metric = "fid_mask"
            if mask_validation is not None:
                current_mask_fid = mask_validation.valid_fid(
                    sample_masks.cpu().detach())
                current_image_fid = (
                    current_intensity_fid + current_depth_fid +
                    current_mask_fid) / 2

            eval_model.train()

            torch.save({
                    'epoch': epoch,
                    'model_state_dict': eval_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'image_fid': current_image_fid,
                    'mask_fid': current_mask_fid,
                    'loss': loss}, f"wear_generation/checkpoint.pth")
            if config.get("use_wandb"):
                wandb.save("wear_generation/checkpoint.pth")

            current_fid = (current_image_fid if best_metric == "fid_image" else
                           current_mask_fid)

            if current_fid <= best[best_metric]:
                best["epoch"] = epoch
                best[best_metric] = current_fid

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': eval_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'image_fid': current_image_fid,
                    'mask_fid': current_mask_fid,
                    'loss': loss}, f"wear_generation/best.pth")

                if config.get("use_wandb"):
                    wandb.save("wear_generation/best.pth")

            if config.get("use_wandb"):

                if config.get("condition") != "mask":
                    num_samples_log = 4
                else:
                    num_samples_log = samples.shape[0]

                for i in range(num_samples_log):

                    if samples != []:

                        sample_intensity = samples[i, 0]
                        sample_intensity = (sample_intensity.cpu().
                                            detach().numpy())
                        cmap = cm.get_cmap("viridis")
                        sample_intensity = cmap(sample_intensity)[:, :, :3]

                        sample_depth = samples[i, 1]
                        sample_depth = (sample_depth.cpu().
                                        detach().numpy())
                        cmap = cm.get_cmap("viridis")
                        sample_depth = cmap(sample_depth)[:, :, :3]

                    if sample_masks != []:

                        sample_mask = sample_masks[i]
                        sample_mask = sample_mask.cpu().detach().numpy()
                        cmap = cm.get_cmap("viridis")
                        sample_mask = cmap(sample_mask)[:, :, :3]

                    if samples != [] and sample_masks != []:
                        sample = np.vstack(
                            (sample_intensity, sample_depth, sample_mask))
                    elif samples != []:
                        # sample = np.vstack(
                        #     (sample_intensity, sample_depth))
                        # sample = np.vstack(
                        #     (sample_intensity))
                        sample = sample_intensity
                    else:
                        sample = sample_mask

                    low_res = torch.nn.functional.interpolate(
                        low_res, (sample.shape[1], sample.shape[1]),
                        mode="nearest")
                    low_res_cmap = []
                    for low_res_channel in low_res[i]:
                        cmap = cm.get_cmap("viridis")
                        low_res_channel = cmap(low_res_channel)[:, :, :3]
                        low_res_cmap.append(low_res_channel)
                    low_res_cmap = np.vstack(low_res_cmap)

                    sample = np.hstack((low_res_cmap, sample))

                    wandb.log({f"Sample_{i}": wandb.Image(sample)},
                              step=epoch, commit=False)

            if config.get("use_wandb"):
                fid_log = {"fid_image": current_image_fid,
                           "fid_mask": current_mask_fid,
                           "fid_intensity": current_intensity_fid,
                           "fid_depth": current_depth_fid,
                           "best_epoch": best["epoch"],
                           "best_fid": best[best_metric]}
                # if (current_image_fid is not None
                #         and current_mask_fid is not None):
                #     fid_log["fid_mean"] = (
                #         (current_image_fid + current_mask_fid) / 2)
                wandb.log(fid_log, step=epoch, commit=False)

        if config.get("use_wandb"):
            wandb.log({"train_loss": epoch_loss}, step=epoch)


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
