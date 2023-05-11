from matplotlib import cm, pyplot as plt
import numpy as np
import torch
import wandb
from train_config import config as config_dict
from torch.utils.data import DataLoader
from unet import Unet, SuperResUnet
from train import train
from losses import HybridLoss

from dataset_grain import GrainDataset
from dataset_wear import WearDataset
from diffusion import Diffusion, get_schedule
from ema import ExponentialMovingAverage

from validate import Validation
from image_transforms import get_rgb


def main():

    # initialize weights and biases logging
    if config_dict["use_wandb"]:
        wandb.init(
            config=config_dict, entity="seko97", project="grain_generation")
    config = Config(config=config_dict, wandb=config_dict["use_wandb"])

    if config.get("use_wandb"):
        wandb.config.update(
            {"beta_0": config.get("beta_0") / (wandb.config.timesteps / 1000),
             "beta_t": config.get("beta_t") / (wandb.config.timesteps / 1000)},
            allow_val_change=True)

    # set some torch defaults
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.get("random_seed"))
    torch.backends.cudnn.determerministic = True
    persistent_workers = True if config.get("num_workers") > 0 else False

    # initialize train dataset
    if config.get("dataset") == "grain":
        dataset_train = GrainDataset(
            config.get("grain_defaults")["root_dir"],
            config.get("grain_defaults")["channel_names"],
            config.get("grain_defaults")["image_idxs"][0],
            config.get("grain_defaults")["patch_size"],
            config.get("img_size"),
            mask_one_hot=config.get("mask_one_hot"),)
    elif config.get("dataset") == "wear":
        dataset_train = WearDataset(
            f"{config.get('wear_defaults')['root_dir']}/train",
            config.get("wear_defaults")["raw_img_size"],
            config.get("img_size"),
            mask_one_hot=config.get("mask_one_hot"),
            label_dist=config.get("condition") == "label_dist")

    # initialize validation dataset
    if config.get("dataset") == "grain":
        dataset_validation = GrainDataset(
            config.get("grain_defaults")["root_dir"],
            config.get("grain_defaults")["channel_names"],
            config.get("grain_defaults")["image_idxs"][1],
            config.get("grain_defaults")["patch_size"],
            config.get("img_size"),
            mask_one_hot=config.get("mask_one_hot"),
            train=False)
    elif config.get("dataset") == "wear":
        dataset_validation = WearDataset(
            f"{config.get('wear_defaults')['root_dir']}/test",
            config.get("wear_defaults")["raw_img_size"],
            config.get("img_size"),
            mask_one_hot=config.get("mask_one_hot"),
            label_dist=config.get("condition") == "label_dist")

    # initialize dataloader
    dataloader_kwargs = {
        "batch_size": config.get("batch_size"),
        "num_workers": config.get("num_workers"),
        "persistent_workers": persistent_workers,
        "pin_memory": True,
        "shuffle": True}
    train_loader = DataLoader(dataset_train, **dataloader_kwargs)
    valid_loader = DataLoader(dataset_validation, **dataloader_kwargs)

    # get number of image channels and classes
    if config.get("dataset") == "grain":
        img_channels = config.get("grain_defaults")["img_channels"]
        num_classes = config.get("grain_defaults")["num_classes"]
    elif config.get("dataset") == "wear":
        img_channels = config.get("wear_defaults")["img_channels"]
        num_classes = config.get("wear_defaults")["num_classes"]

    # initialize Validation for sampling and FID calculation
    if config.get("dataset") == "grain":
        intensity_validation = Validation(img_channels)
        depth_validation = Validation(img_channels)
    elif config.get("dataset") == "wear":
        image_validation = Validation(img_channels)

    # fit real images for FID calculation
    if config.get("pred_type") != "mask":
        if config.get("dataset") == "grain":
            intensity_validation.fit_real_samples(
                train_loader, channel=0)
            depth_validation.fit_real_samples(
                train_loader, channel=1)
        elif config.get("dataset") == "wear":
            image_validation.fit_real_samples(
                train_loader, channel=-1)

    # initialize and fit real masks for FID calculation
    if config.get("pred_type") == "all" or config.get("pred_type") == "mask":
        mask_validation = Validation(img_channels)
        mask_channel = 2 if config.get("dataset") == "grain" else 3
        mask_validation.fit_real_samples(
            train_loader, channel=mask_channel,
            one_hot=config.get("mask_one_hot"))
    else:
        mask_validation = None

    # calculate in- and out-channels depending on prediction type
    if config.get("pred_type") == "all":
        if config.get("mask_one_hot"):
            in_channels = img_channels + num_classes
        else:
            in_channels = img_channels + 1
    elif config.get("pred_type") == "mask":
        if config.get("mask_one_hot"):
            in_channels = num_classes
        else:
            in_channels = 1
    else:
        in_channels = img_channels
    out_channels = (in_channels * 2 if config.get("loss") == "hybrid"
                    else in_channels)

    # load Checkpoint
    if config.get("use_checkpoint"):
        if config.get("checkpoint")["local"]:
            checkpoint = torch.load(config.get("checkpoint"))
        else:
            wandb_api = wandb.Api()
            run = wandb_api.run(config.get("checkpoint")["wandb"])
            run_name = run.name
            model_folder = (
                f"{config.get('dataset')}_generation/models/"
                f"{run_name}")
            print(f"restoring {model_folder}")
            checkpoint = wandb.restore(
                f"wear_generation/{config.get('checkpoint')['filename']}",
                run_path=config.get("checkpoint")["wandb"],
                root=model_folder)
            print("restored")
            checkpoint = torch.load(checkpoint.name)
    else:
        checkpoint = None

    # initialize model
    model_args = (config.get("model_dim"), device)
    model_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "dim_mults": config.get("dim_mults"),
        "num_resnet_blocks": config.get("num_resnet_blocks"),
        "dropout": config.get("dropout"),
        "spade": config.get("condition") == "mask",
        "num_classes": num_classes}
    if not config.get("super_res"):
        model = Unet(*model_args, **model_kwargs)
    else:
        model = SuperResUnet(*model_args, **model_kwargs)

    # initialize multi GPU training
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # initialize EMA
    if config.get("ema"):
        ema = ExponentialMovingAverage(model)
    else:
        ema = None

    # initialize Diffusion
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
        num_classes=num_classes)

    # initialize optimizer
    optimizer = getattr(torch.optim, config.get("optimizer"))(
        model.parameters(),
        lr=config.get("learning_rate"))
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if config.get("loss") == "simple":
        loss = torch.nn.MSELoss()
    elif config.get("loss") == "hybrid":
        loss = HybridLoss(pred_noise=config.get("pred_noise"))

    # log model parameters
    if config.get("use_wandb"):
        wandb.watch(model, log="all")

    # prepare training
    best = {"epoch": 0, "fid": torch.inf}
    start_epoch = 1
    if checkpoint:
        best["epoch"] = checkpoint["epoch"]
        start_epoch = checkpoint["epoch"] + 1
    del checkpoint
    torch.cuda.empty_cache()

    # ################## train loop ################################
    for epoch in range(start_epoch, config.get("epochs") + 1):

        # perform forward backward pass
        epoch_loss = train(model,
                           diffusion,
                           config.get("timesteps"),
                           device,
                           train_loader,
                           optimizer,
                           epoch,
                           loss,
                           ema,
                           img_channels=img_channels,
                           pred_type=config.get("pred_type"),
                           condition=config.get("condition"),
                           super_res=config.get("super_res"),
                           drop_rate=config.get("drop_condition_rate"),
                           round_pred_x_0=config.get("round_pred_x_0"))

        if (epoch % config.get("evaluate_every") == 0 and
                epoch >= config.get("start_eval_epoch")):

            # set model to eval mode
            eval_model = ema.ema_model if ema is not None else model
            eval_model.eval()

            # generate samples
            validation = Validation(img_channels)
            samples = validation.generate_samples(
                config.get("condition"),
                config.get("pred_type"),
                img_channels,
                num_classes,
                diffusion,
                config.get("sampling_steps"),
                config.get("batch_size"),
                eval_model,
                device,
                super_res=config.get("super_res"),
                valid_loader=valid_loader,
                num_samples=config.get("num_samples"),
                guidance_scale=config.get("guidance_scale"),
                pred_noise=config.get("pred_noise"),
                clamp=config.get("clamp"),
                round_pred_x_0=config.get("round_pred_x_0"))

            # process masks
            if "masks" in samples:
                sample_masks = samples["masks"]
                if config.get("round_masks"):
                    sample_masks = torch.round(sample_masks)

                # sample_masks *= num_classes
                # mask_pred[mask_pred == num_classes] = num_classes - 1.0
                # mask_pred = mask_pred.int().float()
                # mask_pred /= num_classes - 1.0

            # calculate label dists loss
            if "label_dists" in samples:
                label_dist_rmse = validation.label_error(
                    sample_masks,  samples["label_dists"])
            else:
                label_dist_rmse = None

            # calculate FIDs
            current_fid = {}
            if config.get("pred_type") != "mask":
                if config.get("dataset") == "grain":
                    current_fid["intensity"] = intensity_validation.valid_fid(
                        samples["images"][:, 0])
                    current_fid["depth"] = depth_validation.valid_fid(
                        samples["images"][:, 1])
                elif config.get("dataset") == "wear":
                    current_fid["image"] = image_validation.valid_fid(
                        samples["images"]
                    )
            if mask_validation is not None:
                current_fid["mask"] = mask_validation.valid_fid(
                    sample_masks[:, 0], one_hot=config.get("mask_one_hot"))

            eval_model.train()

            # save checkpoint
            if config.get("save_models"):
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': eval_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'fid': current_fid,
                        'loss': loss}, f"wear_generation/checkpoint.pth")
                if config.get("use_wandb"):
                    wandb.save("wear_generation/checkpoint.pth")

            mean_current_fid = (sum(list(current_fid.values()))
                                / len(current_fid))

            # save if best model
            if mean_current_fid <= best["fid"]:
                best["epoch"] = epoch
                best["fid"] = mean_current_fid
                if config.get("save_models"):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': eval_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'fid': current_fid,
                        'loss': loss}, f"wear_generation/best.pth")
                    if config.get("use_wandb"):
                        wandb.save("wear_generation/best.pth")

            log_samples = True
            if mean_current_fid > best["fid"] and config.get("log_best"):
                log_samples = False

            if log_samples:
                # log samples and scores in weights and biases
                if config.get("use_wandb"):

                    num_samples_log = 4

                    # log samples
                    for i in range(num_samples_log):
                        if "images" in samples:
                            if config.get("dataset") == "grain":
                                sample_intensity = get_rgb(
                                    samples["images"][i, 0])
                                sample_depth = get_rgb(samples["images"][i, 1])
                                sample_image = np.vstack(
                                    (sample_intensity, sample_depth))
                            elif config.get("dataset") == "wear":
                                sample_image = (samples["images"][i]
                                                .cpu().detach().numpy())
                                sample_image = np.moveaxis(sample_image, 0, -1)
                        if "masks" in samples:
                            sample_mask = get_rgb(sample_masks[i, 0])

                        if "images" in samples and "masks" in samples:
                            sample = np.vstack((sample_image, sample_mask))
                        elif "images" not in samples:
                            sample = sample_mask
                        elif "masks" not in samples:
                            sample = sample_image

                        if "low_res" in samples:
                            low_res = samples["low_res"]
                            low_res = (low_res + 1) / 2
                            low_res_cmap = []
                            for low_res_channel in low_res[i]:
                                low_res_channel = get_rgb(low_res_channel)
                                low_res_cmap.append(low_res_channel)
                            low_res_cmap = np.vstack(low_res_cmap)
                            sample = np.hstack((low_res_cmap, sample))

                        wandb.log({f"Sample_{i}": wandb.Image(sample)},
                                  step=epoch, commit=False)

            if config.get("use_wandb"):
                # log scores
                fid_log = {"mean_fid": mean_current_fid,
                           "best_epoch": best["epoch"],
                           "best_fid": best["fid"]}

                for key in current_fid.keys():
                    fid_log[f"fid_{key}"] = current_fid[key]

                wandb.log(fid_log, step=epoch, commit=False)

                if label_dist_rmse is not None:
                    wandb.log(
                        {"label_dist_error": label_dist_rmse},
                        step=epoch,
                        commit=False)

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
