import math
import os
import numpy as np
import torch
import wandb
from PIL import Image

from unet import Unet, SuperResUnet
from diffusion import Diffusion, get_schedule
from validate import Validation
from image_transforms import get_rgb

# parameters
run_path = {
    "local": None,
    "wandb": "seko97/wear_generation/bga6zhxm",
    "filename": "best.pth"}
sampling_steps = 100

run_path_superres = {
    "local": None,
    "wandb": "seko97/grain_generation/bjv3caz5",
    "filename": "best.pth"}
sampling_steps_superres = 100


num_samples = 5000
superres = False
split = True
colormap = False
round_masks = True

dataset = "wear"
in_channels = 4
out_channels = 8
num_classes = 3
img_channels = 3

# call  wandb API
wandb_api = wandb.Api()

run = wandb_api.run(run_path["wandb"])
run_name = run.name

run_superres = wandb_api.run(run_path_superres["wandb"])
run_superres_name = run_superres.name

# get checkpoint
if run_path["local"] is None:
    model_folder = f"wear_generation/models/{run_name}"
    print(f"restoring {model_folder}")
    checkpoint = wandb.restore(
        f"wear_generation/{run_path['filename']}",
        run_path=run_path["wandb"],
        root=model_folder)
    print("restored")
    checkpoint = torch.load(checkpoint.name)
else:
    checkpoint = torch.load(
        f"wear_generation/{run_path['filename']}")

# get checkpoint superres model
if run_path_superres["local"] is None:
    model_folder_superres = f"{dataset}_generation/models/{run_superres_name}"
    print(f"restoring {model_folder_superres}")
    checkpoint_superres = wandb.restore(
        f"wear_generation/{run_path_superres['filename']}",
        run_path=run_path_superres["wandb"],
        root=model_folder_superres)
    print("restored")
    checkpoint_superres = torch.load(checkpoint_superres.name)
else:
    checkpoint_superres = torch.load(
        f"wear_generation/{run_path_superres['filename']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(
    run.config["model_dim"],
    device,
    in_channels=in_channels,
    out_channels=out_channels,
    dim_mults=run.config["dim_mults"],
    num_resnet_blocks=run.config["num_resnet_blocks"],
    num_classes=num_classes)
model = torch.nn.parallel.DataParallel(model)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

diffusion = Diffusion(
    get_schedule(
        run.config["schedule"],
        run.config["beta_0"],
        run.config["beta_t"],
        run.config["timesteps"]),
    run.config["timesteps"],
    run.config["img_size"],
    in_channels,
    device,
    use_wandb=False,
    num_classes=num_classes)

if superres:

    superres_model = SuperResUnet(
        run_superres.config["model_dim"],
        device,
        in_channels=img_channels + 1,
        out_channels=(img_channels + 1) * 2,
        dim_mults=run_superres.config["dim_mults"],
        num_resnet_blocks=run_superres.config["num_resnet_blocks"],
        num_classes=num_classes)
    superres_model = torch.nn.parallel.DataParallel(superres_model)
    superres_model.load_state_dict(checkpoint_superres["model_state_dict"],
                                   strict=False)
    superres_model.to(device)

    superres_diffusion = Diffusion(
        get_schedule(
            run_superres.config["schedule"],
            run_superres.config["beta_0"],
            run_superres.config["beta_t"],
            run_superres.config["timesteps"]),
        run_superres.config["timesteps"],
        run_superres.config["img_size"],
        device=device,
        in_channels=img_channels + 1,
        use_wandb=False,
        num_classes=num_classes)

save_folder = (
    f"{dataset}_generation/samples/"
    f"{run.name}/epoch{checkpoint['epoch']}_steps{sampling_steps}")
image_validation = Validation(img_channels=img_channels)

generated = 0
for _ in range(math.ceil(num_samples / run.config["batch_size"])):

    if num_samples - generated < run.config["batch_size"]:
        n = num_samples - generated
    else:
        n = run.config["batch_size"]

    samples = image_validation.generate_samples(
        "None",
        num_samples=n,
        pred_type=run.config["pred_type"],
        img_channels=img_channels,
        num_classes=num_classes,
        diffusion=diffusion,
        sampling_steps=sampling_steps,
        batch_size=run.config["batch_size"],
        model=model,
        device=device,
        round_pred_x_0=run.config["round_pred_x_0"])
    if "masks" in samples:
        sample_masks = samples["masks"]
        if round_masks:
            if num_classes > 2:
                sample_masks *= num_classes
                sample_masks[sample_masks == num_classes] = num_classes - 1
                sample_masks = sample_masks.int()
                sample_masks = sample_masks / (num_classes - 1)
            else:
                sample_masks = torch.round(sample_masks)
    if "images" in samples:
        sample_images = samples["images"]

    if superres:

        sample_images = sample_images * 2 - 1
        sample_masks = sample_masks * 2 - 1
        low_res_image = torch.nn.functional.interpolate(
            sample_images, (256, 256), mode="bilinear")
        low_res_mask = torch.nn.functional.interpolate(
            sample_masks, (256, 256), mode="nearest")
        low_res = torch.cat((low_res_image, low_res_mask), dim=1)

        superres_samples = superres_diffusion.sample(
            superres_model,
            n,
            low_res=low_res.to(device),
            sampling_steps=sampling_steps_superres
        )
        sample_images = superres_samples[:, :img_channels]
        sample_masks = superres_samples[:, img_channels:]

    for i in range(n):

        if "images" in samples:
            if dataset == "grain":
                if colormap:
                    sample_intensity = get_rgb(sample_images[i, 0])
                    sample_depth = get_rgb(sample_images[i, 1])
                else:
                    sample_intensity = sample_images[
                        i, 0].cpu().detach().numpy()
                    sample_depth = sample_images[
                        i, 1].cpu().detach().numpy()
            elif dataset == "wear":
                sample_image = (sample_images[i]
                                .cpu().detach().numpy())
                sample_image = np.moveaxis(sample_image, 0, -1)
        if "masks" in samples:
            if dataset == "wear":
                sample_mask = sample_masks[i, 0] / (num_classes - 1)
            elif dataset == "grain":
                sample_mask = torch.round(sample_masks[i, 0])
            if colormap:
                sample_mask = get_rgb(sample_mask)
            else:
                sample_mask = sample_masks[i, 0].cpu().detach().numpy()

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if "images" in samples:
            if dataset == "grain":
                sample_intensity = (
                    sample_intensity * 255).astype(np.uint8)
                sample_depth = (
                    sample_depth * 255).astype(np.uint8)

                if split:
                    image_intensity = Image.fromarray(sample_intensity)
                    image_intensity.save(
                        f"{save_folder}/{i + generated}_intensity.png")
                    image_depth = Image.fromarray(sample_depth)
                    image_depth.save(
                        f"{save_folder}/{i + generated}_depth.png")
                else:
                    sample_image = np.vstack((sample_intensity, sample_depth))

            elif dataset == "wear":
                sample_image = (sample_image * 255).astype(np.uint8)
                if split:
                    image = Image.fromarray(sample_image)
                    image.save(
                        f"{save_folder}/{i + generated}_image.png")
        if "masks" in samples:
            sample_mask = (sample_mask * 255).astype(np.uint8)
            if split:
                image_mask = Image.fromarray(sample_mask)
                image_mask.save(
                        f"{save_folder}/{i + generated}_target.png")

        if "images" not in samples:
            sample = sample_mask
        elif "masks" not in samples:
            sample = sample_image
        if "masks" in samples and "images" in samples and not split:
            sample = np.vstack((sample_image, sample_mask))

        if not split:
            image = Image.fromarray(sample)

            image.save(f"{save_folder}/sample_{i + generated}.png")

    generated += n
