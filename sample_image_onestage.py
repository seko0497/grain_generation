import os
import numpy as np
import torch
import wandb
from PIL import Image

from unet import Unet
from diffusion import Diffusion, get_schedule
from validate import Validation
from image_transforms import get_rgb

# parameters
run_path = {
    "local": None,
    "wandb": "seko97/grain_generation/jfkd4lza",
    "filename": "best.pth"}
sampling_steps = 200

num_samples = 25
round_masks = True

dataset = "grain"
in_channels = 1
out_channels = 2
num_classes = 2
img_channels = 1

# call  wandb API
wandb_api = wandb.Api()

run = wandb_api.run(run_path["wandb"])
run_name = run.name

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

save_folder = (
    f"grain_generation/samples/"
    f"{run.name}/epoch{checkpoint['epoch']}_steps{sampling_steps}")
image_validation = Validation(img_channels=img_channels)

samples = image_validation.generate_samples(
    "None",
    num_samples=num_samples,
    pred_type=run.config["pred_type"],
    img_channels=img_channels,
    num_classes=num_classes,
    diffusion=diffusion,
    sampling_steps=sampling_steps,
    batch_size=run.config["batch_size"],
    model=model,
    device=device)
if "masks" in samples:
    sample_masks = samples["masks"]
    if round_masks:
        sample_masks = torch.round(sample_masks)

for i in range(num_samples):

    if "images" in samples:
        if dataset == "grain":
            sample_intensity = get_rgb(samples["images"][i, 0])
            sample_depth = get_rgb(samples["images"][i, 1])
            sample_image = np.vstack(
                (sample_intensity, sample_depth))
        elif dataset == "wear":
            sample_image = (samples["images"]
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

    sample = (sample * 255).astype(np.uint8)
    image = Image.fromarray(sample)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    image.save(f"{save_folder}/sample_{i}.png")
