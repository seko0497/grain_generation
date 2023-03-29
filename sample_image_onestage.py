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
run_path = "vm-ml/grain_generation/ek2w67iq"
num_samples = 32
sampling_steps = 1000
round_masks = True

dataset = "grain"
in_channels = 1
out_channels = 2
num_classes = 2
img_channels = 1

# call  wandb API
wandb_api = wandb.Api()
run = wandb_api.run(run_path)
run_name = run.name

# restore model checkpoint
# model_folder = f"grain_detection/models/{run_name}"
# print(f"restoring {model_folder}")
# checkpoint = wandb.restore(
#     "grain_detection/best.pth",
#     run_path=run_path,
#     root=model_folder)
# print("restored")
checkpoint = torch.load("wear_generation/best_1200.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = Unet(
    run.config["model_dim"],
    device,
    in_channels=in_channels,
    out_channels=out_channels,
    dim_mults=run.config["dim_mults"],
    num_resnet_blocks=run.config["num_resnet_blocks"],
    num_classes=num_classes)
image_model = torch.nn.parallel.DataParallel(image_model)
image_model.load_state_dict(checkpoint["model_state_dict"])
image_model.to(device)

image_diffusion = Diffusion(
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
generated = 0

for _ in range((num_samples // run.config["batch_size"]) + 1):

    if num_samples - generated < run.config["batch_size"]:
        batch_size = num_samples - generated
    else:
        batch_size = run.config["batch_size"]

    samples = image_validation.generate_samples(
        "None",
        pred_type=run.config["pred_type"],
        img_channels=img_channels,
        num_classes=num_classes,
        diffusion=image_diffusion,
        sampling_steps=sampling_steps,
        batch_size=batch_size,
        model=image_model,
        device=device)
    if "masks" in samples:
        sample_masks = samples["masks"]
        if round_masks:
            sample_masks = torch.round(sample_masks)

    for i in range(batch_size):

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
        image.save(f"{save_folder}/sample_{i + generated}.png")
    generated += batch_size
