# parameters
import os
import numpy as np
import torch
import wandb
from PIL import Image

from unet import Unet
from diffusion import Diffusion, get_schedule
from validate import Validation
from image_transforms import get_rgb


run_path_mask = {
    "local": None,
    "wandb": "vm-ml/grain_generation/ek2w67iq",
    "filename": "best.pth"}
sampling_steps_mask = 200

run_path_image = {
    "local": None,
    "wandb": "vm-ml/grain_generation/ek2w67iq",
    "filename": "best.pth"}
sampling_steps_image = 1000

num_samples = 32

dataset = "grain"
in_channels = 1
out_channels = 2
num_classes = 2
img_channels = 1

# call  wandb API
wandb_api = wandb.Api()

run_mask = wandb_api.run(run_path_mask["wandb"])
run_mask_name = run_mask.name

run_image = wandb_api.run(run_path_image["wandb"])
run_image_name = run_image.name

# get checkpoint mask model
if run_path_mask["local"] is None:
    model_folder_mask = f"grain_detection/models/{run_mask_name}"
    print(f"restoring {model_folder_mask}")
    checkpoint_mask = wandb.restore(
        f"grain_detection/{run_path_mask['filename']}",
        run_path=run_path_mask["wandb"],
        root=model_folder_mask)
    print("restored")
else:
    checkpoint_mask = torch.load(
        f"wear_generation/{run_path_mask['filename']}")

# get checkpoint image model
if run_path_image["local"] is None:
    model_folder_image = f"grain_detection/models/{run_image_name}"
    print(f"restoring {model_folder_image}")
    checkpoint_image = wandb.restore(
        f"grain_detection/{run_path_image['filename']}",
        run_path=run_path_image["wandb"],
        root=model_folder_image)
    print("restored")
else:
    checkpoint_image = torch.load(
        f"wear_generation/{run_path_image['filename']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask_model = Unet(
    run_mask.config["model_dim"],
    device,
    in_channels=in_channels,
    out_channels=out_channels,
    dim_mults=run_mask.config["dim_mults"],
    num_resnet_blocks=run_mask.config["num_resnet_blocks"],
    num_classes=num_classes)
mask_model = torch.nn.parallel.DataParallel(mask_model)
mask_model.load_state_dict(checkpoint_mask["model_state_dict"])
mask_model.to(device)

mask_diffusion = Diffusion(
    get_schedule(
        run_mask.config["schedule"],
        run_mask.config["beta_0"],
        run_mask.config["beta_t"],
        run_mask.config["timesteps"]),
    run_mask.config["timesteps"],
    run_mask.config["img_size"],
    in_channels,
    device,
    use_wandb=False,
    num_classes=num_classes)

image_model = Unet(
    run_image.config["model_dim"],
    device,
    in_channels=in_channels,
    out_channels=out_channels,
    dim_mults=run_image.config["dim_mults"],
    num_resnet_blocks=run_image.config["num_resnet_blocks"],
    num_classes=num_classes,
    spade=True)
image_model = torch.nn.parallel.DataParallel(image_model)
image_model.load_state_dict(checkpoint_image["model_state_dict"])
image_model.to(device)

image_diffusion = Diffusion(
    get_schedule(
        run_image.config["schedule"],
        run_image.config["beta_0"],
        run_image.config["beta_t"],
        run_image.config["timesteps"]),
    run_image.config["timesteps"],
    run_image.config["img_size"],
    in_channels,
    device,
    use_wandb=False,
    num_classes=num_classes)

save_folder = (
    f"grain_generation/samples/"
    f"{run_image.name}_{run_mask.name}/"
    f"epoch{checkpoint_image['epoch']}_epoch{checkpoint_mask['epoch']}"
    f"steps{sampling_steps_image}_{sampling_steps_mask}")
mask_validation = Validation(img_channels=img_channels)
image_validation = Validation(img_channels=img_channels)

generated = 0

for _ in range(((num_samples // run_mask.config["batch_size"]) + 1)):

    if num_samples - generated < run_mask.config["batch_size"]:
        batch_size_mask = num_samples - generated
    else:
        batch_size_mask = run_mask.config["batch_size"]

    sample_masks = mask_validation.generate_samples(
        "None",
        pred_type="mask",
        img_channels=img_channels,
        num_classes=num_classes,
        diffusion=mask_diffusion,
        sampling_steps=sampling_steps_mask,
        batch_size=batch_size_mask,
        model=mask_model,
        device=device)["masks"]

    sample_masks = torch.round(sample_masks)
    sample_masks_one_hot = torch.nn.functional.one_hot(
        sample_masks.long(),
        num_classes=num_classes)

    if run_image.config["batch_size"] < batch_size_mask:
        num_batches = batch_size_mask // run_image.config["batch_size"]
    else:
        num_batches = 1

    for i in range(num_batches):
        sample_masks_batch = sample_masks_one_hot[
            i * run_image.config["batch_size"]:
            (i + 1) * run_image.config["batch_size"]]
        sample_images = image_diffusion.sample(
                image_model,
                run_image.config["batch_size"],
                mask=sample_masks_batch.to(
                    device),
                sampling_steps=sampling_steps_image,
                pred_type="image",
                img_channels=img_channels,
                guidance_scale=run_image.guidance_scale)

        for j in range(run_image.config["batch_size"]):

            if dataset == "grain":
                sample_intensity = get_rgb(sample_images[j, 0])
                sample_depth = get_rgb(sample_images[j, 1])
                sample_image = np.vstack(
                    (sample_intensity, sample_depth))
            elif dataset == "wear":
                sample_image = sample_images.cpu().detach().numpy()
                sample_image = np.moveaxis(sample_image, 0, -1)

            sample_mask = get_rgb(sample_masks[i, 0])

            sample = np.vstack((sample_image, sample_mask))

            sample = (sample * 255).astype(np.uint8)
            image = Image.fromarray(sample)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            image.save(f"{save_folder}/sample_{i + generated}.png")
        generated += run_image.config["batch_size"]
