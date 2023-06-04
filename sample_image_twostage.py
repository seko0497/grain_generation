# parameters
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


run_path_mask = {
    "local": None,
    "wandb": "seko97/grain_generation/jfkd4lza",
    "filename": "best.pth"}
sampling_steps_mask = 100

run_path_image = {
    "local": None,
    "wandb": "seko97/grain_generation/7varmv7t",
    "filename": "best.pth"}
sampling_steps_image = 100

run_path_superres = {
    "local": None,
    "wandb": "seko97/grain_generation/bjv3caz5",
    "filename": "best.pth"}
sampling_steps_superres = 100

num_samples = 5014
superres = True
split = True
colormap = False

dataset = "grain"
num_classes = 2
img_channels = 2

# call  wandb API
wandb_api = wandb.Api()

run_mask = wandb_api.run(run_path_mask["wandb"])
run_mask_name = run_mask.name

run_image = wandb_api.run(run_path_image["wandb"])
run_image_name = run_image.name

run_superres = wandb_api.run(run_path_superres["wandb"])
run_superres_name = run_superres.name

# get checkpoint mask model
if run_path_mask["local"] is None:
    model_folder_mask = f"{dataset}_generation/models/{run_mask_name}"
    print(f"restoring {model_folder_mask}")
    checkpoint_mask = wandb.restore(
        f"wear_generation/{run_path_mask['filename']}",
        run_path=run_path_mask["wandb"],
        root=model_folder_mask)
    print("restored")
    checkpoint_mask = torch.load(checkpoint_mask.name)
else:
    checkpoint_mask = torch.load(
        f"wear_generation/{run_path_mask['filename']}")

# get checkpoint image model
if run_path_image["local"] is None:
    model_folder_image = f"{dataset}_generation/models/{run_image_name}"
    print(f"restoring {model_folder_image}")
    checkpoint_image = wandb.restore(
        f"wear_generation/{run_path_image['filename']}",
        run_path=run_path_image["wandb"],
        root=model_folder_image)
    print("restored")
    checkpoint_image = torch.load(checkpoint_image.name)
else:
    checkpoint_image = torch.load(
        f"wear_generation/{run_path_image['filename']}")

if superres:
    # get checkpoint superres model
    if run_path_superres["local"] is None:
        model_folder_superres = (
            f"{dataset}_generation/models/{run_superres_name}")
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

mask_model = Unet(
    run_mask.config["model_dim"],
    device,
    in_channels=1,
    out_channels=2,
    dim_mults=run_mask.config["dim_mults"],
    num_resnet_blocks=run_mask.config["num_resnet_blocks"],
    num_classes=num_classes)
mask_model = torch.nn.parallel.DataParallel(mask_model)
mask_model.load_state_dict(checkpoint_mask["model_state_dict"], strict=False)
mask_model.to(device)

mask_diffusion = Diffusion(
    get_schedule(
        run_mask.config["schedule"],
        run_mask.config["beta_0"],
        run_mask.config["beta_t"],
        run_mask.config["timesteps"]),
    run_mask.config["timesteps"],
    run_mask.config["img_size"],
    device=device,
    in_channels=1,
    use_wandb=False,
    num_classes=num_classes)

image_model = Unet(
    run_image.config["model_dim"],
    device,
    in_channels=img_channels,
    out_channels=img_channels * 2,
    dim_mults=run_image.config["dim_mults"],
    num_resnet_blocks=run_image.config["num_resnet_blocks"],
    num_classes=num_classes,
    spade=True)
image_model = torch.nn.parallel.DataParallel(image_model)
image_model.load_state_dict(checkpoint_image["model_state_dict"], strict=False)
image_model.to(device)

image_diffusion = Diffusion(
    get_schedule(
        run_image.config["schedule"],
        run_image.config["beta_0"],
        run_image.config["beta_t"],
        run_image.config["timesteps"]),
    run_image.config["timesteps"],
    run_image.config["img_size"],
    device=device,
    in_channels=img_channels,
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
    superres_model.load_state_dict(
        checkpoint_superres["model_state_dict"], strict=False)
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
    f"{run_image.name}_{run_mask.name}/"
    f"epoch{checkpoint_image['epoch']}_epoch{checkpoint_mask['epoch']}"
    f"steps{sampling_steps_image}_{sampling_steps_mask}")
mask_validation = Validation(img_channels=img_channels)

generated = 0

for _ in range(math.ceil(num_samples / run_mask.config["batch_size"])):

    if num_samples - generated < run_mask.config["batch_size"]:
        n = num_samples - generated
    else:
        n = run_mask.config["batch_size"]

    sample_masks = mask_validation.generate_samples(
        "None",
        num_samples=n,
        pred_type="mask",
        img_channels=img_channels,
        num_classes=num_classes,
        diffusion=mask_diffusion,
        sampling_steps=sampling_steps_mask,
        batch_size=run_mask.config["batch_size"],
        model=mask_model,
        device=device,
        round_pred_x_0=(
            run_mask.config["round_pred_x_0"]
            if dataset == "wear" else False))["masks"]

    if dataset == "grain":
        sample_masks = torch.round(sample_masks)
    elif dataset == "wear":
        sample_masks *= num_classes
        sample_masks[sample_masks == num_classes] = num_classes - 1
        sample_masks = sample_masks.int()
    sample_masks_one_hot = torch.nn.functional.one_hot(
        sample_masks.squeeze().long(),
        num_classes=num_classes).float()
    sample_masks_one_hot = sample_masks_one_hot * 2 - 1
    sample_masks_one_hot = torch.moveaxis(sample_masks_one_hot, -1, 1)

    generated_images = 0
    for i in range(math.ceil(n / run_image.config["batch_size"])):

        if n - generated_images < run_image.config["batch_size"]:
            num_samples_images = n - generated_images
        else:
            num_samples_images = run_image.config["batch_size"]

    # if run_image.config["batch_size"] < n:
    #     num_batches = n // run_image.config["batch_size"]
    # else:
    #     num_batches = 1

    # for i in range(num_batches):
        sample_masks_batch = sample_masks_one_hot[
            generated_images:generated_images + num_samples_images]
        sample_images = image_diffusion.sample(
                image_model,
                num_samples_images,
                mask=sample_masks_batch.to(
                    device),
                sampling_steps=sampling_steps_image,
                guidance_scale=run_image.config["guidance_scale"])

        if superres:

            sample_images = sample_images * 2 - 1
            sample_masks = sample_masks * 2 - 1
            low_res_image = torch.nn.functional.interpolate(
                sample_images, (256, 256), mode="bilinear")
            low_res_mask = torch.nn.functional.interpolate(
                sample_masks[i * n:(i + 1) * n], (256, 256), mode="nearest")
            low_res = torch.cat((low_res_image, low_res_mask), dim=1)

            superres_samples = superres_diffusion.sample(
                superres_model,
                n,
                low_res=low_res.to(device),
                sampling_steps=sampling_steps_superres
            )
            sample_images = superres_samples[:, :img_channels]
            sample_masks = superres_samples[:, img_channels:]
            sample_masks = torch.round(sample_masks)

        for j in range(num_samples_images):

            if dataset == "grain":
                if colormap:
                    sample_intensity = get_rgb(sample_images[j, 0])
                    sample_depth = get_rgb(sample_images[j, 1])
                else:
                    sample_intensity = sample_images[
                        j, 0].cpu().detach().numpy()
                    sample_depth = sample_images[
                        j, 1].cpu().detach().numpy()
                sample_image = np.vstack(
                    (sample_intensity, sample_depth))
            elif dataset == "wear":
                sample_image = sample_images[j].cpu().detach().numpy()
                sample_image = np.moveaxis(sample_image, 0, -1)

            if dataset == "wear":
                sample_mask = sample_masks[
                    j + generated_images, 0] / (num_classes - 1)
            else:
                sample_mask = sample_masks[j + generated_images, 0]
            if colormap:
                sample_mask = get_rgb(sample_mask)
            else:
                sample_mask = sample_mask.cpu().detach().numpy()

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            image_index = j + generated + generated_images

            if split:
                if dataset == "grain":
                    sample_intensity = (
                        sample_intensity * 255).astype(np.uint8)
                    image_intensity = Image.fromarray(sample_intensity)
                    image_intensity.save(
                        f"{save_folder}/{image_index}_intensity.png")

                    sample_depth = (
                        sample_depth * 255).astype(np.uint8)
                    image_depth = Image.fromarray(sample_depth)
                    image_depth.save(
                        f"{save_folder}/{image_index}_depth.png")

                if dataset == "wear":
                    sample_image = (sample_image * 255).astype(np.uint8)
                    image = Image.fromarray(sample_image)
                    image.save(
                        f"{save_folder}/{image_index}_image.png"
                    )

                sample_mask = (
                    sample_mask * 255).astype(np.uint8)
                image_mask = Image.fromarray(sample_mask)
                image_mask.save(
                    f"{save_folder}/{image_index}_target.png")
            else:
                sample = np.vstack((sample_image, sample_mask))
                sample = (sample * 255).astype(np.uint8)
                image = Image.fromarray(sample)

                image.save(f"{save_folder}/sample_{image_index}.png")
        generated_images += num_samples_images
    generated += n
