import os
from matplotlib import cm, colors, pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb
from diffusion import Diffusion, get_schedule
from unet import Unet
from PIL import Image
from torch.utils.data import DataLoader
from dataset_wear import WearDataset

from validate import Validation

# checkpoint = torch.load("grain_generation/models/custard-strudel-12/950.pth")
wandb_name = "sleek-leaf-10"

run_path = "vm-ml/grain_generation/z93f0mk2"
checkpoint = wandb.restore("wear_generation/best.pth", run_path=run_path)
checkpoint = torch.load(checkpoint.name)

img_size = (64, 64)

model_dim = 256
dim_mults = (1, 1, 2, 2, 4, 4)
num_resnet_blocks = 2

beta_0 = 0.000025
beta_t = 0.005
timesteps = 4000
schedule = "cosine"
sampling_steps = 1000

batch_size = 64

loss = "hybrid"
pred_mask = "naive"

grid = False
grid_size = [4, 4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 2
out_channels = (in_channels * 2 if loss == "hybrid"
                else in_channels)

model = Unet(model_dim,
             device,
             in_channels=in_channels,
             out_channels=out_channels,
             dim_mults=dim_mults,
             num_resnet_blocks=num_resnet_blocks,
             num_classes=2)

model = torch.nn.parallel.DataParallel(model)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

diffusion = Diffusion(
    get_schedule(schedule, beta_0, beta_t, timesteps),
    timesteps,
    img_size,
    in_channels,
    device,
    use_wandb=False,
    num_classes=2)

# num_samples = grid_size[0] * grid_size[1]
# all_samples = []
# for _ in range(num_samples):
#     samples = diffusion.sample(
#         model, 1, checkpoint["epoch"], sampling_steps)
#     all_samples.append(samples[0])
# samples = torch.stack(all_samples)

# if pred_mask is not None:
#     samples = samples.cpu().detach()
#     sample_images = samples[:, :3]
#     sample_masks = samples[:, -1]
#     sample_masks_rgb = []
#     cmap = cm.get_cmap("viridis")
#     for sample_mask in sample_masks:
#         sample_masks_rgb.append(cmap(sample_mask)[:, :, :3])

#     sample_masks = np.stack(sample_masks_rgb)
#     sample_masks = torch.moveaxis(torch.Tensor(sample_masks), -1, 1)
#     samples = torch.cat((sample_images, torch.Tensor(sample_masks)), dim=2)

save_folder = f"grain_generation/samples/{wandb_name}/5000"

validation = Validation(img_channels=2)
# samples, sample_masks, _ = validation.generate_samples(
#     "None",
#     pred_type="all",
#     img_channels=2,
#     num_classes=2,
#     diffusion=diffusion,
#     sampling_steps=sampling_steps,
#     batch_size=batch_size,
#     model=model,
#     device=device)

samples, sample_masks, _ = validation.generate_samples(
    "None",
    pred_type="all",
    img_channels=1,
    num_classes=2,
    diffusion=diffusion,
    sampling_steps=sampling_steps,
    batch_size=batch_size,
    model=model,
    device=device)

for i in range(batch_size):

    sample_intensity = samples[i, 0]
    sample_intensity = torch.nn.functional.interpolate(
        sample_intensity[None, None],
        (128, 128),
        mode="nearest-exact")[0, 0]
    sample_intensity = (sample_intensity.cpu().
                        detach().numpy())
    # normalizer = colors.Normalize()
    # sample_intensity = normalizer(sample_intensity)
    cmap = cm.get_cmap("viridis")
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(sample_intensity)
    sample_intensity = cmap(sample_intensity)[:, :, :3]
    # axes[1].imshow(sample_intensity)

    # sample_depth = samples[0, 1]
    # sample_depth = (sample_depth.cpu().
    #                 detach().numpy())
    # cmap = cm.get_cmap("viridis")
    # sample_depth = cmap(sample_depth)[:, :, :3]

    sample_mask = sample_masks[i]
    sample_mask = torch.round(sample_mask)
    sample_mask = torch.nn.functional.interpolate(
        sample_mask[None, None],
        (128, 128),
        mode="nearest-exact"
    )[0, 0]
    sample_mask = sample_mask.cpu().detach().numpy()
    cmap = cm.get_cmap("viridis")
    sample_mask = cmap(sample_mask)[:, :, :3]

    sample = np.vstack((sample_intensity, sample_mask))

    sample = (sample * 255).astype(np.uint8)
    image = Image.fromarray(sample)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    image.save(f"{save_folder}/sample_{i}.png")
quit()

if grid:

    fig, axs = plt.subplots(
        nrows=grid_size[0], ncols=grid_size[1], figsize=(10, 10))
    for ax in axs.flatten():
        ax.axis('off')

    image_idx = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            sample = (samples[image_idx] * 255).type(torch.uint8)
            sample = torch.moveaxis(sample, 0, -1).cpu().detach().numpy()
            axs[i, j].imshow(sample)
            image_idx += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.savefig(f'{save_folder}/image_grid.png')
    plt.show()

else:

    for i, sample in enumerate(samples):
        i += 2
        sample = (sample * 255).type(torch.uint8)
        image = Image.fromarray(sample)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        image.save(f"{save_folder}/sample_{i}.png")

# data_loader = DataLoader(WearDataset(
#     f"data/RT100U_processed/train",
#     raw_img_size=(448, 576),
#     img_size=(64, 64)
# ), batch_size=1,
#     shuffle=False)

# sims = []

# for batch in tqdm(data_loader):

#     train_image = batch["I"]
#     train_image = (train_image + 1) / 2

#     sims.append(torch.nn.CosineSimilarity()(
#         torch.flatten(train_image, start_dim=1),
#         torch.flatten(samples.cpu(), start_dim=1)))

# print(torch.argmax(torch.stack(sims)))
# print(torch.max(torch.stack(sims)))

# match = list(data_loader)[torch.argmax(torch.stack(sims))]["I"]

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
# for ax in axs.flatten():
#     ax.axis('off')

# match = (match + 1) / 2
# match = (match[0] * 255).type(torch.uint8)
# match = torch.moveaxis(match, 0, -1).cpu().detach().numpy()

# sample = (samples[0] * 255).type(torch.uint8)
# sample = torch.moveaxis(sample, 0, -1).cpu().detach().numpy()

# axs[0].imshow(match)
# axs[1].imshow(sample)
# plt.show()
