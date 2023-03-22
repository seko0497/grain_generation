import os
from matplotlib import cm, colors, pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb
from diffusion import Diffusion, get_schedule
from unet import Unet, SuperResUnet
from PIL import Image
from torch.utils.data import DataLoader
from dataset_wear import WearDataset

from validate import Validation
from dataset_grain import GrainDataset

wandb_name_image = "absurd-grass-13"
image_model_folder = f"grain_generation/models/{wandb_name_image}/1850"
if not os.path.exists(image_model_folder):
    os.makedirs(image_model_folder)
run_path_image = "vm-ml/grain_generation/xa5ur122"
checkpoint_image = wandb.restore(
    "wear_generation/best.pth",
    run_path=run_path_image,
    root=image_model_folder)
checkpoint_image = torch.load(checkpoint_image.name)

wandb_name_superres = "restful-feather-28"
superres_model_folder = f"grain_generation/models/{wandb_name_superres}/700"
# if not os.path.exists(image_model_folder):
#     os.makedirs(image_model_folder)
# run_path_superres = "vm-ml/grain_generation/kq419lu7"
# print(f"restoring {superres_model_folder}")
# checkpoint_superres = wandb.restore(
#     "wear_generation/checkpoint.pth",
#     run_path=run_path_superres,
#     root=superres_model_folder)
checkpoint_superres = torch.load("wear_generation/best.pth")

img_size = (64, 64)
superres_size = (256, 256)

model_dim = 256
dim_mults = (1, 1, 2, 2, 4, 4)
num_resnet_blocks = 2

beta_0 = 0.000025
beta_t = 0.005
timesteps = 4000
schedule = "cosine"
sampling_steps = 200

batch_size = 64

loss = "hybrid"
pred_mask = "naive"

grid = False
grid_size = [4, 4]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 3
out_channels = (in_channels * 2 if loss == "hybrid"
                else in_channels)

image_model = Unet(
    model_dim,
    device,
    in_channels=in_channels,
    out_channels=out_channels,
    dim_mults=dim_mults,
    num_resnet_blocks=num_resnet_blocks,
    num_classes=2)

superres_model = SuperResUnet(
    model_dim,
    device,
    in_channels=in_channels,
    out_channels=out_channels,
    dim_mults=dim_mults,
    num_resnet_blocks=num_resnet_blocks,
    num_classes=2)

image_model = torch.nn.parallel.DataParallel(image_model)
image_model.load_state_dict(checkpoint_image["model_state_dict"])
image_model.to(device)

superres_model = torch.nn.parallel.DataParallel(superres_model)
superres_model.load_state_dict(checkpoint_superres["model_state_dict"])
superres_model.to(device)

image_diffusion = Diffusion(
    get_schedule(schedule, beta_0, beta_t, timesteps),
    timesteps,
    img_size,
    in_channels,
    device,
    use_wandb=False,
    num_classes=2)

superres_diffusion = Diffusion(
    get_schedule("linear", 0.0001, 0.02, 1000),
    1000,
    superres_size,
    in_channels,
    device,
    use_wandb=False,
    num_classes=2)

save_folder = f"grain_generation/samples/{wandb_name_image}/1850"

image_validation = Validation(img_channels=2)
samples, sample_masks, _, _ = image_validation.generate_samples(
    "None",
    pred_type="all",
    img_channels=2,
    num_classes=2,
    diffusion=image_diffusion,
    sampling_steps=sampling_steps,
    batch_size=batch_size,
    model=image_model,
    device=device)

samples = torch.cat((samples, sample_masks[:, None]), dim=1)
samples = samples * 2 - 1

num = 32
samples = superres_diffusion.sample(
    superres_model,
    num,
    mask=None,
    label_dist=None,
    low_res=samples[:num].to(device),
    sampling_steps=sampling_steps,
    pred_type="all",
    img_channels=2)

# superres_validation = Validation(img_channels=2)

# valid_loader = DataLoader(GrainDataset(
#         "data/grains_txt",
#         img_size=superres_size,
#         image_idxs=[9],
#         mask_one_hot=False,
#         train=False,
#     ), batch_size=1,
#         num_workers=32,
#         persistent_workers=True,
#         pin_memory=True,
#         shuffle=True)

# samples, sample_masks, _, _ = superres_validation.generate_samples(
#     "None",
#     pred_type="all",
#     img_channels=2,
#     num_classes=2,
#     diffusion=superres_diffusion,
#     sampling_steps=sampling_steps,
#     batch_size=num,
#     model=superres_model,
#     device=device,
#     super_res=True,
#     valid_loader=valid_loader
# )
# samples = torch.cat((samples, sample_masks[:, None]), dim=1)

for i in range(num):

    sample_intensity = samples[i, 0]
    sample_intensity = (sample_intensity.cpu().
                        detach().numpy())
    cmap = cm.get_cmap("viridis")
    # sample_intensity = colors.Normalize()(sample_intensity)
    sample_intensity = cmap(sample_intensity)[:, :, :3]

    sample_depth = samples[i, 1]
    sample_depth = (sample_depth.cpu().
                    detach().numpy())
    cmap = cm.get_cmap("viridis")
    # sample_depth = colors.Normalize()(sample_depth)
    sample_depth = cmap(sample_depth)[:, :, :3]

    # sample_mask = sample_masks[
    # i]
    sample_mask = samples[i, 2]
    sample_mask = torch.round(sample_mask)
    sample_mask = sample_mask.cpu().detach().numpy()
    cmap = cm.get_cmap("viridis")
    sample_mask = cmap(sample_mask)[:, :, :3]

    sample = np.vstack((sample_intensity, sample_depth, sample_mask))

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
