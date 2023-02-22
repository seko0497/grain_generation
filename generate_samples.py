import os
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from diffusion import Diffusion, get_schedule
from unet import Unet
from PIL import Image
from torch.utils.data import DataLoader
from dataset_wear import WearDataset

from validate import Validation

checkpoint = torch.load("wear_generation/best.pth")
wandb_name = "drawn-totem-311"

img_size = (256, 256)

model_dim = 64
dim_mults = (1, 2, 4, 8)
num_resnet_blocks = 2

beta_0 = 0.000025
beta_t = 0.005
timesteps = 4000
schedule = "cosine"
sampling_steps = 1000

loss = "hybrid"

grid = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(model_dim,
             device,
             out_channels=3 if loss == "simple" else 6,
             dim_mults=dim_mults,
             num_resnet_blocks=num_resnet_blocks)

model = torch.nn.parallel.DataParallel(model)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

diffusion = Diffusion(
    get_schedule(schedule, beta_0, beta_t, timesteps),
    timesteps,
    img_size,
    3,
    device,
    use_wandb=False)

all_samples = torch.Tensor([]).to(device)
validation = Validation("data/RT100U_processed/train",
                        (448, 576), (64, 64), 0)

# for _ in range(7):

samples = diffusion.sample(model, 4, checkpoint["epoch"], sampling_steps)
# all_samples = torch.cat((all_samples, samples))


# current_fid = validation.valid_fid(all_samples)
# print(current_fid)
# quit()

save_folder = f"wear_generation/samples/{wandb_name}"

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

if grid:

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for ax in axs.flatten():
        ax.axis('off')

    image_idx = 0
    for i in range(2):
        for j in range(2):
            print(image_idx)
            sample = (samples[image_idx] * 255).type(torch.uint8)
            sample = torch.moveaxis(sample, 0, -1).cpu().detach().numpy()
            axs[i, j].imshow(sample)
            image_idx += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.savefig(f'{save_folder}/image_grid.png')
    plt.show()

else:

    for i, sample in samples:
        sample = (sample * 255).type(torch.uint8)
        sample = torch.moveaxis(sample, 0, -1).cpu().detach().numpy()
        image = Image.fromarray(sample)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        image.save(f"{save_folder}/sample_{i}.png")
