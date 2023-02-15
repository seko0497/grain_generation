import os
from matplotlib import pyplot as plt
import torch
from diffusion import Diffusion, get_schedule
from unet import Unet
from PIL import Image

checkpoint = torch.load("wear_generation/best.pth")
wandb_name = "aflame-candles-301"

img_size = (64, 64)

model_dim = 128
dim_mults = (1, 2, 4, 8)
num_resnet_blocks = 2

beta_0 = 0.000025
beta_t = 0.005
timesteps = 4000
schedule = "cosine"
sampling_steps = 100

loss = "hybrid"

grid = False

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
    device,
    use_wandb=False)

samples = diffusion.sample(model, 9, checkpoint["epoch"], sampling_steps)

save_folder = f"wear_generation/samples/{wandb_name}"

if grid:

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    for ax in axs.flatten():
        ax.axis('off')

    image_idx = 0
    for i in range(3):
        for j in range(3):
            print(image_idx)
            sample = (samples[image_idx] * 255).type(torch.uint8)
            sample = torch.moveaxis(sample, 0, -1).cpu().detach().numpy()
            axs[i, j].imshow(sample)
            image_idx += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{save_folder}/image_grid.png')

else:

    for i, sample in samples:
        sample = (sample * 255).type(torch.uint8)
        sample = torch.moveaxis(sample, 0, -1).cpu().detach().numpy()
        image = Image.fromarray(sample)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        image.save(f"{save_folder}/sample_{i}.png")
