import os
import torch
from diffusion import Diffusion
from unet import Unet
from PIL import Image

checkpoint = torch.load("wear_generation/best.pth")
wandb_name = "stilted-surf-294"

img_size = (64, 64)

model_dim = 128
dim_mults = (1, 2, 4, 8)
num_resnet_blocks = 2

beta_0 = 0.00005
beta_t = 0.01
timesteps = 2000
schedule = "cosine"

loss = "simple"

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
    beta_0,
    beta_t,
    timesteps,
    img_size,
    device,
    schedule,
    sampling_steps=None,
    use_wandb=False)

samples = diffusion.sample(model, 8, checkpoint["epoch"])

save_folder = f"wear_generation/samples/{wandb_name}"

for i, sample in enumerate(samples):
    sample = (sample * 255).type(torch.uint8)
    sample = torch.moveaxis(sample, 0, -1).cpu().detach().numpy()
    image = Image.fromarray(sample)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    image.save(f"{save_folder}/sample_{i}.png")
