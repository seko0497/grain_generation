import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

import config
from wear_generation.dataset_wear import WearDataset


class Diffusion:

    def __init__(self, beta_0, beta_t, timesteps, img_size, device):

        self.image_size = img_size

        self.timesteps = timesteps

        self.betas = self.beta_schedule(beta_0, beta_t, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.ones(1), self.alphas_cumprod))[:-1]

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)

        self.posterior_variance = (self.betas *
                                   (1. - self.alphas_cumprod_prev) /
                                   (1. - self.alphas_cumprod))

        self.device = device

    def beta_schedule(self, start, end, timesteps):

        return torch.linspace(start, end, timesteps)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward_process(self, x_0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.extract(
            self.alphas_cumprod, t, x_0.shape
        )

        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return (sqrt_alphas_cumprod_t * x_0 +
                sqrt_one_minus_alphas_cumprod_t * noise), noise

    def sample(self, model, n):

        model.eval()
        with torch.no_grad():

            x = torch.randn((n, 3, self.image_size[0], self.image_size[1]))

            for t in reversed(range(self.timesteps)):

                if t > 0:
                    noise = torch.randn(self.image_size)
                else:
                    noise = torch.zeros(self.image_size)

                alpha = self.extract(
                    self.alphas, torch.full((n,), t),
                    self.image_size)
                alpha_comprod = self.extract(
                    self.alphas_cumprod, torch.full((n,), t),
                    self.image_size)
                posterior = self.extract(
                    self.alphas_cumprod_prev, torch.full((n,), t),
                    self.image_size)

                predicted_noise = model(
                    x.to(self.device), torch.full((n,), t).to(self.device))

                x = (1. / torch.sqrt(alpha) *
                     (
                     - ((1 - alpha) / (torch.sqrt(1 - alpha_comprod)))
                     * predicted_noise.cpu())
                     + posterior * noise)

            model.train()
            # x = (x.clamp(-1, 1) + 1) / 2
            x = (x + 1) / 2
            # x = (x * 255).type(torch.uint8)

            return x


# config = config.get_config()

# diffusion = Diffusion(
#     config["beta_0"],
#     config["beta_t"],
#     config["timesteps"],
#     (500, 500),
#     torch.device("cuda")
# )

# # diffusion.sample(, 4)

# wear_dataset = WearDataset(
#     "data/RT100U_processed/train")
# wear_dataloader = DataLoader(wear_dataset, batch_size=4)

# for batch in wear_dataloader:

#     x_0 = batch["I"]
#     fig, axs = plt.subplots(1, config["timesteps"] // 50)

#     for i, t in enumerate(range(0, config["timesteps"], 50)):

#         noisy_image, noise = diffusion.forward_process(
#             x_0, torch.Tensor([t, t, t, t]).to(torch.int64))
#         noisy_image = (noisy_image - torch.min(noisy_image)) / (
#             torch.max(noisy_image) - torch.min(noisy_image))

#         axs[i].imshow(torch.moveaxis(noisy_image[0], 0, -1))
#     plt.show()
#     quit()
