import torch
import wandb


class Diffusion:

    def __init__(self, beta_0, beta_t, timesteps, img_size, device, schedule,
                 use_wandb=False):

        self.use_wandb = use_wandb

        self.timesteps = timesteps
        self.image_size = img_size

        if schedule == "linear":
            self.betas = self.beta_schedule(beta_0, beta_t, timesteps)
        elif schedule == "cosine":
            self.betas = self.cosine_beta_schedule(timesteps)
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

    def cosine_beta_schedule(self, timesteps, s=0.0008):

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

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

    def sample(self, model, n, epoch):

        model.eval()
        with torch.no_grad():

            samples = torch.Tensor()

            x = torch.randn((n, 3, self.image_size[0], self.image_size[1]))
            samples = torch.cat((samples, x[0]), dim=2)

            for t in reversed(range(self.timesteps)):

                betas_t = self.extract(
                    self.betas,
                    torch.full((n,), t), x.shape)

                sqrt_one_minus_alphas_cumprod_t = self.extract(
                    self.sqrt_one_minus_alphas_cumprod,
                    torch.full((n,), t), x.shape)

                sqrt_recip_alphas_t = self.extract(
                    torch.sqrt(1.0 / self.alphas),
                    torch.full((n,), t), x.shape)

                predicted_noise = model(
                    x.to(self.device), torch.full((n,), t).to(self.device))

                model_mean = (sqrt_recip_alphas_t * (
                    x - ((betas_t / sqrt_one_minus_alphas_cumprod_t)
                         * predicted_noise.cpu())
                    ))

                if t == 0:

                    x = model_mean

                else:

                    model_mean = model_mean.clamp(-1, 1)
                    posterior_variance_t = self.extract(
                        self.posterior_variance,
                        torch.full((n,), t), x.shape
                    )

                    noise = torch.randn_like(x)
                    x = (model_mean + torch.sqrt(posterior_variance_t) * noise)

                if t % (self.timesteps / 10) == 0:
                    samples = torch.cat((samples, x[0]), dim=2)

            samples = (samples.clamp(-1, 1) + 1) / 2
            samples = (samples * 255).type(torch.uint8)

            if self.use_wandb:

                wandb.log({"Sample_evolution": wandb.Image(
                    torch.moveaxis(samples, 0, -1).cpu().detach().numpy())},
                    step=epoch, commit=False)

            model.train()
            x = (x.clamp(-1, 1) + 1) / 2

            return x

    def mean(self, x_t, x_0, t):

        sqrt_alphas_cumprod_prev_t = self.extract(
            torch.sqrt(self.alphas_cumprod_prev), t, x_0.shape)

        one_minus_alphas_cumprod_t = self.extract(
            1. - self.alphas_cumprod, t, x_0.shape)

        sqrt_alphas = self.extract(
            torch.sqrt(self.alphas, t, x_0.shape))

        one_minus_alphas_cumprod_prev_t = self.extract(
            1. - self.alphas_cumprod_prev, t, x_0.shape)

        beta_t = self.extract(
            self.betas, t, x_0.shape)

        sum_1 = ((sqrt_alphas_cumprod_prev_t * beta_t) /
                 one_minus_alphas_cumprod_t) * x_0

        sum_2 = ((sqrt_alphas * one_minus_alphas_cumprod_prev_t) /
                 one_minus_alphas_cumprod_t) * x_t

        return sum_1 + sum_2

    def q_posterior(self, x_t, x_0, t):

        posterior_variance_t = self.extract(
            self.posterior_variance, t, x_0.shape)

        return self.mean(x_t, x_0, t), posterior_variance_t

    def vlb_loss(self, x_t, x_0, t):

        pass

# DEBUG
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
