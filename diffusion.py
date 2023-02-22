import numpy as np
import torch
import wandb
from tqdm import tqdm


class Diffusion:

    def __init__(self, betas, timesteps, img_size, out_channels, device,
                 use_wandb=False):

        self.use_wandb = use_wandb

        self.timesteps = timesteps
        self.image_size = img_size
        self.out_channels = out_channels

        self.betas = betas
        self.calculate_alphas(self.betas)

        self.device = device

    def calculate_alphas(self, betas):

        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.ones(1), self.alphas_cumprod))[:-1]

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (betas *
                                   (1. - self.alphas_cumprod_prev) /
                                   (1. - self.alphas_cumprod))

        self.posterior_log_varaiance_clipped = torch.log(
            torch.cat(
                (self.posterior_variance[1].view(1),
                 self.posterior_variance[1:])))

        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t).to(self.device)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward_process(self, x_0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_0).to(self.device)

        sqrt_alphas_cumprod_t = self.extract(
            self.alphas_cumprod, t, x_0.shape
        )

        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return (sqrt_alphas_cumprod_t * x_0 +
                sqrt_one_minus_alphas_cumprod_t * noise), noise

    def sample(self, model, n, epoch, sampling_steps=None):

        model.eval()
        with torch.no_grad():

            samples = torch.Tensor().to(self.device)

            x = torch.randn(
                (n, self.out_channels, self.image_size[0], self.image_size[1]))
            x = x.to(self.device)
            samples = torch.cat((samples, x[0]), dim=2)

            if sampling_steps is None:
                timesteps = range(self.timesteps)
            else:
                timesteps = torch.round(
                    torch.linspace(
                        0, self.timesteps - 1, sampling_steps)).int()

                last_alpha_cumprod = 1.0
                original_betas = self.betas
                new_betas = []
                for i, alpha_cumprod in enumerate(self.alphas_cumprod):

                    if i in timesteps:
                        new_betas.append(
                            1 - alpha_cumprod / last_alpha_cumprod)
                        last_alpha_cumprod = alpha_cumprod
                self.betas = torch.Tensor(new_betas)

                self.calculate_alphas(self.betas)

            for t in tqdm(
                 reversed(range(len(timesteps))), total=len(timesteps)):

                prediction = model(
                    x.to(self.device),
                    torch.full((n,), timesteps[t]).to(self.device))

                if prediction.shape[1] == self.out_channels * 2:

                    model_mean, model_var = (prediction[:, :self.out_channels],
                                             prediction[:, self.out_channels:])

                    model_mean, model_var = self.p(
                        model_mean, model_var, x,
                        torch.full((n,), t),
                        learned_var=True)

                else:

                    posterior_variance_t = self.extract(
                        self.posterior_variance, torch.full((n,), t), x.shape)
                    model_mean, model_var = (
                        prediction, posterior_variance_t)

                    model_mean, model_var = self.p(
                        model_mean, model_var, x, torch.full((n,), t))

                if t == 0:

                    x = model_mean

                else:

                    noise = torch.randn_like(x)
                    x = (model_mean + torch.sqrt(model_var) * noise)

            model.train()
            x = (x.clamp(-1, 1) + 1) / 2

            # reset old betas after sampling
            self.calculate_alphas(original_betas)
            self.betas = original_betas

            return x

    def mean(self, x_t, x_0, t):

        sqrt_alphas_cumprod_prev_t = self.extract(
            torch.sqrt(self.alphas_cumprod_prev), t, x_0.shape)

        one_minus_alphas_cumprod_t = self.extract(
            1. - self.alphas_cumprod, t, x_0.shape)

        sqrt_alphas = self.extract(
            torch.sqrt(self.alphas), t, x_0.shape)

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

        posterior_log_variance_clipped_t = self.extract(
            self.posterior_log_varaiance_clipped, t, x_0.shape)

        return self.mean(x_t, x_0, t), posterior_log_variance_clipped_t

    def p(self, model_mean, model_var, x_t, t, learned_var=False):

        if learned_var:
            # Equation 15 improved ddpm
            min_log = self.extract(
                self.posterior_log_varaiance_clipped, t, x_t.shape)
            max_log = self.extract(
                torch.log(self.betas), t, x_t.shape)
            frac = (model_var + 1) / 2
            model_log_var = frac * max_log + (1 - frac) * min_log
            model_var = torch.exp(model_log_var)

        sqrt_recip_alphas_cumprod_t = self.extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )

        sqrt_recipm1_alphas_cumprod_t = self.extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        pred_x_0 = (sqrt_recip_alphas_cumprod_t * x_t
                    - sqrt_recipm1_alphas_cumprod_t * model_mean)
        pred_x_0 = pred_x_0.clamp(-1, 1)

        # get q_posterior mean of predicted x_0
        model_mean, __ = self.q_posterior(x_t, pred_x_0, t)

        return model_mean, model_var

    def q(self, x_0, t):

        mean = self.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = self.extract(1.0 - self.alphas_cumprod, t, x_0.shape)

        log_variance = self.extract(
            self.log_one_minus_alphas_cumprod, t, x_0.shape)

        return mean, variance, log_variance


def get_schedule(schedule, beta_0, beta_t, timesteps, s=0.008):

    if schedule == "linear":
        return torch.linspace(beta_0, beta_t, timesteps)
    elif schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


# DEBUG

# diffusion = Diffusion(
#     config["beta_0"],
#     config["beta_t"],
#     config["timesteps"],
#     (128, 128),
#     torch.device("cuda"),
#     config["schedule"]
# )

# kl = diffusion.vlb_loss(
#     torch.zeros((4, 3, 128, 128)),
#     torch.zeros((4, 3, 128, 128)),
#     torch.zeros((4, 3, 128, 128)),
#     torch.zeros((4, 3, 128, 128)),
#     torch.zeros((4, ), dtype=torch.int64)
# )

# print()
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
