import torch
import wandb
from tqdm import tqdm


class Diffusion:

    def __init__(self, beta_0, beta_t, timesteps, img_size, device, schedule,
                 sampling_steps=None, use_wandb=False):

        self.use_wandb = use_wandb

        self.timesteps = timesteps
        self.image_size = img_size
        self.sampling_steps = sampling_steps

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

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (self.betas *
                                   (1. - self.alphas_cumprod_prev) /
                                   (1. - self.alphas_cumprod))

        self.posterior_log_varaiance_clipped = torch.log(
            torch.cat(
                (self.posterior_variance[1].view(1),
                 self.posterior_variance[1:]))
        )

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

    def sample(self, model, n, epoch):

        model.eval()
        with torch.no_grad():

            samples = torch.Tensor().to(self.device)

            x = torch.randn((n, 3, self.image_size[0], self.image_size[1]))
            x = x.to(self.device)
            samples = torch.cat((samples, x[0]), dim=2)

            timesteps = (self.timesteps if self.sampling_steps is None
                         else self.sampling_steps)

            for t in tqdm(
                 reversed(range(timesteps)), total=timesteps):

                prediction = model(
                    x.to(self.device), torch.full((n,), t).to(self.device))

                if prediction.shape[1] == 6:

                    model_mean, model_var = (prediction[:, :3],
                                             prediction[:, 3:])

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

                if t % (timesteps / 10) == 0:
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
