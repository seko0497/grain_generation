import numpy as np
import torch


def normal_kl(mean1, logvar1, mean2, logvar2):

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )


def approx_standard_normal_cdf(x):

    return 0.5 * (
        1.0 + torch.tanh(
            np.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(
                cdf_delta.clamp(min=1e-12))),
    )

    return log_probs


class HybridLoss(torch.nn.Module):

    def __init__(self, lam=0.001, pred_noise=True) -> None:
        super().__init__()

        self.mse = torch.nn.MSELoss()
        self.lam = lam
        self.pred_noise = pred_noise

    def vlb_loss(self, true_mean, true_var, out_mean, out_var, x_0, t):

        kl = normal_kl(true_mean, true_var, out_mean, out_var)
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=out_mean, log_scales=0.5 * out_var
        )
        decoder_nll = decoder_nll.mean(
            dim=list(range(1, len(decoder_nll.shape)))) / np.log(2.0)

        loss = torch.where((t == 0), decoder_nll, kl)

        return loss

    def forward(self, noise, noise_pred, x_0, t, true_mean, true_var, out_mean,
                out_var,):

        target = noise if self.pred_noise else x_0
        loss_simple = self.mse(noise_pred, target)
        loss_vlb = self.vlb_loss(
            true_mean, true_var, out_mean.detach(), out_var, x_0, t
        )

        loss_hybrid = loss_simple + self.lam * loss_vlb

        return loss_hybrid.mean()
