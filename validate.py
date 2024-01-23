import math
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from losses import HybridLoss
from image_transforms import get_rgb, down_upsample


class Validation():

    def __init__(self, img_channels=3):

        self.fid = FrechetInceptionDistance(
            normalize=True, reset_real_features=False)
        self.img_channels = img_channels

    def fit_real_samples(self, dataloader, channel=None, one_hot=False):
        """
        Fits real samples of a dataloader for FID calculation.
        If channel is not None it converts the given channel to a rgb colormap.
        """

        for batch in tqdm(dataloader):

            if channel != -1 or one_hot:
                if one_hot:
                    masks_argmax = torch.argmax(
                            batch["I"], dim=1).float()
                    map = masks_argmax / (batch["I"].shape[1] - 1)
                elif channel != -1:
                    map = batch["I"][:, channel]
                    map = (map + 1) / 2
                image = torch.Tensor(get_rgb(map))
                image = torch.moveaxis(image, -1, 1)
            else:
                image = (batch["I"][:, :self.img_channels] + 1) / 2
            self.fid.update(image, real=True)

    def valid_hybrid_loss(
            self, model, data_loader, device, diffusion, timesteps,
            pred_type="all", img_channels=3, drop_rate=0.2):

        epoch_loss = 0
        mask = None

        with torch.no_grad():
            model.eval()

            for batch_idx, batch in enumerate(tqdm(data_loader)):

                if pred_type == "all":
                    x_0 = batch["I"].to(device)
                elif pred_type == "mask":
                    x_0 = batch["I"][:, img_channels:].to(device)
                else:
                    x_0 = batch["I"][:, :img_channels].to(device)
                    mask = batch["I"][:, img_channels:].to(device)
                    drop_mask = (
                        torch.rand(
                            (mask.shape[0], 1, 1, 1)) > drop_rate).float()
                    mask *= drop_mask.to(device)

                if "L" in batch.keys():
                    label_dist = batch["L"].to(device)
                    if torch.rand(1) <= drop_rate:
                        label_dist = None
                else:
                    label_dist = None

                t = torch.randint(
                    0, timesteps, (x_0.shape[0],), dtype=torch.int64)
                noisy_image, noise = diffusion.forward_process(x_0, t)

                noisy_image = noisy_image.to(device)
                noise = noise.to(device)

                output = model(noisy_image, t.to(device), mask=mask,
                               label_dist=label_dist)

                true_mean, true_log_var_clipped = diffusion.q_posterior(
                    noisy_image, x_0, t)
                out_mean, out_var = diffusion.p(
                    output[:, :x_0.shape[1]], output[:, x_0.shape[1]:],
                    noisy_image, t, learned_var=True, pred_type=pred_type,
                    img_channels=img_channels)

                loss_fn = HybridLoss()
                loss = loss_fn(
                    noise, output[:, :x_0.shape[1]], x_0, t.to(device),
                    true_mean, true_log_var_clipped, out_mean, out_var)

                epoch_loss += loss.item()

        model.train()
        return epoch_loss / len(data_loader)

    def valid_fid(self, samples, one_hot=False):

        if len(samples.shape) == 3 or one_hot:
            if one_hot:
                masks_argmax = torch.argmax(
                        samples, dim=1).float()
                map = masks_argmax / (samples.shape[1] - 1)
            elif len(samples.shape) == 3:
                map = (samples + 1) / 2
            image = torch.Tensor(get_rgb(map))
            image = torch.moveaxis(image, -1, 1)
        else:
            image = (samples + 1) / 2

        self.fid.reset()
        self.fid.update(image.cpu(), real=False)
        fid = self.fid.compute()

        return fid

    def generate_samples(self, condition, pred_type, img_channels,
                         num_classes,
                         diffusion, sampling_steps, batch_size, model, device,
                         num_samples=1,
                         super_res=False, valid_loader=None, clamp=True,
                         pred_noise=True, guidance_scale=0.2,
                         round_pred_x_0=False):

        sample_dict = {}
        samples = []
        label_dists = []

        # sample
        if condition == "mask":
            generated = 0

            for batch in valid_loader:

                if num_samples - generated < batch_size:
                    n = num_samples - generated
                else:
                    n = batch_size
                sample_mask = batch["I"][:, img_channels:]
                sample_batch = diffusion.sample(
                    model,
                    sample_mask.shape[0],
                    mask=sample_mask.to(
                        device),
                    label_dist=None,
                    sampling_steps=sampling_steps,
                    guidance_scale=guidance_scale,
                    pred_noise=pred_noise,
                    clamp=clamp,
                    round_pred_x_0=round_pred_x_0)
                sample_mask = torch.argmax(sample_mask, dim=1, keepdim=True)
                samples.append(
                    torch.cat((sample_batch.cpu(), sample_mask), dim=1))
                generated += n
                if generated == num_samples:
                    break
            samples = torch.cat(samples)

        elif super_res:

            generated = 0
            low_res_images = []

            for batch in valid_loader:

                if num_samples - generated < batch_size:
                    n = num_samples - generated
                else:
                    n = batch_size

                low_res = down_upsample(batch["I"][:n], img_channels)
                low_res_images.append(low_res)
                sample_batch = samples.append(diffusion.sample(
                    model,
                    n,
                    mask=None,
                    label_dist=None,
                    low_res=low_res.to(device),
                    sampling_steps=sampling_steps,
                    guidance_scale=guidance_scale,
                    pred_noise=pred_noise,
                    clamp=clamp,
                    round_pred_x_0=round_pred_x_0))
                generated += n
                if generated == num_samples:
                    break
            samples = torch.cat(samples)
            low_res_images = torch.cat(low_res_images)

        else:

            generated = 0
            for _ in range(math.ceil(num_samples / batch_size)):

                if num_samples - generated < batch_size:
                    n = num_samples - generated
                else:
                    n = batch_size

                if condition == "label_dist":
                    label_dist = torch.zeros((n, num_classes)).to(device)
                    indices = torch.randint(num_classes, size=(n,))
                    label_dist[torch.arange(n), indices] = 1
                    label_dists.append(label_dist)
                    label_dists = torch.cat(label_dists)
                    sample_dict["label_dists"] = label_dists
                else:
                    label_dist = None

                samples.append(diffusion.sample(
                    model,
                    n,
                    mask=None,
                    label_dist=label_dist,
                    sampling_steps=sampling_steps,
                    guidance_scale=guidance_scale,
                    pred_noise=pred_noise,
                    clamp=clamp,
                    round_pred_x_0=round_pred_x_0))
                generated += n
            samples = torch.cat(samples)

        # split images and masks
        if pred_type == "all" or condition == "mask":
            sample_dict["masks"] = samples[:, img_channels:]
            sample_dict["images"] = samples[:, :img_channels]
        elif pred_type == "mask":
            sample_dict["masks"] = samples
        if super_res:
            sample_dict["low_res"] = low_res_images

        return sample_dict

    def label_error(self, pred_masks, label_dists):

        num_classes = label_dists.shape[1]
        pred_label_dists = []
        for mask in pred_masks:

            pred_label_dist = torch.zeros((num_classes,))
            if mask.shape[0] == num_classes:
                mask = torch.argmax(mask, dim=0)
            mask *= num_classes
            mask[mask == num_classes] = num_classes - 1
            mask = mask.int()
            if torch.unique(mask).shape[0] == num_classes:
                pred_label_dist[-1] = 1.0
            elif torch.unique(mask).shape[0] > 1:
                pred_label_dist[torch.unique(mask).int()[-1].item() - 1] = 1.0

            pred_label_dists.append(pred_label_dist)
        pred_label_dists = torch.stack(pred_label_dists)

        accuracy = (pred_label_dists.to(label_dists.device).argmax(dim=1) ==
                    label_dists.argmax(dim=1))

        return accuracy.float().mean()
