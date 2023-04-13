from matplotlib import cm
import torch


def get_rgb(image, colomap="viridis"):

    rgb_image = image.cpu().detach().numpy()
    cmap = cm.get_cmap(colomap)
    rgb_image = cmap(rgb_image)[..., :3]

    return rgb_image


def down_upsample(image, img_channels):

    # downsample
    low_res = torch.nn.functional.interpolate(
                image, (64, 64), mode="area")
    low_res[:, img_channels] = (low_res[:, 2] == 1.0)
    low_res[:, img_channels] = low_res[:, 2] * 2 - 1

    # upsample
    low_res_image = torch.nn.functional.interpolate(
        low_res[:, :img_channels], (256, 256), mode="bilinear")
    low_res_mask = torch.nn.functional.interpolate(
        low_res[:, img_channels:], (256, 256), mode="nearest")
    low_res = torch.cat((low_res_image, low_res_mask), dim=1)

    return low_res
