from matplotlib import cm


def get_rgb(image, colomap="viridis"):

    rgb_image = image.cpu().detach().numpy()
    cmap = cm.get_cmap(colomap)
    rgb_image = cmap(rgb_image)[..., :3]

    return rgb_image
