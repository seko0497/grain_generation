from json import encoder
from torch import embedding
import wandb


def get_config():

    # Data config

    train_dataset = "data/RT100U_processed"

    raw_img_size = (448, 576)
    img_size = (128, 128)

    local = False

    # Model config

    beta_0 = 0.0001
    beta_t = 0.02
    timesteps = 1000
    schedule = "cosine"
    model_dim = 128

    # Train config

    batch_size = 32
    optimizer = "Adam"
    loss = "MSELoss"
    learning_rate = 0.00001
    epochs = 10000
    num_workers = 32

    # Eval config

    evaluate_every = 10

    random_seed = 1234
    use_wandb = True

    if local:
        num_workers = 0
        batch_size = 2

    config = {
        "train_dataset": train_dataset,
        "raw_img_size": raw_img_size,
        "img_size": img_size,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "loss": loss,
        "random_seed": random_seed,
        "epochs": epochs,
        "num_workers": num_workers,
        "learning_rate": learning_rate,
        "evaluate_every": evaluate_every,
        "use_wandb": use_wandb,
        "beta_0": beta_0,
        "beta_t": beta_t,
        "timesteps": timesteps,
        "schedule": schedule,
        "model_dim": model_dim
    }

    return config
