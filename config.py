from json import encoder
from torch import embedding
import wandb


def get_config():

    # Data config

    train_dataset = "data/RT100U_processed"

    img_size = (448, 576)

    # Model config

    beta_0 = 0.0001
    beta_t = 0.015
    timesteps = 500
    time_emb_dim = 32

    # Train config

    batch_size = 2
    optimizer = "Adam"
    loss = "MSELoss"
    learning_rate = 0.00001
    epochs = 200
    num_workers = 0

    # Eval config

    evaluate_every = 10

    random_seed = 1234
    use_wandb = False

    config = {
        "train_dataset": train_dataset,
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
        "time_emb_dim": time_emb_dim
    }

    return config
