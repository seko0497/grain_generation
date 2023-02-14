from json import encoder
from torch import embedding
import wandb

# Data config

train_dataset = "data/RT100U_processed"

raw_img_size = (448, 576)
img_size = (64, 64)

local = False
use_wandb = True

# Model config

beta_0 = 0.0001
beta_t = 0.02
timesteps = 2000
schedule = "cosine"
model_dim = 128
dim_mults = (1, 2, 4, 8)
num_resnet_blocks = 2


# Train config

batch_size = 32
optimizer = "Adam"
loss = "MSELoss"
learning_rate = 0.00001
epochs = 2000
ema = False
num_workers = 12
loss = "hybrid"

# Eval config

evaluate_every = 10

random_seed = 1234

if local:
    num_workers = 0
    batch_size = 1

config = {
    "train_dataset": train_dataset,
    "raw_img_size": raw_img_size,
    "img_size": img_size,
    "batch_size": batch_size,
    "optimizer": optimizer,
    "loss": loss,
    "random_seed": random_seed,
    "epochs": epochs,
    "ema": ema,
    "num_workers": num_workers,
    "learning_rate": learning_rate,
    "evaluate_every": evaluate_every,
    "use_wandb": use_wandb,
    "beta_0": beta_0,
    "beta_t": beta_t,
    "timesteps": timesteps,
    "schedule": schedule,
    "model_dim": model_dim,
    "dim_mults": dim_mults,
    "num_resnet_blocks": num_resnet_blocks
}
