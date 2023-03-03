# Data config

train_dataset = "data/RT100U_processed"

raw_img_size = (448, 576)
img_size = (64, 64)
img_channels = 3

local = True
use_wandb = False

checkpoint = None

# Model config

beta_0 = 0.0001
beta_t = 0.02
timesteps = 1000
schedule = "linear"
model_dim = 64
dim_mults = (1, 1, 2, 2, 4, 4)
num_resnet_blocks = 2
dropout = 0.1

mask_one_hot = False
pred_type = "all"  # "all, mask or image"
condition = "label_dist"  # "None, label_dist or mask"
num_classes = 3
label_norm = ([0.97099304, 0., 0.], [0.99629211, 0.0241394, 0.02012634])


# Train config

batch_size = 4
optimizer = "Adam"
loss = "MSELoss"
learning_rate = 0.00001
epochs = 1000
ema = False
num_workers = 32
loss = "hybrid"


# Eval config

evaluate_every = 50
start_eval_epoch = 300
sampling_steps = 100

random_seed = 1234

if local:
    num_workers = 0
    batch_size = 1

config = {
    "train_dataset": train_dataset,
    "raw_img_size": raw_img_size,
    "img_size": img_size,
    "img_channels": img_channels,
    "batch_size": batch_size,
    "optimizer": optimizer,
    "loss": loss,
    "random_seed": random_seed,
    "epochs": epochs,
    "ema": ema,
    "num_workers": num_workers,
    "learning_rate": learning_rate,
    "evaluate_every": evaluate_every,
    "start_eval_epoch": start_eval_epoch,
    "sampling_steps": sampling_steps,
    "use_wandb": use_wandb,
    "checkpoint": checkpoint,
    "beta_0": beta_0,
    "beta_t": beta_t,
    "timesteps": timesteps,
    "schedule": schedule,
    "model_dim": model_dim,
    "dim_mults": dim_mults,
    "num_resnet_blocks": num_resnet_blocks,
    "dropout": dropout,
    "mask_one_hot": mask_one_hot,
    "pred_type": pred_type,
    "condition": condition,
    "num_classes": num_classes,
    "label_norm": label_norm
}
