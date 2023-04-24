# Data config

dataset = "wear"

grain_defaults = {
    "root_dir": "data/grains_txt",
    "channel_names": ["intensity", "depth"],
    "image_idxs": ([7, 2, 5, 8, 6, 10, 1, 4], [9]),
    "img_channels": 2,
    "num_classes": 2,
    "patch_size": 256
}
wear_defaults = {
    "root_dir": "data/RT100U_processed",
    "raw_img_size": (448, 576),
    "img_channels": 3,
    "num_classes": 3
}

img_size = (256, 256)

local = False
use_wandb = True

checkpoint = None
save_models = True

# Model config

beta_0 = 0.0001
beta_t = 0.02
timesteps = 1000
schedule = "linear"
model_dim = 64
dim_mults = (1, 1, 2, 2, 4, 4)
num_resnet_blocks = 2
dropout = 0.1
drop_condition_rate = 0.2
guidance_scale = 1.5
clamp = True
pred_noise = True
round_pred_x_0 = True

# Data config

mask_one_hot = False
pred_type = "mask"  # "all, mask or image"
condition = "None"  # "None, label_dist or mask"
super_res = False


# Train config

batch_size = 64
optimizer = "AdamW"
learning_rate = 0.00001
epochs = 3000
ema = False
num_workers = 32
loss = "hybrid"


# Eval config

evaluate_every = 50
start_eval_epoch = 0
log_best = False
sampling_steps = 100
num_samples = 50
round_masks = False

random_seed = 1234

if local:
    num_workers = 0
    batch_size = 2
    save_models = False

config = {
    "dataset": dataset,
    "grain_defaults": grain_defaults,
    "wear_defaults": wear_defaults,
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
    "start_eval_epoch": start_eval_epoch,
    "log_best": log_best,
    "sampling_steps": sampling_steps,
    "num_samples": num_samples,
    "round_masks": round_masks,
    "use_wandb": use_wandb,
    "checkpoint": checkpoint,
    "save_models": save_models,
    "beta_0": beta_0,
    "beta_t": beta_t,
    "timesteps": timesteps,
    "schedule": schedule,
    "model_dim": model_dim,
    "dim_mults": dim_mults,
    "num_resnet_blocks": num_resnet_blocks,
    "dropout": dropout,
    "drop_condition_rate": drop_condition_rate,
    "guidance_scale": guidance_scale,
    "mask_one_hot": mask_one_hot,
    "pred_type": pred_type,
    "condition": condition,
    "clamp": clamp,
    "pred_noise": pred_noise,
    "round_pred_x_0": round_pred_x_0,
    "super_res": super_res,
}
