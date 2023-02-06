from main import main
import wandb

sweep_config = {
    "method": "random"
}

parameters_dict = {
    "batch_size": {
        "values": [4, 8, 16, 32, 64, 128]},
    "learning_rate": {
        "values": [0.00001, 0.0001]},
    "timesteps": {
        "values": [1000, 2000, 3000, 4000]},
    "model_dim": {
        "values": [64, 96, 128, 192]},
    "img_size": {
        "values": [(64, 64), (128, 128)]},
    "schedule": {
        "values": ["linear", "cosine"]}
}

sweep_config["parameters"] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="wear_generation")
wandb.agent(sweep_id, main, count=100, project="wear_generation")
