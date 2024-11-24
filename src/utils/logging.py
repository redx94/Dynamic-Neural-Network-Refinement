
import wandb

def setup_wandb(project_name):
    wandb.init(project=project_name)
    print(f"W&B initialized for project: {project_name}")
