import torch


def create_optimizer(parameters, optimizer_name, learning_rate, weight_decay):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, scheduler_name):
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def step_scheduler(scheduler, metric):
    if scheduler is None:
        return

    scheduler.step(metric)


def get_current_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]
