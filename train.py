import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import os

import hydra
from omegaconf import DictConfig

import utils


# torch.autograd.set_detect_anomaly(True)


def train_model(cfg: DictConfig, prefix: str | None = None) -> utils.Trainer:
    model_name = cfg.meta.model
    prefix = cfg.meta.prefix if prefix is None else prefix
    print(f"Training {prefix}{model_name}...")

    model_kwargs = cfg.model[model_name]
    if prefix == "":
        model: nn.Module = getattr(utils, model_name)(**model_kwargs)

    elif prefix == "Bayesian":
        model: nn.Module = getattr(utils, "Bayesian" + model_name)(**model_kwargs)
        if hasattr(model, "from_pretrained"):
            pretrained_model: nn.Module = getattr(utils, model_name)(**model_kwargs)
            pretrained_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models", f"{model_name}.pth"))) # type: ignore
            model.from_pretrained(pretrained_model)

    elif prefix == "Squeezed":
        bayesian_model = getattr(utils, "Bayesian" + model_name)(**model_kwargs)
        bayesian_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models", f"Bayesian{model_name}.pth"))) # type: ignore

        model_kwargs = model_kwargs.copy()
        model_kwargs.cfg = bayesian_model.squeezed_cfg()

        model: nn.Module = getattr(utils, model_name)(**model_kwargs)

    else:
        raise AssertionError(f"Unsupported prefix: {prefix}")
    
    model_name_ = (prefix if prefix == "Bayesian" else "") + model_name


    optim_class = cfg.optim[model_name_].optim_class
    optim_kwargs = cfg.optim[model_name_].optim_kwargs
    optimizer: Optimizer = getattr(optim, optim_class)(model.parameters(), **optim_kwargs)

    scheduler_class = cfg.optim[model_name_].scheduler_class
    scheduler_kwargs = cfg.optim[model_name_].scheduler_kwargs
    scheduler: LRScheduler = getattr(optim.lr_scheduler, scheduler_class)(optimizer, **scheduler_kwargs)

    train_loader, test_loader = utils.get_dataloaders(cfg.meta.dataset, **cfg.data[cfg.meta.dataset])

    if model_name_ == "BayesianVGG":
        def reg_coef_lambda(epoch: int) -> float:
            return min(1., 0.1 ** (2 - epoch // 100 / 2))
            # return min(0.3 ** (4 - epoch // 100), 1.)
            # return min(0.1 ** (2 - epoch // 200), 1.)
            # return min(1., 0.1 ** (2 - epoch // 200))
            # return min(1., 0.5 ** (8 - epoch // 50))
    elif model_name_ == "BayesianLeNet":
        def reg_coef_lambda(epoch: int) -> float:
            if epoch < 10:
                return 0.
            elif epoch < 60:
                return 2 ** ((epoch - 10) / 50) - 1
            else:
                return 1.
        # def reg_coef_lambda(epoch: int) -> float:
        #     return min(1., epoch / 50)
    else:
        def reg_coef_lambda(epoch: int) -> float:
            return 1.

    trainer = (utils.BayesianModelTrainer if prefix == "Bayesian" else utils.Trainer)(model, train_loader, test_loader, optimizer, scheduler, reg_coef_lambda=reg_coef_lambda, **cfg.trainer[model_name_])

    wandb_info = cfg.meta.wandb_info
    if wandb_info.name is None:
        wandb_info = wandb_info.copy()
        wandb_info.name = prefix + model_name

    trainer.train(**wandb_info)

    test_metrics = trainer.test()

    print("Final metrics")
    for key, value in test_metrics.items():
        print(key + ":", value)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), "models", prefix + model_name + "1.pth")) # type: ignore

    return trainer


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.meta.prefix == "All":
        for prefix in ["", "Bayesian", "Squeezed"]:
            train_model(cfg, prefix=prefix)

    else:
        train_model(cfg)


if __name__ == "__main__":
    main()
