import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Iterable
from .nn import BayesianLinear

import wandb


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExponentialMovingAverage:
    def __init__(self, parameters: Iterable[Tensor], decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.params = parameters

    def update(self) -> None:
        for shadow_param, param in zip(self.shadow_params, self.params):
            shadow_param.data = self.decay * shadow_param.data + (1.0 - self.decay) * param.data

    def store(self) -> None:
        self.backup_params = [p.clone().detach() for p in self.params]

    def restore(self) -> None:
        for backup_param, param in zip(self.backup_params, self.params):
            param.data.copy_(backup_param.data)

    def copy_to(self) -> None:
        for shadow_param, param in zip(self.shadow_params, self.params):
            param.data.copy_(shadow_param.data)


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader[tuple[Tensor, Tensor]],
            n_train: int,
            test_loader: DataLoader[tuple[Tensor, Tensor]],
            n_test: int,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            *,
            criterion: Callable[[Tensor, Tensor], Tensor] = nn.CrossEntropyLoss(),
            epochs: int = 100,
            log_interval: int = 100,
            device: torch.device = DEVICE,
            reg_coef_lambda: Callable[[int], float] = lambda epoch: 1.0,
            decay: float = 0.999,
        ) -> None:

        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.n_train = n_train
        self.test_loader = test_loader
        self.n_test = n_test
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epochs = epochs
        self.log_interval = log_interval
        self.reg_coef_lambda = reg_coef_lambda

        self.ema = ExponentialMovingAverage(list(self.model.parameters()), decay=decay)

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return self.criterion(output, target)

    def log_metrics(self, **metrics: float) -> None:
        wandb.log(metrics) # type: ignore

    def train(self, project: str = "neuron sparsity", entity: str = "antoniibelyshev") -> None:
        wandb.init(project=project, entity=entity) # type: ignore

        for epoch in range(self.epochs):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                reg = self.model.kl() / self.n_train
                total_loss = loss + self.reg_coef_lambda(epoch) * reg
                total_loss.backward()
                self.optimizer.step()

                self.ema.update()

                self.log_metrics(loss=loss.item(), reg=reg.item())

            self.test(epoch)

            self.scheduler.step()

    def test(self, epoch: int) -> None:
        metrics: dict[str, float] = {}

        self.model.eval()
        self.ema.store()
        self.ema.copy_to()

        loss = torch.zeros(1, device=self.device)
        correct = torch.zeros(1, device=self.device)
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            metrics["loss"] = loss.item() / len(self.test_loader)
            metrics["accuracy"] = 100. * correct.item() / self.n_test

            total_neurons = torch.zeros(1, device=self.device)
            sparsed_neurons = torch.zeros(1, device=self.device)
            
            bayesian_conv_layers = [layer for layer in self.model.features_layers if isinstance(layer, BayesianLinear)]
            for i, layer in enumerate(bayesian_conv_layers):
                mask = layer.get_mask()
                sparsity = mask.float().mean()
                neuron_sparsity = mask.all(1 if len(mask.shape) == 2 else (1, 2, 3)).float().mean()

                total_neurons += mask.shape[0]
                sparsed_neurons += mask.all(1 if len(mask.shape) == 2 else (1, 2, 3)).sum().item()

                metrics[f"sparsity_{i}"] = sparsity.item()
                metrics[f"neuron_sparsity_{i}"] = neuron_sparsity.item()

            total_sparsity = 100. * sparsed_neurons.item() / total_neurons.item()

            metrics["total_sparsity"] = total_sparsity

        self.ema.restore()

        self.log_metrics(**metrics)
