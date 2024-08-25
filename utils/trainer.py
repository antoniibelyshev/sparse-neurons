import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable
from .bayesian_nn import BayesianLinear

import wandb

import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.99


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader[tuple[Tensor, Tensor]],
            test_loader: DataLoader[tuple[Tensor, Tensor]],
            optimizer: Optimizer,
            scheduler: LRScheduler,
            *,
            criterion: Callable[[Tensor, Tensor], Tensor] = nn.CrossEntropyLoss(),
            epochs: int = 100,
            device: torch.device = DEVICE,
            reg_coef_lambda: Callable[[int], float] = lambda epoch: 1.0,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.n_train = len(train_loader.dataset) # type: ignore
        self.test_loader = test_loader
        self.n_test = len(test_loader.dataset) # type: ignore
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epochs = epochs
        self.reg_coef_lambda = reg_coef_lambda

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return self.criterion(output, target)

    def log(self, **metrics: float) -> None:
        wandb.log(metrics) # type: ignore

    def train(self, name="default experiment", project: str = "neuron sparsity", entity: str = "antonii-belyshev") -> None:
        run = wandb.init(name=name, project=project, entity=entity) # type: ignore

        for epoch in range(self.epochs):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                if hasattr(self.model, "kl"):
                    reg = self.model.kl() / self.n_train
                    total_loss = loss + self.reg_coef_lambda(epoch) * reg
                    total_loss.backward()
                    self.log(loss=loss.item(), reg=reg.item())
                else:
                    loss.backward() # type: ignore
                    self.log(loss=loss.item())
                self.optimizer.step()
    
            
            self.log(**self.test(epoch))

            self.scheduler.step()

        run.finish() # type: ignore

    def test(self, epoch: int) -> dict[str, float]:
        metrics: dict[str, float] = {}

        self.model.eval()

        loss = 0.0
        correct = 0.0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            metrics["eval_loss"] = loss / len(self.test_loader)
            metrics["accuracy"] = 100. * correct / self.n_test

        metrics["epoch"] = epoch

        return metrics


class BayesianModelTrainer(Trainer):
    def test(self, epoch: int) -> dict[str, float]:
        metrics = super().test(epoch)

        self.model.eval()

        with torch.no_grad():
            total_neurons = torch.zeros(1, device=self.device)
            sparsed_neurons = torch.zeros(1, device=self.device)

            bayesian_conv_layers = [layer for layer in self.model.get_bayesian_layers() if isinstance(layer, BayesianLinear)]

            for i, layer in enumerate(bayesian_conv_layers):
                mask = layer.equivalent_dropout_rate < threshold
                sparsity = 1 - mask.float().mean()
                neuron_sparsity = (mask.sum(list(range(1, mask.dim()))) == 0).float().mean()

                total_neurons += mask.shape[0]
                sparsed_neurons += (mask.sum(list(range(1, mask.dim()))) == 0).int().sum().item()

                metrics[f"sparsity_{i}"] = sparsity.item()
                metrics[f"neuron_sparsity_{i}"] = neuron_sparsity.item()

                if epoch % 10 == 0:
                    min_dropout_rate = layer.equivalent_dropout_rate.amin(list(range(1, mask.dim())))

                    plt.figure(figsize=(10, 6))
                    sorted_values = sorted(1 - min_dropout_rate.detach().cpu(), reverse=True)
                    plt.bar(range(len(sorted_values)), sorted_values)
                    plt.xlabel('Neuron Index')
                    plt.ylabel('1 - max(equivalent_dropout_rate)')
                    plt.title('Sorted 1 - max(equivalent_dropout_rate) per Layer')
                    plt.yscale('log')

                    plt.tight_layout()
                    plt_path = "sparsity_plot.png"
                    plt.savefig(plt_path)

                    wandb.log({f"sparsity_plot layer {i}": wandb.Image(plt_path)})

                    plt.close()

            total_neuron_sparsity = 100. * sparsed_neurons.item() / total_neurons.item()
            metrics["total_neuron_sparsity"] = total_neuron_sparsity

        return metrics
