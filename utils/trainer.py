import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable
from .bayesian_nn import BayesianLinear

import wandb


class Trainer:
    """
    A class for training and testing neural network models.

    Args:
        model (nn.Module): The model to be trained and evaluated.
        train_loader (DataLoader[tuple[Tensor, Tensor]]): DataLoader for training data.
        test_loader (DataLoader[tuple[Tensor, Tensor]]): DataLoader for testing data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        scheduler (LRScheduler): Learning rate scheduler.
        criterion (Callable[[Tensor, Tensor], Tensor], optional): Loss function. Default is nn.CrossEntropyLoss().
        epochs (int, optional): Number of training epochs. Default is 100.
        device (torch.device, optional): Device to run the model on. Default is CUDA if available, else CPU.
        reg_coef_lambda (Callable[[int], float], optional): Regularization coefficient as a function of epoch. Default is lambda _: 1.0.
    """

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
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            reg_coef_lambda: Callable[[int], float] = lambda _: 1.0,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.n_train = len(train_loader.dataset)  # type: ignore
        self.test_loader = test_loader
        self.n_test = len(test_loader.dataset)  # type: ignore
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epochs = epochs
        self.reg_coef_lambda = reg_coef_lambda

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Computes the loss for given outputs and targets.

        Args:
            output (Tensor): Model predictions.
            target (Tensor): Ground truth labels.

        Returns:
            Tensor: Computed loss.
        """
        return self.criterion(output, target)

    def log(self, **metrics: float) -> None:
        """
        Logs metrics to WandB.

        Args:
            **metrics (float): Metrics to log.
        """
        wandb.log(metrics)  # type: ignore

    def train(
            self,
            entity: str = "antonii-belyshev",
            project: str = "neuron sparsity",
            name: str = "default experiment",
    ) -> None:
        """
        Trains the model and logs metrics to WandB.

        Args:
            entity (str, optional): WandB entity name. Default is "antonii-belyshev".
            project (str, optional): WandB project name. Default is "neuron sparsity".
            name (str, optional): WandB run name. Default is "default experiment".
        """
        run = wandb.init(name=name, project=project, entity=entity)  # type: ignore

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
                    loss.backward()  # type: ignore
                    self.log(loss=loss.item())
                self.optimizer.step()

            self.log(**self.test(epoch))
            self.scheduler.step()

        run.finish()  # type: ignore

    def test(self, epoch: int | None = None) -> dict[str, float]:
        """
        Evaluates the model on the test dataset.

        Args:
            epoch (int | None, optional): Current epoch number. Default is None.

        Returns:
            dict[str, float]: Dictionary of evaluation metrics including loss and accuracy.
        """
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

        if epoch is not None:
            metrics["epoch"] = epoch

        return metrics


class BayesianModelTrainer(Trainer):
    """
    Trainer for Bayesian models with additional sparsity metrics.

    Args:
        See Trainer for arguments.
    """

    def test(self, epoch: int | None = None) -> dict[str, float]:
        """
        Evaluates the Bayesian model on the test dataset and computes sparsity metrics.

        Args:
            epoch (int | None, optional): Current epoch number. Default is None.

        Returns:
            dict[str, float]: Dictionary of evaluation metrics including loss, accuracy, and neuron sparsity.
        """
        metrics = super().test(epoch)

        self.model.eval()

        with torch.no_grad():
            total_neurons = torch.zeros(1, device=self.device)
            sparsed_neurons = torch.zeros(1, device=self.device)

            bayesian_conv_layers = [layer for layer in self.model.get_bayesian_layers() if isinstance(layer, BayesianLinear)]

            for i, layer in enumerate(bayesian_conv_layers):
                sparsity = 1 - layer.get_weight_mask().float().mean()
                neuron_sparsity = 1 - layer.get_out_mask().float().mean()

                total_neurons += layer.out_features
                sparsed_neurons += int(layer.out_features * neuron_sparsity)

                metrics[f"sparsity_{i}"] = sparsity.item()
                metrics[f"neuron_sparsity_{i}"] = neuron_sparsity.item()

            total_neuron_sparsity = sparsed_neurons.item() / total_neurons.item()
            metrics["total_neuron_sparsity"] = total_neuron_sparsity

        return metrics
