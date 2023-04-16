"""Base Model"""
from dataclasses import dataclass, field
import logging
from abc import ABCMeta, abstractmethod
from random import choices
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig, OmegaConf

from dl.attention import (
    GeneralAttention,
    AdditiveAttention,
    DotProductAttention,
    ConcatAttention,
)

# from attention import (
#     GeneralAttention,
#     AdditiveAttention,
#     DotProductAttention,
#     ConcatAttention,
# )

ATTENTION_TYPES = {
    "dot": DotProductAttention,
    "scaled_dot": DotProductAttention,
    "general": GeneralAttention,
    "additive": AdditiveAttention,
    "concat": ConcatAttention,
}


@dataclass
class SingleStepRNNConfig:
    """Configuration for RNN"""

    rnn_type: str
    input_size: int
    hidden_size: int
    num_layers: int
    bidirectional: bool
    learning_rate: float = field(default=1e-3)
    optimizer_params: Dict = field(default_factory=dict)
    lr_scheduler: Optional[str] = field(default=None)
    lr_scheduler_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.rnn_type = self.rnn_type.upper()
        assert self.rnn_type in [
            "LSTM",
            "GRU",
            "RNN",
        ], f"{self.rnn_type} is not a valid RNN type"


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__()
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        self.save_hyperparameters(config)
        # The concatenated output dim of the embedding layer
        self._build_network()
        self._setup_loss()
        self._setup_metrics()

    @abstractmethod
    def _build_network(self):
        pass

    def _setup_loss(self):
        self.loss = nn.MSELoss()

    def _setup_metrics(self):
        self.metrics = [torchmetrics.functional.mean_absolute_error]
        self.metrics_name = ["MAE"]

    def calculate_loss(self, y_hat, y, tag):
        computed_loss = self.loss(y_hat, y)
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag == "valid") or (tag == "test"),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
        )
        return computed_loss

    def calculate_metrics(self, y, y_hat, tag):
        metrics = []
        for metric, metric_str in zip(self.metrics, self.metrics_name):
            avg_metric = metric(y_hat, y)
            self.log(
                f"{tag}_{metric_str}",
                avg_metric,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=True,
            )
        return metrics

    @abstractmethod
    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        pass

    @abstractmethod
    def predict(
        self, batch: Tuple[torch.Tensor, torch.Tensor], ret_model_output: bool = False
    ):
        pass

    def training_step(self, batch, batch_idx):
        y_hat, y = self.forward(batch)
        loss = self.calculate_loss(y_hat, y, tag="train")
        _ = self.calculate_metrics(y, y_hat, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():

            y_hat, y = self.forward(batch)
            _ = self.calculate_loss(y_hat, y, tag="valid")
            _ = self.calculate_metrics(y, y_hat, tag="valid")
        return y_hat, y

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            y_hat, y = self.forward(batch)
            _ = self.calculate_loss(y_hat, y, tag="test")
            _ = self.calculate_metrics(y, y_hat, tag="test")
        return y_hat, y

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            pred = self.predict(batch)
        return pred

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            **self.hparams.optimizer_params,
        )
        if self.hparams.lr_scheduler is not None:
            try:
                self._lr_scheduler = getattr(
                    torch.optim.lr_scheduler, self.hparams.lr_scheduler
                )
            except AttributeError as e:
                print(
                    f"{self.hparams.lr_scheduler} is not a valid learning rate sheduler defined in the torch.optim.lr_scheduler module"
                )
                raise e
            if isinstance(self._lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(
                        opt, **self.hparams.lr_scheduler_params
                    ),
                }
            else:
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(
                        opt, **self.hparams.lr_scheduler_params
                    ),
                    "monitor": self.hparams.lr_scheduler_monitor_metric,
                }
        else:
            return opt


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout=0.0, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


#CHECK Multistep Horizon!!!!

@dataclass
class TransformerConfig:
    """Configuration for Transformer"""

    input_size: int
    d_model: int
    n_heads: int
    n_layers: int
    ff_multiplier: int = 4
    activation: str = "relu"  # 'gelu'
    multi_step_horizon: int = 1
    dropout: float = 0.0
    learning_rate: float = field(default=1e-3)
    optimizer_params: Dict = field(default_factory=dict)
    lr_scheduler: Optional[str] = field(default=None)
    lr_scheduler_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.activation in [
            "relu",
            "gelu",
        ], "Invalid activation. Should be relu or gelu"


class TransformerModel(BaseModel):
    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__(config)

    def _build_network(self):
        self.input_projection = nn.Linear(
            self.hparams.input_size, self.hparams.d_model, bias=False
        )
        self.pos_encoder = PositionalEncoding(self.hparams.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.d_model,
            nhead=self.hparams.n_heads,
            dropout=self.hparams.dropout,
            dim_feedforward=self.hparams.d_model * self.hparams.ff_multiplier,
            activation=self.hparams.activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.hparams.n_layers
        )
        # self.decoder = nn.Linear(self.hparams.d_model, self.hparams.multi_step_horizon)
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.d_model, 100),
            nn.ReLU(),
            nn.Linear(100, self.hparams.multi_step_horizon*8),
        )
        self._src_mask = None

    def _generate_square_subsequent_mask(self, sz, reset_mask=False):
        if self._src_mask is None or reset_mask:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self._src_mask = mask
        return self._src_mask

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        mask = self._generate_square_subsequent_mask(x.shape[1]).to(x.device)
        # Projecting input dimension to d_model
        x_ = self.input_projection(x)
        # Adding positional encoding
        x_ = self.pos_encoder(x_)
        # Encoding the input
        x_ = self.transformer_encoder(x_, mask)
        # Decoding the input
        y_hat = self.decoder(x_)
        # constructing a shifted by one target so that all the outputs from the decoder can be trained
        # also unfolding so that at each position we can train all H horizon forecasts
        y = torch.cat([x[:, 1:, :], y], dim=1).squeeze(-1).unfold(1, y.size(1), 1)

        return y_hat, y

    def predict(
        self, batch: Tuple[torch.Tensor, torch.Tensor], ret_model_output: bool = False
    ):
        with torch.no_grad():
            y_hat, _ = self.forward(batch)
            # We only need the last position prediction in prediction task
            y_hat = y_hat[:, -1, :].unsqueeze(1)
        return y_hat

