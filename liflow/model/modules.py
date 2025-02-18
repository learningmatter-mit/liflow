import lightning as L
import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch

from liflow.model.models import DualPaiNN


class FlowModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = DualPaiNN(
            num_features=cfg.model.num_features,
            num_radial_basis=cfg.model.num_radial_basis,
            num_layers=cfg.model.num_layers,
            num_elements=cfg.model.num_elements,
            r_max=cfg.model.r_max,
            r_offset=cfg.model.r_offset,
            ref_temp=cfg.model.ref_temp,
        )
        self.save_hyperparameters()

    def setup(self, stage: str):
        pass

    def forward(self, batch: Batch, t_per_graph: torch.Tensor) -> torch.Tensor:
        batch["time"] = t_per_graph[batch.batch]
        return self.model(batch)

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # Sum over the xyz dimensions and average over the nodes
        return (outputs - targets).pow(2).sum(-1).mean()

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        # Sample flow time
        t = torch.rand((batch.num_graphs,), device=self.device, dtype=torch.float)

        # Interpolate
        x_cond = batch["positions_1"].clone()
        disp = (batch["positions_2"] - batch["positions_1"]).clone()
        batch["positions_1"] = batch["positions_1"] + batch["prior"]

        _t = t[batch.batch][:, None]
        x_t = (1.0 - _t) * batch["positions_1"] + _t * batch["positions_2"]
        dx_dt = batch["positions_2"] - batch["positions_1"]

        # Compute the flow and loss
        batch["positions_1"] = x_cond
        batch["positions_2"] = x_t
        outputs = self(batch, t_per_graph=t)
        if self.cfg.model.prediction_mode == "velocity":
            loss = self.compute_loss(outputs, dx_dt)
        elif self.cfg.model.prediction_mode == "data":
            loss = self.compute_loss(outputs, disp)
        else:
            raise ValueError(
                f"Unknown prediction mode: {self.cfg.model.prediction_mode}"
            )

        # Log the loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def configure_optimizers(self):
        if self.cfg.optimizer.class_name == "Adam":
            optimizer_class = torch.optim.Adam
        else:
            raise ValueError(
                f"Unknown optimizer class: {self.cfg.optimizer.class_name}"
            )
        return optimizer_class(self.model.parameters(), lr=self.cfg.optimizer.lr)

    @torch.no_grad()
    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        # Preset the interpolation parameter for consistency
        t_list = torch.linspace(
            0.0,
            1.0,
            self.cfg.valid.num_time_steps,
            device=self.device,
            dtype=torch.float,
        )
        orig_batch = batch.clone()

        # Compute the loss for each time step
        loss_list = []
        for t in t_list:
            # Sample flow time
            t_per_graph = t.expand(batch.num_graphs).clone()

            # Interpolate
            x_cond = orig_batch["positions_1"].clone()
            disp = (orig_batch["positions_2"] - orig_batch["positions_1"]).clone()
            batch["positions_1"] = (
                orig_batch["positions_1"] + orig_batch["prior"]
            ).clone()
            batch["positions_2"] = orig_batch["positions_2"].clone()
            _t = t_per_graph[batch.batch][:, None]
            x_t = (1.0 - _t) * batch["positions_1"] + _t * batch["positions_2"]
            dx_dt = batch["positions_2"] - batch["positions_1"]

            # Compute the flow and loss
            batch["positions_1"] = x_cond
            batch["positions_2"] = x_t
            outputs = self(batch, t_per_graph=t_per_graph)
            if self.cfg.model.prediction_mode == "velocity":
                loss = self.compute_loss(outputs, dx_dt)
            elif self.cfg.model.prediction_mode == "data":
                loss = self.compute_loss(outputs, disp)
            loss_list.append(loss)

        loss = torch.stack(loss_list).mean()
        self.log("valid/loss", loss, on_epoch=True, batch_size=batch.num_graphs)
        return loss
