import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from liflow.data.modules import NoisedFrameDataModule, TimeDelayedPairDataModule
from liflow.model.modules import FlowModule
from liflow.utils.ema import EMA, EMAModelCheckpoint
from liflow.utils.prior import get_prior

torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg):
    # Raise error if data paths are not provided
    if cfg.data.data_path is None:
        raise ValueError("Config: data.data_path must be provided")

    # Task selection
    if cfg.task is None:
        raise ValueError("Config: task must be provided")
    elif cfg.task == "propagate":
        prior = get_prior(cfg.propagate_prior.class_name, **cfg.propagate_prior.params)
        datamodule = TimeDelayedPairDataModule(
            **cfg.data, prior=prior, seed=cfg.seed, neighbor_list_both_ends=False
        )
        model = (
            FlowModule(cfg)
            if cfg.model.pretrained_ckpt is None
            else FlowModule.load_from_checkpoint(cfg.model.pretrained_ckpt)
        )
    elif cfg.task == "correct":
        noise = get_prior(cfg.correct_noise.class_name, **cfg.correct_noise.params)
        prior = get_prior(cfg.correct_prior.class_name, **cfg.correct_prior.params)
        datamodule = NoisedFrameDataModule(
            **cfg.data, noise=noise, prior=prior, seed=cfg.seed
        )
        model = (
            FlowModule(cfg)
            if cfg.model.pretrained_ckpt is None
            else FlowModule.load_from_checkpoint(cfg.model.pretrained_ckpt)
        )
    else:
        raise ValueError(f"Task {cfg.task} not found")

    # Setup loggers
    loggers = []
    if cfg.logger.csv:
        loggers.append(CSVLogger(save_dir="."))
    if cfg.logger.wandb and not cfg.logger.debug:
        loggers.append(WandbLogger(**cfg.logger.wandb))

    # Setup callbacks and trainer
    callbacks = []
    for callback_info in cfg.trainer.callbacks:
        try:
            if callback_info.class_name == "EMAModelCheckpoint":
                callback_class = EMAModelCheckpoint
            elif callback_info.class_name == "EMA":
                callback_class = EMA
            else:
                callback_class = getattr(L.pytorch.callbacks, callback_info.class_name)
        except AttributeError:
            raise ValueError(f"Callback {callback_info.class_name} not found")
        callback = callback_class(**callback_info.params)
        callbacks.append(callback)
    if cfg.logger.wandb.offline:
        try:
            from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback
        except ImportError:
            raise ImportError(
                "wandb-osh is required for offline wandb logging. Please install it."
            )
        callbacks.append(TriggerWandbSyncLightningCallback())

    cfg.trainer.__delattr__("callbacks")
    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=loggers)

    # Train model
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
