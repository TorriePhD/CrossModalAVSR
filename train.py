import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from avg_ckpts import ensemble
from datamodule.data_module import DataModule
from pathlib import Path


@hydra.main(version_base="1.3", config_path="configsMulti", config_name="configMulti")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    if cfg.resume_from_checkpoint is not None:
        if not Path(cfg.resume_from_checkpoint).is_file():
            cfg.resume_from_checkpoint = None
        else:
            cfg.pretrained_model_path = None

    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Set modules and trainer
    # if cfg.data.modality in ["audio", "visual"]:
    from lightning import ModelModule
    # elif cfg.data.modality == "audiovisual":
    #     from lightning_av import ModelModule
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    #get name of exp_dir
    exp_dir_name = os.path.basename(os.path.normpath(cfg.exp_dir))
    logger = TensorBoardLogger(save_dir=str(Path(cfg.exp_dir).parent/"lightning_logs"), name=exp_dir_name, log_graph=True,version=cfg.exp_name)
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    with torch.autograd.detect_anomaly():
        trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=cfg.resume_from_checkpoint)
    ensemble(cfg)
    # #load ensemble
    state_dict = torch.load(os.path.join(cfg.exp_dir, cfg.exp_name, "model_avg_10.pth"), map_location=lambda storage, loc: storage)
    modelmodule.model.load_state_dict(state_dict)
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
