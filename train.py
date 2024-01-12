import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from avg_ckpts import ensemble
from datamodule.data_module import DataModule
from pathlib import Path



@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    if cfg.trainer.resume_from_checkpoint is not None:
        if not Path(cfg.trainer.resume_from_checkpoint).is_file():
            cfg.trainer.resume_from_checkpoint = None
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
    logger = TensorBoardLogger(save_dir="/home/st392/fsl_groups/grp_lip/compute/results/lightning_logs", name=exp_dir_name, log_graph=True,version=cfg.exp_name)
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)
    ensemble(cfg)


if __name__ == "__main__":
    main()
