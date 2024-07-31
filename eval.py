import logging
import os

import hydra
import torch

from pytorch_lightning import Trainer
from lightning import ModelModule
from datamodule.data_module import DataModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    # Set modules and trainer
    from lightning import ModelModule
    print(f"evaluating model: {cfg.pretrained_model_path}")
    print(f"signal to noise ratio: {cfg.decode.snr_target}")
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(num_nodes=cfg.trainer.num_nodes, gpus=cfg.trainer.gpus)
    # Training and testing
    # state_dict = torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)["state_dict"]
    # state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    # state_dict = torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
    # modelmodule.model.load_state_dict(state_dict)
    #save the results of the model in a file
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
