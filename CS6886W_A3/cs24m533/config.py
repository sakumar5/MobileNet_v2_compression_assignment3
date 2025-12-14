"""----------------------------------------------------------------
Modules:
    dataclasses : Defines structured configuration classes for training settings.
    typing      : Provides type hints for configuration parameters.
    argparse    : Parses command-line arguments into configuration fields.
----------------------------------------------------------------"""
import dataclasses
from dataclasses import dataclass
from typing import Optional

"""---------------------------------------------
* class name :
*       TrainConfig
*
* purpose:
*       Stores and manages all training,
*       quantization, and logging parameters.
*
* Input parameters:
*       None : values are initialized as attributes
*
* return:
*       TrainConfig object
---------------------------------------------"""
class TrainConfig:
    # Data
    data_dir: str = "./data"
    num_workers: int = 4

    # Model
    width_mult: float = 1.0
    dropout: float = 0.2
    num_classes: int = 10

    # Training
    batch_size: int = 128
    epochs: int = 200
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_schedule: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.0
    seed: int = 42

    # Compression
    weight_bits: int = 8
    activation_bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    quantize_activations_during_train: bool = False
    quantize_weights_during_forward: bool = False

    # Logging
    use_wandb: bool = False
    wandb_project: str = "cs6886_assignment3"
    wandb_run_name: Optional[str] = None
    log_interval: int = 100
    out_dir: str = "./checkpoints"
    resume: str = None

    """---------------------------------------------
    * def name :
    *       as_dict
    *
    * purpose:
    *       Converts configuration values into a dictionary.
    *
    * Input parameters:
    *       self : configuration object
    *
    * return:
    *       Dictionary of configuration parameters
    ---------------------------------------------"""
    def as_dict(self):
        return dataclasses.asdict(self)
