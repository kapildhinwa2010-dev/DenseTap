from .core import DenseCrossLayerTF, softmax
from .data import make_toroidal_task, torus_lift
from .train import Trainer, paired_report
from .budget import nparams, solve, PRESET_5M, PRESET_20K
__all__ = ["DenseCrossLayerTF", "softmax", "make_toroidal_task", "torus_lift",
           "Trainer", "paired_report", "nparams", "solve", "PRESET_5M", "PRESET_20K"]
__version__ = "0.2.0"
