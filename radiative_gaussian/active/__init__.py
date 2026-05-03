from .rand_selector import RandSelector
from .ensemble import EnsembleSelector
methods_dict = {
    "rand": RandSelector,
    "ensemble": EnsembleSelector
}