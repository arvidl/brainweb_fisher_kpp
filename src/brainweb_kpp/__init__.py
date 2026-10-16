"""Educational 3-D Fisher–KPP solver. Length: mm; time: days."""

from .model import Parameters, simulate
from .data import load_brainweb

__all__ = ["Parameters", "simulate", "load_brainweb"]
