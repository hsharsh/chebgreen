from .greenlearning.utils import DataProcessor
from .greenlearning.model import *
from abc import ABC

class ChebGreen(ABC):
    def __init__(self, path, Theta, generateData = True, ) -> None:
        super().__init__()
