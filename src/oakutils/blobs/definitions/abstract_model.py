from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import torch


class ModelInput(Enum):
    COLOR = 0
    GRAY = 1
    DEPTH = 2

class AbstractModel(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def input_type(self) -> ModelInput:
        """
        The type of input this model takes
        """

    @classmethod
    @abstractmethod
    def input_names(self) -> List[str]:
        """
        The names of the input tensors
        """
        pass

    @classmethod
    @abstractmethod
    def output_names(self) -> List[str]:
        """
        The names of the output tensors
        """
        pass
