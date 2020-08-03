from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from torch import Tensor


class Metric(ABC):

    @abstractmethod
    def process(self, frame: Union[np.ndarray, Tensor]) -> float:
        pass
