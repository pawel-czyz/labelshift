import math
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


class StrictClassifier:

    def __init__(self, confusion_matrix: ArrayLike) -> None:
        """

        Args:
            confusion matrix: the confusion matrix of the requested classifier. All entries should sum up to 1.
        
        """
        self.confusion_matrix = np.asarray(confusion_matrix, dtype=float)

    def generate(self, n_classes: int) -> np.ndarray:
        raise NotImplementedError
