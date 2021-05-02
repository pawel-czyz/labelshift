import math
from typing import Optional

import numpy as np


class StrictClassifier:

    def __init__(self, confusion_matrix: np.ndarray) -> None:
        """

        Args:
            confusion matrix: the confusion matrix of the requested classifier. All entries should sum up to 1.
        
        """
        self.confusion_matrix = np.asarray(confusion_matrix, dtype=float)

    def generate(n_classes: int) -> np.ndarray:
        pass
