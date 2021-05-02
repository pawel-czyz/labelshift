import numpy as np

def recalibrate(predictions: "arraylike", *, training: "arraylike", test: "arraylike") -> np.ndarray:
    """Recalibrate the probabilities predicted by a classifier under the prior probability shift assumption.

    Args:
        predictions: array with classifier predictions, shape (n_samples, n_classes).
        training_prevalences: the ith component is the probability of observing class i in the training data set. Shape (n_classes,).
        test_prevalences: the ith component is the probability of observing class i in the test data set. Shape (n_classes,).
    
    Returns:
        recalibrated predictions. Shape (n_samples, n_classes).
    
    Note:
        If the classifier has been biased towards some classes, this bias will be increased.
    """
    predictions = np.array(predictions, dtype=float)
    training_prevalences = np.array(training, dtype=float)
    test_prevalences = np.array(test, dtype=float)

    assert predictions.shape[1] == len(training_prevalences) == len(test_prevalences), "Shapes are not compatible."

    


    pass
