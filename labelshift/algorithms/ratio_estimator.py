"""Implements the ratio estimator described in:

Afonso Fernandes Vaz, Rafael Izbicki, Rafael Bassi Stern;
Quantification Under Prior Probability Shift: the Ratio Estimator and its Extensions
Journal of Machine Learning Research, 20(79):1−33, 2019.

More precisely, we implemented the estimator from Remark 5 on page 5
using a bit different notation.

Consider the true label ``Y \\in \\{1, ..., L\\}`` and a given function

  ``f: X --> R^K``.

In the paper there is a requirement ``K=L``, although we will not strictly enforce it
in this implementation.

Then, a projection of ``f`` onto the first ``K-1`` components is constructed:

  ``g: X --> R^{K-1}``

and the following vector is constructed

 ``g_hat[k] = E_unlabeled[ g(X)[k] ] \\in R^{K-1}``

as well as the following matrix

  ``G_hat[k, l] = E_labeled[ g(X)[k]  | Y = l] \\in R^{(K-1) x L}``

If ``y \\in R^L `` is the prevalence vector,
it is supposed to fulfil the following equations:

  ``
     g_hat[k] = sum_l( G_hat[k, l] y[l] )
     sum_l( y[l] ) = 1
  ``

To make sure it is a probability vector, it can be projected onto the (L-1)-simplex.

According to our conventions, we will use the transpose of this matrix:

  ``H_hat[l, k] = G_hat[l, k] = E_labeled[ g(X)[k]  | Y = l] \\in R^{L x (K-1)}.``

"""
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from labelshift.algorithms.bbse import solve_system
from labelshift.adjustments import project_onto_probability_simplex


def prevalence_from_vector_and_matrix(
    vector: np.ndarray,
    matrix: np.ndarray,
    restricted: bool = True,
    enforce_square: bool = True,
    rcond: float = 1e-4,
) -> np.ndarray:
    """Calculates the prevalence vector from the ``g_hat`` vector
    and the ``H_hat = transpose(G_hat)`` matrix.

    Args:
        vector: the ``\\hat g`` vector, shape (K-1,),
          where K is the number of values C = g(X) can take
        matrix: shape (L, K-1)
        enforce_square: whether a square solver should be used.
          In that case one needs K = L.
        rcond: parameter of the solver
        restricted: whether to use restricted or unrestricted estimator

    Returns:
        estimated prevalence vector, shape (L,)
    """
    K = vector.shape[0] + 1
    L = matrix.shape[0]

    if matrix.shape != (L, K - 1):
        raise ValueError(
            f"The matrix has shape {matrix.shape} rather than {(L, K - 1)}."
        )

    new_vector = np.ones(K, dtype=float)
    new_vector[: K - 1] = vector

    new_matrix = np.ones((L, K), dtype=float)
    new_matrix[:, : (K - 1)] = matrix

    unrestricted = solve_system(
        matrix=new_matrix.T,
        vector=new_vector,
        square_solver=enforce_square,
        rcond=rcond,
    )
    if restricted:
        return project_onto_probability_simplex(unrestricted)
    else:
        return unrestricted


def calculate_vector_and_matrix_from_predictions(
    unlabeled_predictions: ArrayLike,
    labeled_predictions: ArrayLike,
) -> None:
    """This method has not been implemented yet."""
    raise NotImplementedError


def calculate_vector_and_matrix_from_summary_statistics(
    n_c_unlabeled: ArrayLike,
    n_y_and_c_labeled: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        n_c_unlabeled: vector of shape (K,)
        n_y_and_c_labeled: matrix of shape (L, K)

    Returns:
        vector_c: vector of shape (K-1,)
        matrix_y_c: matrix of shape (L, K-1)
    """
    n_c = np.asarray(n_c_unlabeled, dtype=float)
    n_y_c = np.asarray(n_y_and_c_labeled, dtype=float)

    if n_y_c.shape[1] != n_c.shape[0]:
        raise ValueError(
            f"Shape mismatch: {n_y_c.shape} is not compatible with {n_c.shape}."
        )

    p_c = n_c / n_c.sum()
    p_y_c = n_y_c / n_y_c.sum(axis=1)[:, None]

    return p_c[:-1], p_y_c[:, :-1]


def prevalence_from_summary_statistics(
    n_c_unlabeled: ArrayLike,
    n_y_and_c_labeled: ArrayLike,
    restricted: bool = True,
    enforce_square: bool = True,
    rcond: float = 1e-4,
) -> np.ndarray:
    """Estimates the prevalence vector from the sufficient statistic.

    Args:
        n_c_unlabeled: vector of shape (K,)
        n_y_and_c_labeled: matrix of shape (L, K)
        enforce_square: whether a square solver should be used.
          In that case one needs K = L.
        rcond: parameter of the solver
        restricted: whether to use restricted or unrestricted estimator

    Returns:
        estimated prevalence vector, shape (L,)
    """
    vector, matrix = calculate_vector_and_matrix_from_summary_statistics(
        n_c_unlabeled=n_c_unlabeled, n_y_and_c_labeled=n_y_and_c_labeled
    )
    return prevalence_from_vector_and_matrix(
        vector=vector,
        matrix=matrix,
        restricted=restricted,
        enforce_square=enforce_square,
        rcond=rcond,
    )
