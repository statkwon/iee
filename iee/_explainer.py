import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from scipy.fft import rfft
from sklearn.base import is_regressor, is_classifier, BaseEstimator
from sklearn.inspection._partial_dependence import _partial_dependence_brute
from sklearn.utils.validation import check_is_fitted
from torch import Tensor
from torch.nn import Module


class Explainer:
    def __init__(
        self,
        model: BaseEstimator | Module,
        X: Optional[ndarray | Tensor] = None,
        mode: Optional[str] = None,
        grid_resolution: int = 256,
    ):
        if issubclass(type(model), BaseEstimator):
            check_is_fitted(model)
            if is_regressor(model):
                self._is_regressor = True
            elif is_classifier(model):
                self._is_regressor = False
            else:
                raise ValueError("'model' must be a sklearn regressor or classifer.")
            self._is_sklearn = True
        elif issubclass(type(model), Module):
            if mode is None:
                raise ValueError("'mode' cannot be None for torch Module.")
            elif mode == "regressor":
                self._is_regressor = True
            elif mode == "classifier":
                self._is_regressor = False
            else:
                raise ValueError(
                    "'mode' must be either 'regression' or 'classification'"
                )
            self._is_sklearn = False
        else:
            raise ValueError(
                "'model' must be a subclass of sklearn 'BaseEstimator' or torch 'Module'."
            )

        if grid_resolution < 1:
            raise ValueError("'grid_resolution' must be strictly greater than 1.")

        if X is not None:
            self._X = X
        else:
            self._X = None

        self._model = model
        self.grid_resolution = grid_resolution

    def _grid_from_Xj(self, X: ndarray, j: int) -> ndarray:
        X_grid = self._X if self._X is not None else X
        try:
            uniques = np.unique(X_grid[:, j])
        except TypeError as exc:
            raise ValueError(
                f"The column #{j} contains mixed data types. Finding unique "
                "categories fail due to sorting. It usually means that the column "
                "contains `np.nan` values together with `str` categories. Such use "
                "case is not yet supported."
            ) from exc
        if len(uniques) < self.grid_resolution:
            warnings.warn(
                f"The feature #{j} has less unique values({len(uniques)}) than grid resolution({self.grid_resolution})."
            )
        grid = np.linspace(
            min(uniques), max(uniques), num=self.grid_resolution, endpoint=True
        )

        return grid.reshape(-1, 1)

    def _independent_conditional_expectation_sklearn(
        self, X: ndarray, j: int
    ) -> (ndarray, ndarray):
        grid = self._grid_from_Xj(X, j)
        _, predictions = _partial_dependence_brute(self._model, grid, [j], X, "auto")
        if not self._is_regressor and predictions.ndim == 2:
            predictions = np.stack((1 - predictions, predictions))
        return grid, predictions

    def _independent_conditional_expectation_torch(
        self, X: Tensor, j: int
    ) -> (ndarray, ndarray):
        n, _ = X.shape
        grid = self._grid_from_Xj(X.numpy(), j)
        X_repeated = X.repeat_interleave(self.grid_resolution, 0)
        X_repeated[:, j] = torch.tensor(
            np.tile(grid.squeeze(), n), dtype=X_repeated.dtype
        )
        predictions = (
            self._model(X_repeated)
            .T.reshape(-1, n, self.grid_resolution)
            .squeeze(0)
            .detach()
            .numpy()
        )
        return grid, predictions

    def _independent_conditional_expectation(
        self, X: ndarray | Tensor, j: int
    ) -> (ndarray, ndarray):
        if self._is_sklearn:
            return self._independent_conditional_expectation_sklearn(X, j)
        else:
            return self._independent_conditional_expectation_torch(X, j)

    def __call__(self, X: ndarray | Tensor, scaled: bool = True) -> ndarray:
        n, p = X.shape

        # Get ice values for each feature
        ice = []
        self._grid = []
        for j in range(p):
            grid, predictions = self._independent_conditional_expectation(X, j)
            self._grid.append(grid)
            ice.append(predictions)
        self._ice = np.array(
            ice
        )  # (p, n, grid_resolution) | (p, n_classes, n, grid_resolution)

        length = self.grid_resolution // 2 + 1
        hpf = np.arange(1, length + 1) ** 2

        self._iee_values = None
        if self._is_regressor:
            self._ice = self._ice.transpose(1, 0, 2)  # (n, p, grid_resolution)
            self._iee_values = np.zeros([n, p])

            for i in range(n):
                for j in range(p):
                    ft = rfft(self._ice[i, j]) / (self.grid_resolution / 2 + 1)
                    self._iee_values[i, j] = sum((abs(ft * hpf) ** 2))
        elif not self._is_regressor:
            self._ice = self._ice.transpose(
                2, 0, 1, 3
            )  # (n, p, n_classes, grid_resolution)
            n_classes = self._ice.shape[2]
            self._iee_values = np.zeros([n, p, n_classes])

            for i in range(n):
                for j in range(p):
                    for k in range(n_classes):
                        ft = rfft(self._ice[i, j, k]) / (self.grid_resolution / 2 + 1)
                        self._iee_values[i, j, k] = sum((abs(ft * hpf) ** 2))

        if scaled:
            if self._iee_values.max() != 0:
                self._iee_values /= self._iee_values.max()
            else:
                warnings.warn(f"Maximum is 0.")

        return self._iee_values

    def visualize(
        self, i: int, j: int, *, k: Optional[int] = None, ax: plt.axis, **kwargs
    ):
        if k is None:
            ax.plot(
                self._grid[j],
                self._ice[i, j],
                label=round(self._iee_values[i, j], 3),
                **kwargs,
            )
        else:
            ax.plot(
                self._grid[j],
                self._ice[i, j, k],
                label=round(self._iee_values[i, j, k], 3),
                **kwargs,
            )
