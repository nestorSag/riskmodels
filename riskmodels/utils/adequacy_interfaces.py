"""
This module contains a few base interfaces for surplus capacity models defined in `riskmodels.adequacy.capacity_models`. 

"""
from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
import copy
import typing as t
from zlib import adler32
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import bisect

from pydantic import BaseModel as BasePydanticModel, validator


class BaseCapacityModel(ABC):

    """Main interface for capacity models outlining the basic list of methods"""

    @abstractmethod
    def cdf(self):
        pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def lole(self):
        pass

    @abstractmethod
    def eeu(self):
        pass


class BaseBivariateMonteCarlo(BasePydanticModel, BaseCapacityModel):

    """Base class for calculating time-collapsed risk indices for bivariate capacity surplus distributions using Monte Carlo. Implements calculations based on an assumed capacity trace, but crucially leaves unimplemented the method to compute this surplus trace; subclasses inheriting from this one can implement different ways to calculate this, such as simulation from bivariate distribution objects or loading traces from a file in the case of sequential Monte Carlo models. Only veto policies are implemented.

    Args:
        season_length (int): Length of individual peak seasons
    """

    def __repr__(self):
        return f"Surplus MonteCarlo model of type {self.__class__.__name__}"

    season_length: int

    def itc_flow(self, sample: np.ndarray, itc_cap: int = 1000) -> np.ndarray:
        """Returns the interconnector flow from a sample of bivariate pre interconnection surplus values. The flow is expressed as flow to area 1 being positive and flow to area 2 being negative.

        Args:
            sample (np.ndarray): Bivariate surplus sample
            itc_cap (int, optional): Interconnection capacity

        Returns:
            np.ndarray
        """
        flow = np.zeros(
            (
                len(
                    sample,
                )
            ),
            dtype=np.float32,
        )

        if itc_cap == 0:
            return flow

        flow_from_area_1_idx = np.logical_and(sample[:, 0] > 0, sample[:, 1] < 0)
        flow_to_area_1_idx = np.logical_and(sample[:, 0] < 0, sample[:, 1] > 0)

        # flows are bounded by interconnection capacity, shortfall size and spare available capacity in each area.
        flow[flow_from_area_1_idx] = -np.minimum(
            itc_cap,
            np.minimum(
                sample[:, 0][flow_from_area_1_idx], -sample[:, 1][flow_from_area_1_idx]
            ),
        )
        flow[flow_to_area_1_idx] = np.minimum(
            itc_cap,
            np.minimum(
                -sample[:, 0][flow_to_area_1_idx], sample[:, 1][flow_to_area_1_idx]
            ),
        )

        return flow

    def simulate(self, itc_cap: int = 1000):
        """Simulate from post-interconnection surplus distribution

        Args:
            itc_cap (int, optional): Interconnection capacity

        """
        pre_itc_sample = self.get_pre_itc_sample()
        flow = self.itc_flow(pre_itc_sample, itc_cap)
        # add flow to pre itc sample
        pre_itc_sample[:, 0] += flow
        pre_itc_sample[:, 1] -= flow
        return pre_itc_sample

    def cdf(self, x: np.ndarray, itc_cap: int = 1000):
        """Estimate the CDF of bivariate post-interconnection surplus evaluated at x

        Args:
            x (np.ndarray): point to be evaluated
            itc_cap (int, optional): interconnection capacity

        """
        samples = self.simulate(itc_cap)
        u = samples <= x.reshape((1, 2))  # componentwise comparison
        v = (
            u.dot(np.ones((2, 1))) >= 2
        )  # equals 1 if and only if both components fulfill the above condition
        return np.mean(v)  # return empirical CDF estimate

    def lole(self, itc_cap: int = 1000, area: int = 0):
        """Calculates loss of load expectation for one of the areas in the system
        Args:
            itc_cap (int, optional): Interconnection capacity
            area (int, optional): Area index (0 or 1); if area=-1, systemwide lole is returned.

        """
        # take as loss of load when shortfalls are at least 0.1MW in size; this induces a negligible amount of bias but solves numerical issues when comparing post-itc surpluses to 0 to flag shortfalls.
        x = np.array([-1e-1, -1e-1], dtype=np.float32)
        if area in [0, 1]:
            x[1 - area] = np.Inf
            return self.season_length * self.cdf(x, itc_cap)
        elif area == -1:
            return self.season_length * (
                self.cdf(np.array([np.Inf, 0]), itc_cap)
                + self.cdf(np.array([0, np.Inf]), itc_cap)
                - self.cdf(np.array([0, 0]), itc_cap)
            )
        else:
            raise ValueError("area must be in [-1,0,1]")

    def eeu(self, itc_cap: int = 1000, area: int = 0):
        """Calculates expected energy unserved for one of the areas in the system

        Args:
            itc_cap (int, optional): Interconnection capacity
            area (int, optional): Area index (0 or 1).

        """
        samples = self.simulate(itc_cap)
        if area in [0, 1]:
            return -self.season_length * np.mean(np.minimum(samples[:, area], 0))
        elif area == -1:
            return -self.season_length * (
                np.mean(np.minimum(samples[:, 0], 0))
                + np.mean(np.minimum(samples[:, 1], 0))
            )
        else:
            raise ValueError("area must be in [-1,0,1]")

    @abstractmethod
    def get_pre_itc_sample(self) -> np.ndarray:
        """Returns a pre-interconnection surplus sample

        Returns:
            np.ndarray: Sample
        """
        pass
