"""
This module contains utilities for the execution of multi-core map-reduce operations when using sequential capacity models from `riskmodels.adequacy.capacity_models`. Classes defined here are the workers of map-reduce computations and act as wrappers for `numpy` arrays that have been persisted as files and are read in parallel at execution time. These classes are not meant to be instantiated directly, but can be accessed through custom mappers and reducers passed to sequential capacity model instances.

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

from pydantic import BaseModel as BasePydanticModel

from riskmodels.utils.adequacy_interfaces import (
    BaseCapacityModel,
    BaseBivariateMonteCarlo,
)


class PersistedTraces(BasePydanticModel):

    """Wrapper class for persisted files of simulated traces. This class is not meant to be instantiated directly by the end user."""

    traces: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_file(cls, trace_filepath: str):
        """Loads a pickled numpy array that contains conventional generation traces

        Args:
            trace_filepath (str): Path to file
        """
        return cls(traces=np.load(trace_filepath, allow_pickle=True))

    def __add__(self, other: float):
        return type(self)(traces=self.samples + other)

    def __mul__(self, other: float):
        return type(self)(traces=self.samples * other)


class UnivariateTraces(BaseCapacityModel, BasePydanticModel):

    """Wrapper class for the workers of univariate map-reduce computations; it uses a file containing sequences of conventional generation traces to perform custom computations. Instances of this class are not meant to be instantiated directly by the end user.

    Args:
        gen_filepath (str): folder with conventional generation data
        demand (np.ndarray): demand data
        renewables (np.ndarray): renewables data
        season_length (int): number of timesteps per peak season

    """

    gen_filepath: str
    demand: np.ndarray
    renewables: np.ndarray
    season_length: int

    class Config:
        arbitrary_types_allowed = True

    # @validator("season_length", allow_reuse=True)
    # def season_length_validator(cls, season_length):
    #   if season_length is None:
    #     return len(self.demand)

    @property
    def surplus_trace(self):
        # this return a 2-dimensional array where each row is a trace sample, and each column is a timestep within the trace. A trace may contain multiple concatenated peak seasons
        return PersistedTraces.from_file(self.gen_filepath).traces - (
            self.demand - self.renewables
        )

    @property
    # number of traces in file
    def n_traces(self):
        return len(self.surplus_trace)

    def cdf(self, x: float) -> t.Tuple[float, int]:
        """Evaluates the surplus distribution's CDF. Also returns the number of seasons used to calculate it.

        Args:
            x (float): Description

        Returns:
            t.Tuple[float, int]: A tuple with the estimated value and the number of seasons used to calculate it.
        """
        trace = self.surplus_trace
        return np.mean(trace < x)

    def simulate(self) -> np.ndarray:
        """Returns a simulated trace for surplus values

        Returns:
            np.ndarray: simulated surplus values
        """
        return self.surplus_trace

    def simulate_lold(self) -> np.ndarray:
        """Returns a simulated trace for energy unserved

        Returns:
            np.ndarray: Simulated energy unserved
        """
        trace = self.surplus_trace
        n_traces, trace_length = trace.shape
        if trace_length % self.season_length != 0:
            raise ValueError("Trace length is not a multiple of season length.")
        target_shape = (
            n_traces * (trace_length // self.season_length),
            self.season_length,
        )  # reshape as (# peak seasons x peak season length)
        return np.sum(
            (np.maximum(0.0, -self.surplus_trace) > 1e-1).reshape(target_shape), axis=1
        )  # 1e-1 to avoid problems with numerical rounding errors

    def simulate_eu(self) -> np.ndarray:
        """Returns a simulated trace for energy unserved

        Returns:
            np.ndarray: Simulated energy unserved
        """
        trace = self.surplus_trace
        n_traces, trace_length = trace.shape
        if trace_length % self.season_length != 0:
            raise ValueError("Trace length is not a multiple of season length.")
        target_shape = (
            n_traces * (trace_length // self.season_length),
            self.season_length,
        )  # reshape as (# peak seasons x within-peak-season timestamp)
        return np.sum(np.maximum(0.0, -trace).reshape(target_shape), axis=1)

    def lole(self) -> float:
        """Evaluates the distribution's season-wise LOLE. Also returns the number of seasons used to calculate it.

        Returns:
            t.Tuple[float, int]: A tuple with the estimated value and the number of seasons used to calculate it.
        """

        # cdf_value, n = self.cdf(0.0)
        # return self.season_length * cdf_value, n
        trace = self.surplus_trace

        n_traces, trace_length = trace.shape
        if trace_length % self.season_length != 0:
            raise ValueError("Trace length is not a multiple of season length.")
        seasons_per_trace = int(trace_length / self.season_length)
        n_total_seasons = n_traces * seasons_per_trace

        return np.sum(trace < 0) / n_total_seasons

    def eeu(self) -> float:
        """Evaluates the distribution's season-wise expected energy unserved. Also returns the number of seasons used to calculate it.


        Returns:
            t.Tuple[float, int]: A tuple with the estimate value and the number of seasons used to calculate it.
        """

        trace = self.surplus_trace

        n_traces, trace_length = trace.shape
        if trace_length % self.season_length != 0:
            raise ValueError("Trace length is not a multiple of season length.")
        seasons_per_trace = int(trace_length / self.season_length)
        n_total_seasons = n_traces * seasons_per_trace

        return np.sum(np.maximum(0.0, -trace)) / n_total_seasons

    def get_surplus_df(self, shortfalls_only: bool = True) -> pd.DataFrame:
        """Returns a data frame with time occurrence information of observed surplus values and shortfalls.

        Args:
            shortfalls_only (bool, optional): If True, only shortfall rows are returned

        Returns:
            pd.DataFrame: A data frame with the surplus values, a 'season_time' column with the within-season time of occurrence (0,1,...,season_length-1), a 'file_id' column that indicates which file was used to compute the value, and a 'season' column to indicate which season the value was observed in.

        """
        pd.options.mode.chained_assignment = None  # supress false positive warnings

        trace = self.surplus_trace
        df = pd.DataFrame({"surplus": trace.reshape(-1)})
        df["time"] = np.arange(len(df))
        # filter by shortfall
        if shortfalls_only:
            df = df.query("surplus < 0")
        # add season features
        raw_time = np.array(df["time"])
        df["season_time"] = raw_time % self.season_length
        df["season"] = (raw_time / self.season_length).astype(np.int32)
        df = df.drop(columns=["time"])
        df["file_id"] = Path(self.gen_filepath).name

        pd.options.mode.chained_assignment = "warn"  # reset default

        return df


class BivariateTraces(BaseBivariateMonteCarlo):

    """Wrapper class for the workers of bivariate map-reduce computations; it uses a file containing sequences of conventional generation traces to perform custom computations. Instances of this class are not meant to be instantiated directly by the end user. This class implements both veto and share policies.

    Args:
        univariate_traces (t.List[UnivariateTraces]): Univariate traces
        policy (str): Either 'veto' or 'share'
    """

    univariate_traces: t.List[UnivariateTraces]
    policy: str

    class Config:
        arbitrary_types_allowed = True

    @property
    def surplus_trace(self):
        """This returns the traces as a 3-dimensional array where the axes correspond to area, simulated trace and within-trace time respectively. Each trace may contain multiple concatenated peak seasons"""
        return np.array([t.surplus_trace for t in self.univariate_traces])

    @property
    # number of traces in each file
    def n_traces(self):
        return self.univariate_traces[0].n_traces

    def get_pre_itc_sample(self) -> np.ndarray:
        """Returns a pre-interconnection surplus sample as a two-dimensional array where realisations of different peak seasons have been concatenated for each area (each row is a single time step and each column is an area).

        Returns:
            np.ndarray: Sample
        """
        return np.stack(
            [t.surplus_trace.reshape(-1) for t in self.univariate_traces], axis=1
        )

    def itc_flow(self, sample: np.ndarray, itc_cap: int = 1000) -> np.ndarray:
        """Returns the interconnector flow from a sample of bivariate pre interconnection surplus values. The flow is expressed as flow to area 1 being positive and flow to area 2 being negative.

        Args:
            sample (np.ndarray): Bivariate surplus sample
            itc_cap (int, optional): Interconnection capacity

        Returns:
            np.ndarray

        """
        if self.policy == "veto" or itc_cap == 0:
            return super().itc_flow(sample, itc_cap)
        elif self.policy == "share":
            flow = np.zeros(
                (
                    len(
                        sample,
                    )
                ),
                dtype=np.float32,
            )
            # split individual surplus traces
            s1, s2 = sample[:, 0], sample[:, 1]
            # market-driven shortfall-sharing conditions from a share policy only really kick in under specific conditions; in all other situations, the policy is identical to veto.
            # briefly, this is mostly but not entirely because of interconnector constraints
            share_cond = np.logical_and(s1 + s2 < 0, s1 < itc_cap, s2 < itc_cap)
            # market-driven flows are determined by demand in addition to surpluses; tile demand vector to perform flow calculations
            d1, d2 = (
                self.univariate_traces[0].demand,
                self.univariate_traces[1].demand,
            )  # demand arrays
            if len(d1) != len(d2):
                raise ValueError("Traces of demand are not the same length.")

            k = len(sample) / len(d1)  # tiling factor
            if k - int(k) != 0:
                raise ValueError(
                    "Length of surplus samples is not a multiple of demand array length."
                )
            k = int(k)
            # tile demand ratio directly (demand ratio is used in flow equation below)
            r = np.tile(d1 / (d1 + d2), k)
            # compute share flow when applicable
            flow[share_cond] = np.minimum(
                itc_cap,
                np.maximum(
                    -itc_cap,
                    r[share_cond] * s2[share_cond]
                    - (1 - r[share_cond]) * s1[share_cond],
                ),
            )
            # compute veto flow for all other entries
            flow[np.logical_not(share_cond)] = super().itc_flow(
                sample[np.logical_not(share_cond)], itc_cap
            )
            return flow
        else:
            raise ValueError("policy must be either 'veto' or 'share'")

    def simulate_lold(self, itc_cap: int = 1000) -> np.ndarray:
        """Returns a simulated trace for loss of load duration

        Returns:
            np.ndarray: Simulated loss of load duration
        """
        lold_vectors = (
            np.maximum(0.0, -self.simulate(itc_cap)) > 1e-1
        ).T  # avoid numerical rounding errors with offset 1e-1
        n = len(lold_vectors[0])
        if n % self.season_length != 0:
            raise ValueError(
                "Simulated series length is not a multiple of season length."
            )
        return np.stack(
            [
                v.reshape((n // self.season_length, self.season_length)).sum(axis=1)
                for v in lold_vectors
            ],
            axis=1,
        )

    def simulate_eu(self, itc_cap: int = 1000) -> np.ndarray:
        """Returns a simulated trace for energy unserved

        Returns:
            np.ndarray: Simulated energy unserved
        """
        eu_vectors = np.maximum(0.0, -self.simulate(itc_cap)).T
        n = len(eu_vectors[0])
        if n % self.season_length != 0:
            raise ValueError(
                "Simulated series length is not a multiple of season length."
            )
        return np.stack(
            [
                v.reshape((n // self.season_length, self.season_length)).sum(axis=1)
                for v in eu_vectors
            ],
            axis=1,
        )

    def get_surplus_df(
        self, shortfalls_only: bool = True, itc_cap: int = 1000
    ) -> pd.DataFrame:
        """Returns a data frame with time occurrence information of observed post-interconnection surplus values and shortfalls.

        Args:
            shortfalls_only (bool, optional): Whether to return only rows corresponding to shortfalls.
            itc_cap (int, optional): Interconnector policy

        Returns:
            pd.DataFrame: A data frame with the surplus values, a 'season_time' column with the within-season time of occurrence (0,1,...,season_length-1), a 'file_id' column that indicates which file was used to compute the value, and a 'season' column to indicate which season the value was observed in.

        """
        trace = self.simulate(itc_cap)
        df = pd.DataFrame(trace, columns=["surplus1", "surplus2"])
        df["time"] = np.arange(len(df))
        df["file_id"] = Path(
            self.univariate_traces[0].gen_filepath
        ).name  # file name is identical for both areas
        # filter by shortfall
        if shortfalls_only:
            df = df.query("surplus1 < 0 or surplus2 < 0")
        # add season features
        df["season_time"] = df["time"] % self.season_length
        df["season"] = (df["time"] / self.season_length).astype(np.int32)
        df = df.drop(columns=["time"])
        return df
