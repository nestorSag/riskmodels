"""
This module implements power system capacity models for adequacy assessment calculations. There are sequential and non-sequential (also called time-collapsed) implementations:

In the case of sequential models, because Monte Carlo estimation is the only way to compute statistical properties of time series models for capacity surpluses, these rely heavily on large-scale simulation and computation. The classes of this module implement multi-core processing following a map-reduce pattern on a large number of simulated traces that are persisted in multiple files. The main classes are `UnivariateSequential` and `BivariateSequential`.

In the case of non-sequential or time-collapsed, only two-area models are implemented, as single-area time collapsed models reduce to fairly simple discrete probability distributions and can be handled using the `univariate` module. The main classes are `BivariateNSMonteCarlo` and `BivariateNSEmpirical` empirical, the `NS` standing for non-sequential. The former takes arbitrary bivariate ACG and net demand distributions and computes time-collapsed metrics using Monte Carlo estimation; however, it only implements a `veto` policy in which capacity can only flow to other areas after domestic demand has been satisfied.

The latter uses a non-sequential ACG model from the `acg_models` module and takes as net demand model the empirical distribution of historic net demand, passed as arrays of historic wind output and demand. This model computes EEU and LOLE metrics exactly by processing the two-dimensional ACG probability distribution, and also implements a `share` policy in which shortfalls can be shared in proportion to demand across areas, up to the interconnector capacity. The actual numerical implementation is written in `C`, to which this class interfaces.
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

import riskmodels.univariate as univar
from riskmodels.bivariate import Independent, BaseDistribution
from riskmodels.adequacy import acg_models

from riskmodels.utils.adequacy_interfaces import (
    BaseBivariateMonteCarlo,
    BaseCapacityModel,
)
from riskmodels.utils.map_reduce import UnivariateTraces, BivariateTraces

from tqdm import tqdm

from c_sequential_models_api import ffi, lib as C_CALL
from c_bivariate_surplus_api import ffi, lib as C_API

pd.options.mode.chained_assignment = None  # default='warn'


class BivariateNSMonteCarlo(BaseBivariateMonteCarlo):

    """Non-sequential bivariate capacity surplus model that takes arbitrary bivariate distributions (inheriting from `riskmodels.bivariate.BaseDistribution`) for ACG and net demand. It calculates time-collapsed risk metrics by simulation and only implements a veto policy between areas, this is, areas will only export spare available capacity. For models that implement a `share` policy, see `BivariateSequential` and `BivariateNSEmpirical`.

    Args:
        gen_distribution (BaseDistribution): available conventional generation distribution
        net_demand (BaseDistribution): net demand distribution
        size (int): Sample size for Monte Carlo estimation

    """

    gen_distribution: BaseDistribution
    net_demand: BaseDistribution
    size: int

    class Config:
        arbitrary_types_allowed = True

    def get_pre_itc_sample(self) -> np.ndarray:
        """Returns a pre-interconnection surplus sample by simulating the passed bivariate distributions for available conventional generation and net demand

        Returns:
            np.ndarray: Sample
        """
        return self.gen_distribution.simulate(self.size) - self.net_demand.simulate(
            self.size
        )





class BivariateNSEmpirical(BaseCapacityModel):

    """Bivariate Non-sequential capacity model that uses a hindcast net demand distribution to compute exact LOLE and EEU risk indices; it implements both `veto` and `share` policies"""

    def __repr__(self):
        return f"Bivariate empirical surplus model with {len(self.demand_data)} observations"

    def __init__(
        self,
        demand_data: np.ndarray,
        renewables_data: np.ndarray,
        gen_distributions: t.List[acg_models.NonSequential],
        season_length: int = None,
    ):
        """
        Args:
            demand_data (np.ndarray): Demand data array with two columns
            renewables_data (np.ndarray): Renewable generation data array with two columns
            gen_distributions (t.List[acg_models.NonSequential]): List of non-sequential ACG objects
            season_length (int, optional): length of peak season. If None, it is set as the length of demand data
        
        """
        gen_distribution = Independent(x=gen_distributions[0], y=gen_distributions[1])
        warnings.warn("Coercing data to integer values.", stacklevel=2)

        self.demand_data = np.ascontiguousarray(demand_data, dtype=np.int32)
        self.renewables_data = np.ascontiguousarray(renewables_data, dtype=np.int32)
        self.net_demand_data = np.ascontiguousarray(
            self.demand_data - self.renewables_data
        )

        if not isinstance(gen_distribution.x, univar.Binned) or not isinstance(
            gen_distribution.y, univar.Binned
        ):
            raise TypeError(
                "Marginal generation distributions must be instances of Binned (i.e. integer support)."
            )

        # save hard copy of relevant arrays from generation
        # this is needed because strange things happened when copying arrays directly from the Independent instance. This somehow solves the issue
        self.convgen1 = {
            "min": gen_distribution.x.min,
            "max": gen_distribution.x.max,
            "cdf_values": np.ascontiguousarray(
                np.copy(gen_distribution.x.cdf_values, order="C")
            ),
            "expectation_vals": np.ascontiguousarray(
                np.cumsum(gen_distribution.x.support * gen_distribution.x.pdf_values)
            ),
        }

        self.convgen2 = {
            "min": gen_distribution.y.min,
            "max": gen_distribution.y.max,
            "cdf_values": np.ascontiguousarray(
                np.copy(gen_distribution.y.cdf_values, order="C")
            ),
            "expectation_vals": np.ascontiguousarray(
                np.cumsum(gen_distribution.y.support * gen_distribution.y.pdf_values)
            ),
        }

        self.gen_distribution = gen_distribution
        self.MARGIN_BOUND = int(np.iinfo(np.int32).max / 2)

        if season_length is None:
            warnings.warn("Using length of demand data as season length.", stacklevel=2)
        self.season_length = (
            len(self.demand_data) if season_length is None else season_length
        )

    def cdf(self, x: np.ndarray, itc_cap: int = 1000, policy: str = "veto"):
        """Evaluates the bivariate post-interconnection time-collapsed capacity surplus distribution's cumulative distribution function

        Args:
            x (np.ndarray): value at which to evaluate the cdf
            itc_cap (int, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, capacity shortfalls are shared according to demand proportions across areas. Shortfalls can extend from one area to another by diverting power.

        """

        # if x is an n x 2 matrix
        if len(x.shape) == 2 and len(x) > 1:
            return np.array([self.cdf(v, itc_cap, policy) for v in x])

        # bound and unbunble component values
        x = np.clip(x, a_min=-self.MARGIN_BOUND, a_max=self.MARGIN_BOUND)
        x1, x2 = x.reshape(-1)

        # convgen1, convgen2 = self.gen_distribution.x, self.gen_distribution.y
        n = len(self.net_demand_data)

        cdf = 0

        for k in range(n):
            net_demand1, net_demand2 = self.net_demand_data[k]
            demand1, demand2 = self.demand_data[k]
            point_cdf = C_API.cond_bivariate_power_margin_cdf_py_interface(
                np.int32(self.convgen1["min"]),
                np.int32(self.convgen2["min"]),
                np.int32(self.convgen1["max"]),
                np.int32(self.convgen2["max"]),
                ffi.cast("double *", self.convgen1["cdf_values"].ctypes.data),
                ffi.cast("double *", self.convgen2["cdf_values"].ctypes.data),
                np.int32(x1),
                np.int32(x2),
                np.int32(net_demand1),
                np.int32(net_demand2),
                np.int32(demand1),
                np.int32(demand2),
                np.int32(itc_cap),
                np.int32(policy == "share"),
            )

            cdf += point_cdf

        return cdf / n

    def system_lolp(self, itc_cap: int = 1000):
        """Computes the system-wide post-interconnection loss of load probability. This is, the probability that at least one area will experience a shortfall.

        Args:
            itc_cap (int, optional): Interconnector capacity

        """

        def trapezoid_prob(ulc, c):

            ulc1, ulc2 = ulc
            return C_API.trapezoid_prob_py_interface(
                np.int32(ulc1),
                np.int32(ulc2),
                np.int32(c),
                np.int32(self.convgen1["min"]),
                np.int32(self.convgen2["min"]),
                np.int32(self.convgen1["max"]),
                np.int32(self.convgen2["max"]),
                ffi.cast("double *", self.convgen1["cdf_values"].ctypes.data),
                ffi.cast("double *", self.convgen2["cdf_values"].ctypes.data),
            )

        n = len(self.net_demand_data)
        gen = self.gen_distribution
        lolp = 0

        c = itc_cap
        for k in range(n):
            net_demand1, net_demand2 = self.net_demand_data[k]
            # system-wide lolp does not depend on the policy
            point_lolp = (
                gen.cdf(np.array([net_demand1 - c - 1, np.Inf]))
                + gen.cdf(np.array([np.Inf, net_demand2 - c - 1]))
                - gen.cdf(np.array([net_demand1 + c, net_demand2 - c - 1]))
                + trapezoid_prob((net_demand1 - c - 1, net_demand2 + c), 2 * c)
            )
            lolp += point_lolp

        return lolp / n

    def lole(self, itc_cap: int = 1000, policy: str = "veto", area: int = 0):
        """Computes the post-interconnection loss of load expectation.

        Args:
            itc_cap (int, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, capacity shortfalls are shared according to demand proportions across areas. Shortfalls can extend from one area to another by diverting power.
            area (int, optional): Area for which to evaluate LOLE; if area=-1, system-wide lole is returned
        """
        if area == -1:
            return self.season_length * self.system_lolp(itc_cap)

        x = np.array([np.Inf, np.Inf])
        x[area] = -1
        # m = (-1,np.Inf)
        lolp = self.cdf(x=x, itc_cap=itc_cap, policy=policy)

        return self.season_length * lolp

    def swap_axes(self):
        """Utility method to flip components in bivariate distribution objects"""
        self.demand_data = np.flip(self.demand_data, axis=1)
        self.renewables_data = np.flip(self.renewables_data, axis=1)
        self.net_demand_data = np.flip(self.net_demand_data, axis=1)

        aux = copy.deepcopy(self.convgen1)
        self.convgen1 = copy.deepcopy(self.convgen2)
        self.convgen2 = aux

        self.gen_distribution = Independent(
            x=self.gen_distribution.y, y=self.gen_distribution.x
        )

    def eeu(self, itc_cap: int = 1000, policy: str = "veto", area: int = 0) -> float:
        """Computes the post-interconnection expected energy unserved (EEU).
        
        Args:
            itc_cap (int, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, capacity shortfalls are shared according to demand proportions across areas. Shortfalls can extend from one area to another by diverting power.
            area (int, optional): Area for which to evaluate eeu; if area=-1, systemwide eeu is returned
        
        Returns:
            float: Estimated EEU
        
        """

        if area == -1:
            return self.eeu(itc_cap, policy, 0) + self.eeu(itc_cap, policy, 1)

        if area == 1:
            self.swap_axes()

        n = len(self.net_demand_data)
        epus = 0

        for k in range(n):
            # print(i)
            net_demand1, net_demand2 = self.net_demand_data[k]
            d1, d2 = self.demand_data[k]
            if policy == "share":
                point_EPU = C_API.cond_eeu_share_py_interface(
                    np.int32(d1),
                    np.int32(d2),
                    np.int32(net_demand1),
                    np.int32(net_demand2),
                    np.int32(itc_cap),
                    np.int32(self.convgen1["min"]),
                    np.int32(self.convgen2["min"]),
                    np.int32(self.convgen1["max"]),
                    np.int32(self.convgen2["max"]),
                    ffi.cast("double *", self.convgen1["cdf_values"].ctypes.data),
                    ffi.cast("double *", self.convgen2["cdf_values"].ctypes.data),
                    ffi.cast("double *", self.convgen1["expectation_vals"].ctypes.data),
                )
            elif policy == "veto":
                point_EPU = C_API.cond_eeu_veto_py_interface(
                    np.int32(net_demand1),
                    np.int32(net_demand2),
                    np.int32(itc_cap),
                    np.int32(self.convgen1["min"]),
                    np.int32(self.convgen2["min"]),
                    np.int32(self.convgen1["max"]),
                    np.int32(self.convgen2["max"]),
                    ffi.cast("double *", self.convgen1["cdf_values"].ctypes.data),
                    ffi.cast("double *", self.convgen2["cdf_values"].ctypes.data),
                    ffi.cast("double *", self.convgen1["expectation_vals"].ctypes.data),
                )
            else:
                raise ValueError(f"Policy name ({policy}) not recognised.")

            epus += point_EPU / n

        if area == 1:
            self.swap_axes()

        return epus * self.season_length

    def get_pointwise_risk(
        self, x: np.ndarray, itc_cap: int = 1000, policy: str = "veto"
    ) -> np.ndarray:
        """Calculates the post-interconnection time-collapsed shortfall probability for each one of the net demand observations
        
        Args:
            x (np.ndarray): point to evaluate CDF at
            itc_cap (int, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'
        
        
        Returns:
            np.ndarray: array with post-interconnection time-collapsed shortfall probabilities
        """
        pointwise_cdfs = np.empty((len(self.demand_data),))
        for k, (demand_row, renewables_row) in enumerate(
            zip(self.demand_data, self.renewables_data)
        ):
            pointwise_cdfs[k] = type(self)(
                demand_data=demand_row.reshape((1, 2)),
                renewables_data=renewables_row.reshape((1, 2)),
                gen_distribution=self.gen_distribution,
            ).cdf(x=x, itc_cap=itc_cap, policy=policy)

        return pointwise_cdfs

    def simulate(self, size: int, itc_cap: int = 1000, policy="veto"):
        """Simulate the post-interconnection bivariate surplus distribution

        Args:
            size (int): Sample size
            itc_cap (int, optional): Interconnector capacity
            policy (str): one of 'veto' or 'share'

        """
        return self.simulate_region(
            size, np.array([np.Inf, np.Inf]), itc_cap, policy, False
        )

    def simulate_region(
        self,
        size: int,
        upper_bounds: np.ndarray,
        itc_cap: int = 1000,
        policy: str = "veto",
        shortfall_region: bool = False,
    ):

        """Simulate post-interconnection bivariate surplus distribution conditioned to a rectangular region bounded above.

        Args:
            size (int): Sample size
            upper_bounds (np.ndarray): region's upper bounds
            itc_cap (int, optional): Interconnector capacity
            policy (str): one of 'veto' or 'share'
            shortfall_region (bool, optional): If True, upper bounds are ignored and the sampling region becomes the shortfall region, this is, min(S_1, S_2) < 0, or equivalently, that in which at least one area has a shortfall.

        """
        seed = np.random.randint(low=0, high=1e8)
        upper_bounds = np.clip(
            upper_bounds, a_min=-self.MARGIN_BOUND, a_max=self.MARGIN_BOUND
        )
        m1, m2 = upper_bounds
        n = len(self.demand_data)

        simulated = np.ascontiguousarray(np.zeros((size, 2)), dtype=np.int32)
        # convgen1, convgen2 = self.gen_distribution.x, self.gen_distribution.y
        ### calculate conditional probability of each historical observation conditioned to the region of interest
        if shortfall_region:
            warnings.warn(
                "Simulating from shortfall region; ignoring passed upper bounds.",
                stacklevel=2,
            )
            pointwise_cdfs = (
                self.get_pointwise_risk(
                    x=np.array([0, np.Inf]), itc_cap=itc_cap, policy=policy
                )
                + self.get_pointwise_risk(
                    x=np.array([np.Inf, 0]), itc_cap=itc_cap, policy=policy
                )
                - self.get_pointwise_risk(
                    x=np.array([0, 0]), itc_cap=itc_cap, policy=policy
                )
            )
            intersection = False
        else:
            pointwise_cdfs = self.get_pointwise_risk(
                x=upper_bounds, itc_cap=itc_cap, policy=policy
            )
            intersection = True

        # numerical rounding error sometimes output negative probabilities of the order of 1e-30
        pointwise_cdfs = np.clip(pointwise_cdfs, a_min=0.0, a_max=np.Inf)

        total_prob = np.sum(pointwise_cdfs)
        if total_prob <= 1e-8:
            if fixed_area == 1:
                self.swap_axes()
            raise Exception(
                f"Region has probability {total_prob}; too small to simulate accurately"
            )
        else:
            probs = pointwise_cdfs / total_prob

            samples_per_row = np.random.multinomial(
                n=size, pvals=probs, size=1
            ).reshape((len(probs),))
            nonzero_samples = samples_per_row > 0
            ## only pass rows which induce at least one simulated value
            row_weights = np.ascontiguousarray(
                samples_per_row[nonzero_samples], dtype=np.int32
            )

            net_demand = np.ascontiguousarray(
                self.net_demand_data[nonzero_samples, :], dtype=np.int32
            )

            demand = np.ascontiguousarray(
                self.demand_data[nonzero_samples, :], dtype=np.int32
            )

            C_API.region_simulation_py_interface(
                np.int32(size),
                ffi.cast("int *", simulated.ctypes.data),
                np.int32(self.convgen1["min"]),
                np.int32(self.convgen2["min"]),
                np.int32(self.convgen1["max"]),
                np.int32(self.convgen2["max"]),
                ffi.cast("double *", self.convgen1["cdf_values"].ctypes.data),
                ffi.cast("double *", self.convgen2["cdf_values"].ctypes.data),
                ffi.cast("int *", net_demand.ctypes.data),
                ffi.cast("int *", demand.ctypes.data),
                ffi.cast("int *", row_weights.ctypes.data),
                np.int32(net_demand.shape[0]),
                np.int32(m1),
                np.int32(m2),
                np.int32(itc_cap),
                int(seed),
                int(intersection),
                int(policy == "share"),
            )

            return simulated

    def simulate_conditional(
        self,
        size: int,
        fixed_value: int,
        fixed_area: int,
        itc_cap: int = 1000,
        policy: str = "veto",
    ):
        """Simulate post-interconnection capacity surplus distribution conditioned to a value in the other area's surplus

        Args:
            size (int): Sample size
            fixed_value (int): Surplus value conditioned on
            fixed_area (TYPE): Area conditioned on
            itc_cap (int, optional): Interconnector capacity
            policy (str): one of 'veto' or 'share'

        """
        seed = np.random.randint(low=0, high=1e8)
        m1 = np.clip(fixed_value, a_min=-self.MARGIN_BOUND, a_max=self.MARGIN_BOUND)
        m2 = self.MARGIN_BOUND

        simulated = np.ascontiguousarray(np.zeros((size, 2)), dtype=np.int32)

        ### calculate conditional probability of each historical observation given
        ### margin value tuple m

        if fixed_area == 0:
            x = np.array([fixed_value, np.Inf])
            y = x - 1
        else:
            x = np.array([np.Inf, fixed_value])
            y = x - 1

        pointwise_cdfs = self.get_pointwise_risk(
            x=x, itc_cap=itc_cap, policy=policy
        ) - self.get_pointwise_risk(x=y, itc_cap=itc_cap, policy=policy)

        pointwise_cdfs = np.clip(pointwise_cdfs, a_min=0.0, a_max=np.Inf)

        ## rounding errors can make probabilities negative of the order of 1e-60
        total_prob = np.sum(pointwise_cdfs)

        if total_prob <= 1e-12:
            raise Exception(
                f"Region has low probability ({total_prob}); too small to simulate accurately"
            )
        else:
            probs = pointwise_cdfs / total_prob

            samples_per_row = np.random.multinomial(
                n=size, pvals=probs, size=1
            ).reshape((len(probs),))
            nonzero_samples = samples_per_row > 0
            ## only pass rows which induce at least one simulated value
            row_weights = np.ascontiguousarray(
                samples_per_row[nonzero_samples], dtype=np.int32
            )

            if fixed_area == 1:
                self.swap_axes()

            net_demand = np.ascontiguousarray(
                self.net_demand_data[nonzero_samples, :], dtype=np.int32
            )

            demand = np.ascontiguousarray(
                self.demand_data[nonzero_samples, :], dtype=np.int32
            )

            C_API.conditioned_simulation_py_interface(
                np.int32(size),
                ffi.cast("int *", simulated.ctypes.data),
                np.int32(self.convgen1["min"]),
                np.int32(self.convgen2["min"]),
                np.int32(self.convgen1["max"]),
                np.int32(self.convgen2["max"]),
                ffi.cast("double *", self.convgen1["cdf_values"].ctypes.data),
                ffi.cast("double *", self.convgen2["cdf_values"].ctypes.data),
                ffi.cast("int *", net_demand.ctypes.data),
                ffi.cast("int *", demand.ctypes.data),
                ffi.cast("int *", row_weights.ctypes.data),
                np.int32(net_demand.shape[0]),
                np.int32(m1),
                np.int32(itc_cap),
                int(seed),
                int(policy == "share"),
            )

            if fixed_area == 1:
                self.swap_axes()

        return simulated[
            :, 1
        ]  # first column has variable conditioned on (constant value)


class UnivariateSequential(BaseCapacityModel, BasePydanticModel):

    """Univariate model for capacity surplus using a sequential available conventional generation model, implementing Monte Carlo evaluations through map-reduce patterns. Worker instances are of type UnivariateTraces.

    Args:
        gen_dir (str): folder with conventional generation data
        demand (np.ndarray): one-dimensionsl demand data
        renewables (np.ndarray): one-dimensional renewables data
        season_length (int): number of timesteps per peak season
        n_cores (int, optional): number of cores to use for map-reduce operations
        offset (float, optional): offset parameter added to loaded traces. Defaults to 0.0
        scale (float, optional): multiplicative rescaling factor for loaded traces. Defaults to 1.0

    """

    gen_dir: str
    demand: np.ndarray
    renewables: np.ndarray
    season_length: int
    n_cores: t.Optional[int] = 2
    offset: float = 0.0
    scale: float = 1.0

    _worker_class = UnivariateTraces

    @validator("demand", "renewables", allow_reuse=True)
    def check_shape_and_order(cls, data):
        # ensures passed arrays are one-dimensional and follow row-major order
        return np.ascontiguousarray(data, np.float32).reshape(-1)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def _persist_gen_traces(
        cls, args: t.Tuple[acg_models.Sequential, t.Dict, Path, int]
    ) -> None:
        """Persists a sequence of traces according to specified arguments as a numpy file

        Args:
            args (t.Tuple[acg_models.Sequential, t.Dict, Path]): trace generation parameters
        """
        gen, call_kwargs, filename, compress_files = args
        traces = gen.simulate_seasons(**call_kwargs)
        if compress_files:
            np.savez_compressed(filename, traces=traces)
        else:
            np.save(filename, traces)

    @classmethod
    def init(
        cls,
        output_dir: str,
        n_traces: int,
        n_files: int,
        gen: acg_models.Sequential,
        demand: np.ndarray,
        renewables: np.ndarray,
        season_length: int,
        n_cores: int = 4,
        burn_in: int = 100,
        seed: int = None,
        compress_files: bool = False,
        offset: float = 0.0,
        scale: float = 1.0
    ) -> UnivariateSequential:
        """Generate and persists traces of conventional generation in multiple files, and uses them to instantiate a capacity surplus model. Returns a surplus model ready to perform computations on the generated files.
        
        Args:
            output_dir (str): Output directory for trace files
            n_traces (int): Total number of season traces to simulate
            n_files (int): Number of files to create. Making this a multiple of the available number of cores and ensuring that each file is on the order of 500 MB (~ 125 million floats) is probably optimal.
            gen (acg_models.Sequential): Sequential conventional generation instance.
            demand (np.ndarray): Demand data
            renewables (np.ndarray): renewable generation data
            season_length (int): Peak season length.
            n_cores (int, optional): Number of cores to use.
            burn_in (int, optional): Parameter passed to acg_models.Sequential.simulate_seasons.
            seed (int, optional): Random seed passed to C backend. If not passed, output file paths are hashed to obtained it; this is because different seeds are needed for each file, otherwise traces are identical across files.
            compress_files (bool, optional): Whether ACG trace files should be compressed
            offset (float, optional): offset parameter added to loaded traces. Defaults to 0.0
            scale (float, optional): multiplicative rescaling factor for loaded traces. Defaults to 1.0
        
        Returns:
            UnivariateSequential: Sequential surplus model
        
        """

        # create dir if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if len(demand) != len(renewables):
            raise ValueError("demand and renewables must have the same length.")

        trace_length = len(demand)

        if trace_length % season_length != 0:
            raise ValueError("trace_length must be divisible by season_length.")

        if n_traces <= 0 or not isinstance(n_traces, int):
            raise ValueError("n_traces must be a positive integer")

        if n_files <= 0 or not isinstance(n_files, int):
            raise ValueError("n_files must be a positive integer")

        # compute file size (in terms of number of traces)
        file_sizes = [int(n_traces / n_files) for k in range(n_files)]
        file_sizes[-1] += n_traces - sum(file_sizes)

        # create argument list for multithreaded execution
        arglist = []
        seasons_per_trace = int(trace_length / season_length)
        for idx, file_size in enumerate(file_sizes):
            output_path = Path(output_dir) / str(idx)
            file_seed = (
                seed + idx
                if seed is not None
                else abs(adler32(str(output_path).encode("utf-8"))) % (1024 * 1024)
            )
            call_kwargs = {
                "size": file_size,
                "season_length": season_length,
                "seasons_per_trace": seasons_per_trace,
                "burn_in": burn_in,
                "seed": file_seed,
            }
            arglist.append((gen, call_kwargs, output_path, compress_files))

        # create files in parallel
        with Pool(n_cores) as executor:
            jobs = list(
                tqdm(
                    executor.imap(cls._persist_gen_traces, arglist), total=len(arglist)
                )
            )

        return cls(
            gen_dir=output_dir,
            demand=np.array(demand),
            renewables=np.array(renewables),
            season_length=season_length,
            n_cores=n_cores,
            offset=offset,
            scale=scale
        )

    def create_mapred_arglist(
        self, mapper: t.Union[str, t.Callable], str_map_kwargs: t.Dict
    ) -> t.List[t.Dict]:
        """Create named arguments list to instantiate each worker in map reduce execution

        Args:
            mapper (t.Union[str, t.Callable]): If a string, the method of that name is called on each worker instance. If a function, it must take as only argument a worker instance.
            str_map_kwargs (t.Tuple, optional): Named arguments passed to the mapper function when it is passed as a string.

        Returns:
            t.List[t.Any]: Named arguments list
        """
        arglist = []
        # create arglist for parallel execution
        for file in Path(self.gen_dir).iterdir():
            kwargs = {
                "gen_filepath": str(file),
                "demand": self.demand,
                "renewables": self.renewables,
                "season_length": self.season_length,
                "offset": self.offset,
                "scale": self.scale
            }
            arglist.append((kwargs, mapper, str_map_kwargs))

        return arglist

    @classmethod
    def execute_map(
        cls, call_args: t.Tuple[t.Dict, t.Union[str, t.Callable], t.Tuple]
    ) -> t.Tuple[t.Any, int]:
        """Instantiate a worker with the passed arguments and execute mapper function on it. Returns both the result of the mapper function and the number of traces processed; the latter is helpful when results from the mappers are aggregated, e.g. global averaging.

        Args:
            call_args (t.Tuple[t.Dict, t.Union[str, t.Callable], t.Tuple]): A triplet with named arguments to instantiate the workers, the function to call on instantiated workers as a string or callable object, and additional unnamed arguments passed to the mapper if given as a string.

        Returns:
            t.Tuple[t.Any, int]: tuple with mapper output and the number of traces processed

        """
        worker_kwargs, map_func, str_map_kwargs = call_args
        worker = cls._worker_class(**worker_kwargs)
        n_traces = worker.n_traces
        if isinstance(map_func, str):
            return getattr(worker, map_func)(**str_map_kwargs), n_traces
        elif isinstance(map_func, t.Callable):
            return map_func(worker), n_traces
        else:
            raise ValueError("map_func must be a string or a function.")

    def map_reduce(
        self,
        mapper: t.Union[str, t.Callable],
        reducer: t.Optional[t.Callable],
        str_map_kwargs: t.Dict = {},
    ) -> t.Any:
        """Performs map-reduce processing operations on each persisted generation trace file, given mapper and reducer functions

        Args:
            mapper (t.Union[str, t.Callable]): If a string, the method of that name is called on each worker instance (of class UnivariateTraces). If a function, it must take as only argument a worker instance.
            reducer (t.Optional[t.Callable]): This function must take as input a list where each entry is a tuple with the mapper output and the number of traces processed by the mapper, in that order. If None, no reducer is applied.
            str_map_kwargs (t.Dict, optional): Named arguments passed to the mapper function when passed as a string.

        Returns:
            t.Any: Map-reduce output

        """

        arglist = self.create_mapred_arglist(mapper, str_map_kwargs)

        with Pool(self.n_cores) as executor:
            mapped = list(
                tqdm(executor.imap(self.execute_map, arglist), total=len(arglist))
            )

        if reducer is not None:
            return reducer(mapped)
        else:
            return mapped

    def cdf(self, x: float) -> float:
        """Computes the surplus' cumulative distribution function (CDF) evaluated at a point

        Args:
            x (float): Point at which to evaluate the CDF

        Returns:
            float: CDF estimate
        """

        def reducer(mapped):
            n_traces = np.sum([n for _, n in mapped])
            return np.array([n * val for val, n in mapped]).sum() / n_traces

        return self.map_reduce(mapper="cdf", reducer=reducer, str_map_kwargs={"x": x})

    def simulate(self):
        raise NotImplementedError(
            "This class does not implement a simulate() method. Use get_surplus_df() to get the trace of shortfalls or the full trace of surplus values; alternatively see methods simulate_eu() and simulate_lold()."
        )

    def lole(self) -> float:
        """Computes the loss of load expectation

        Returns:
            float: lole estimate
        """
        return self.season_length * self.cdf(
            x=-1e-1
        )  # tiny offset to avoid issues with numerical rounding errors from adding millions of numbers together

    def eeu(self) -> float:
        """Computes the expected energy unserved (EEU)

        Returns:
            float: estimated EEU
        """

        def reducer(mapped):
            n_traces = np.sum([n for _, n in mapped])
            return np.array([n * val for val, n in mapped]).sum() / n_traces

        return self.map_reduce(mapper="eeu", reducer=reducer)

    def simulate_eu(self) -> np.ndarray:
        """Simulates per-peak-season energy userved

        Returns:
            np.ndarray: array with one entry per peak season
        """

        def reducer(mapped):
            eu_samples = np.concatenate([samples for samples, _ in mapped], axis=0)
            return eu_samples

        return self.map_reduce(mapper="simulate_eu", reducer=reducer)

    def simulate_lold(self):
        """Simulates per-peak-season loss of load duration

        Returns:
            np.ndarray: array with one entry per peak season
        """

        def reducer(mapped):
            lold_samples = np.concatenate([samples for samples, _ in mapped], axis=0)
            return lold_samples

        return self.map_reduce(mapper="simulate_lold", reducer=reducer)

    def get_surplus_df(self, shortfalls_only: bool = True) -> pd.DataFrame:
        """Returns a data frame with time occurrence information of observed surplus values and shortfalls.

        Args:
            shortfalls_only (bool, optional): If True, only shortfall rows are returned

        Returns:
            pd.DataFrame: A data frame with the surplus values, a 'season_time' column with the within-season time of occurrence (0,1,...,season_length-1), a 'file_id' column that indicates which file was used to compute the value, and a 'season' column to indicate which season the value was observed in.

        """

        def reducer(mapped):
            # compute global season number when merging results from different files (each with their own season numbering)
            trace_length = len(self.demand)
            seasons_per_trace = trace_length / self.season_length
            if seasons_per_trace != int(seasons_per_trace):
                raise ValueError(
                    f"trace length ({trace_length}) is not a multiple of season length ({season_length})"
                )
            seasons_per_trace = int(seasons_per_trace)
            past_seasons = 0
            for df, n_traces in mapped:
                df["season"] += past_seasons
                past_seasons += n_traces * seasons_per_trace
            return pd.concat([df for df, n in mapped])

        return self.map_reduce(
            mapper="get_surplus_df",
            reducer=reducer,
            str_map_kwargs={"shortfalls_only": shortfalls_only},
        )

    def __str__(self):
        return f"Map-reduce based sequential surplus model using trace files in {self.gen_dir}"


class BivariateSequential(UnivariateSequential):

    """Bivariate model for capacity surplus using a sequential available conventional generation model, implementing Monte Carlo evaluations through map-reduce patterns. Worker instances are of type BivariateTraces.

    Args:
        gen_dir (str): folder with conventional generation data
        demand (np.ndarray): two-dimensional demand array with two columns
        renewables (np.ndarray): two-dimensional renewables data array with two columns
        season_length (int): number of timesteps per peak season
        n_cores (int, optional): number of cores to use for map-reduce operations
        offsets (t.Iterable, optional): offset parameters added to loaded traces for each area. Defaults to [0.0, 0.0]
        scales (t.Iterable, optional): multiplicative rescaling factors for loaded traces in each area. Defaults to [1.0, 1.0]
    """
    offsets: t.Iterable = [0.0, 0.0]
    scales: t.Iterable = [1.0, 1.0]
    _worker_class = BivariateTraces
    _area_indices = [0, 1]

    @validator("demand", "renewables", allow_reuse=True)
    def check_shape_and_order(cls, data):
        # ensures passed arrays are twi-dimensional and follow row-major order
        if len(data.shape) != 2 or data.shape[1] != 2:
            raise ValueError("passed data arrays must be two-dimensional with two columns")
        return np.ascontiguousarray(data, np.float32)

    @property
    def filedirs(self):
        return [Path(self.gen_dir) / str(area) for area in self._area_indices]

    @classmethod
    def init(
        cls,
        output_dir: str,
        n_traces: int,
        n_files: int,
        gens: t.List[acg_models.Sequential],
        demand: np.ndarray,
        renewables: np.ndarray,
        season_length: int,
        n_cores: int = 4,
        burn_in: int = 100,
        compress_files: bool = False,
        offsets: float = [0.0, 0.0],
        scales: float = [1.0, 1.0]
    ) -> BivariateSequential:
        """Generate and persists traces of conventional generation in multiple files, and uses them to instantiate a capacity surplus model. Returns a model ready to perform computations on the generated files.

        Args:
            output_dir (str): output directory for trace files
            n_traces (int): Total number of traces to simulate; a trace is a sequence of at least one peak season
            n_files (int): Number of files to create. Making this a multiple of the available number of cores and ensuring that each file is on the order of 500 MB (~ 125 million floats) is probably good enough.
            gens (acg_models.Sequential): List of sequential conventional generation instances, one per system.
            demand (np.ndarray): Demand data
            renewables (np.ndarray): Renewables data
            season_length (int): Peak season length.
            n_cores (int, optional): Number of cores to use.
            burn_in (int, optional): Parameter passed to acg_models.Sequential.simulate_seasons.
            compress_files (bool, optional): Whether ACG trace files should be compressed
            offsets (t.Iterable, optional): offset parameters added to loaded traces for each area. Defaults to [0.0, 0.0]
            scales (t.Iterable, optional): multiplicative rescaling factors for loaded traces in each area. Defaults to [1.0, 1.0]

        Returns:
            BivariateSequential: Sequential surplus model


        """
        for area, gen, univar_demand, univar_renewables, offset, scale in zip(
            cls._area_indices, gens, demand.T, renewables.T, offsets, scales
        ):
            out_dir = Path(output_dir) / str(area)
            UnivariateSequential.init(
                output_dir=str(out_dir),
                n_traces=n_traces,
                n_files=n_files,
                gen=gen,
                demand=univar_demand,
                renewables=univar_renewables,
                season_length=season_length,
                n_cores=n_cores,
                burn_in=burn_in,
                compress_files=compress_files,
                offset=offset,
                scale=scale
            )

        return cls(
            gen_dir=output_dir,
            demand=np.array(demand),
            renewables=np.array(renewables),
            season_length=season_length,
            n_cores=n_cores,
            offsets=offsets,
            scales=scales
        )

    def create_mapred_arglist(
        self, mapper: t.Union[str, t.Callable], str_map_kwargs: t.Dict, policy: str
    ) -> t.List[t.Dict]:
        """Create named arguments list to instantiate each worker in map reduce execution

        Args:
            mapper (t.Union[str, t.Callable]): If a string, the method of that name is called on each worker instance. If a function, it must take as only argument a worker instance.
            str_map_kwargs (t.Tuple, optional): Named arguments passed to the mapper function when it is passed as a string.
            policy (str, optional): shortfall-sharing interconnection policy

        Returns:
            t.List[t.Any]: Named arguments list
        """
        arglist = []

        # use univariate class logic to build named argument lists, whose instances are then passed as arguments to bivariate models.
        univariate_trace_pairs = []
        for filedir, demand_array, renewables_array, offset, scale in zip(
            self.filedirs, self.demand.T, self.renewables.T, self.offsets, self.scales
        ):
            univar_model = UnivariateSequential(
                gen_dir=str(filedir),
                demand=demand_array,
                renewables=renewables_array,
                season_length=self.season_length,
                offset=offset,
                scale=scale
            )

            univar_arglist = univar_model.create_mapred_arglist(
                mapper=mapper, str_map_kwargs=str_map_kwargs
            )
            # we only care for named arguments to initialise univariate surplus models
            univariate_trace_pairs.append(
                [UnivariateTraces(**named_args) for named_args, _, _ in univar_arglist]
            )

        # policy is passed here at the worker instantiation level. This is to take advantage of bivariate surplus code from the iid module. Said module implements everything for a veto policy, so it is reused. But said module does not take policy as an argument, and to overcome this in the inheriting subclass, the policy is passed at instantiation time and a reimplementation of the itc_flow method in BivariateTraces looks at the passed value to pick the correct flow equations. This is convoluted but avoids code duplication.

        # each trace here correspond to an area in the system
        for trace_x, trace_y in zip(*univariate_trace_pairs):
            bivariate_args = {
                "univariate_traces": [trace_x, trace_y],
                "season_length": self.season_length,
                "policy": policy,
            }
            arglist.append((bivariate_args, mapper, str_map_kwargs))

        return arglist

    def map_reduce(
        self,
        mapper: t.Union[str, t.Callable],
        reducer: t.Optional[t.Callable],
        str_map_kwargs: t.Dict = {},
        policy: str = "veto",
        itc_cap: float = 1000.0,
    ) -> t.Any:
        """Performs map-reduce processing operations on each persisted generation trace file, given mapper and reducer functions

        Args:
            mapper (t.Union[str, t.Callable]): If a string, the method with that name is called on each worker instance (of class `BivariateTraces`). If a function, it must take as only argument a worker instance.
            reducer (t.Optional[t.Callable]): This function must take as input a list where each entry is a tuple with the mapper output and the number of traces processed by the mapper, in that order. If None, the mapper output is returned.
            str_map_kwargs (t.Dict, optional): Named arguments passed to the mapper method when passed as a string.
            policy (str, optional): shortfall-sharing interconnection policy
            itc_cap (float, optional): Description

        Returns:
            t.Any: Description

        """

        # itc_cap will be passed as an extra named argument to the mapper function, because it is an argument in all of BaseBivariateMonteCarlo methods, which are used to perform the calculations
        str_map_kwargs["itc_cap"] = itc_cap

        # policy is passed as an argument at worker instantiation time to avoid code duplication. See comments on the create_mapred_arglist method.
        arglist = self.create_mapred_arglist(mapper, str_map_kwargs, policy)

        with Pool(self.n_cores) as executor:
            mapped = list(
                tqdm(executor.imap(self.execute_map, arglist), total=len(arglist))
            )

        if reducer is not None:
            return reducer(mapped)
        else:
            return mapped

    def cdf(self, x: np.ndarray, itc_cap: float = 1000.0, policy="veto"):
        """Evaluates the bivariate post-interconnection time-collapsed capacity surplus distribution's cumulative distribution function

        Args:
            x (np.ndarray): value at which to evaluate the cdf
            itc_cap (int, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, capacity shortfalls are shared according to demand proportions across areas. Shortfalls can extend from one area to another by diverting power.

        """

        def reducer(mapped):
            n_traces = np.sum([n for _, n in mapped])
            return np.array([n * val for val, n in mapped]).sum() / n_traces

        return self.map_reduce(
            mapper="cdf",
            reducer=reducer,
            itc_cap=itc_cap,
            policy=policy,
            str_map_kwargs={"x": x},
        )

    def lole(self, itc_cap: float = 1000.0, policy="veto", area: int = 0) -> float:
        """Computes the post-interconnection loss of load expectation (LOLE)
        
        Args:
            itc_cap (float, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.
            area (int, optional): Area for which to evaluate LOLE; if area=-1, system-wide lole is returned
        
        Returns:
            float: estimated LOLE
        
        """
        offset = (
            -1e-1
        )  # this avoids numerical issues from adding up millions of numbers in the calculations
        if area in [0, 1]:
            x = np.zeros((2,), dtype=np.float32) + offset  # tiny offset
            x[1 - area] = np.Inf
            return self.season_length * self.cdf(x, itc_cap=itc_cap, policy=policy)
        elif area == -1:
            x = np.array([offset, np.Inf])
            prob = (
                self.cdf(x, itc_cap=itc_cap, policy=policy)
                + self.cdf(np.flip(x), itc_cap=itc_cap, policy=policy)
                - self.cdf(np.minimum(offset, x), itc_cap=itc_cap, policy=policy)
            )
            return self.season_length * prob
        else:
            raise ValueError("area must be in [-1,0,1]")

    def eeu(self, itc_cap: float = 1000.0, policy="veto", area: int = 0) -> float:
        """Computes the post-interconnection expected energy unserved (EEU)
        
        Args:
            itc_cap (float, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.
            area (int, optional): Area for which to evaluate eeu; if area=-1, systemwide eeu is returned
        
        Returns:
            float: estimated EEU
        """

        def reducer(mapped):
            n_traces = np.sum([n for _, n in mapped])
            return np.array([n * val for val, n in mapped]).sum() / n_traces

        return self.map_reduce(
            mapper="eeu",
            reducer=reducer,
            itc_cap=itc_cap,
            policy=policy,
            str_map_kwargs={"area": area},
        )

    def get_surplus_df(
        self, shortfalls_only: bool = True, itc_cap: float = 1000.0, policy="veto"
    ) -> pd.DataFrame:
        """Returns a data frame with time occurrence information of observed surplus values and shortfalls.

        Args:
            shortfalls_only (bool, optional): If True, only shortfall rows are returned
            itc_cap (int, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.

        Returns:
            pd.DataFrame: A data frame with the surplus values, a 'season_time' column with the within-season time of occurrence (0,1,...,season_length-1), a 'file_id' column that indicates which file was used to compute the value, and a 'season' column to indicate which season the value was observed in.

        """
        def reducer(mapped):
            return pd.concat([df for df, n in mapped])

        return self.map_reduce(
            mapper="get_surplus_df",
            reducer=reducer,
            str_map_kwargs={"shortfalls_only": shortfalls_only},
            itc_cap=itc_cap,
            policy=policy,
        )

    def __str__(self):
        return f"Map-reduce based sequential surplus model using trace files in {self.gen_dir}"

    def simulate_eu(self, itc_cap: float = 1000.0, policy="veto") -> np.ndarray:
        """Simulates per-peak-season energy unserved for both areas

        Args:
            itc_cap (int, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.

        Returns:
            np.ndarray: array with one row per peak season and one column per area
        """

        def reducer(mapped):
            eu_samples = np.concatenate([sample for sample, _ in mapped], axis=0)
            return eu_samples

        return self.map_reduce(
            mapper="simulate_eu",
            reducer=reducer,
            itc_cap=itc_cap,
            policy=policy,
        )

    def simulate_lold(self, itc_cap: float = 1000.0, policy="veto") -> np.ndarray:
        """Simulates per-peak-season loss of load duration for both areas

        Args:
            itc_cap (int, optional): interconnection capacity
            policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.

        Returns:
            np.ndarray: array with one row per peak season and one column per area
        """

        def reducer(mapped):
            lold_samples = np.concatenate([sample for sample, _ in mapped], axis=0)
            return lold_samples

        return self.map_reduce(
            mapper="simulate_lold",
            reducer=reducer,
            itc_cap=itc_cap,
            policy=policy,
        )
