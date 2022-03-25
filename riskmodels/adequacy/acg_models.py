"""
This module implements models for available conventional generation (ACG). The implementation of the non-sequential models assumes generating units can be either fully available or fully unavailable at any given time. The sequential model represents each generating unit as a Markov chain whose transition probabilities are based on their overall availability and mean time to repair values. Both models assume statistical independence between generating units. 

"""
from __future__ import annotations
import warnings
import typing as t

from riskmodels.univariate import Binned

import numpy as np
import pandas as pd

from c_sequential_models_api import ffi, lib as C_API


class NonSequential(Binned):

    """Available conventional generation model in which generators are assumed statistically independent of each other, and each one can be either 100% or 0% available at any given time with a certain probability. No serial correlation between states is assumed, i.e. this is a non-sequential or time-collapsed model. See class methods `from_generator_df` and `from_generator_csv_file` to instantiate this class."""

    @classmethod
    def from_generator_df(cls, df: pd.DataFrame) -> NonSequential:
        """Takes a dataframe object and builds the generation model from it.

        Args:
            df (pd.DataFrame): dataframe with colums 'availability' and 'capacity', where the former is the probability of the generating unit being available and the latter the nameplate capacity; each row represents an individual generator

        Returns:
            NonSequential: fitted model

        """
        warnings.warn("Coercing capacity values to integers")

        capacity_values = np.array(df["capacity"], dtype=np.int32)
        availability_values = np.array(df["availability"])

        if np.any(availability_values < 0) or np.any(availability_values > 1):
            raise Exception(
                f"Availabilities must be in the interval [0,1]; found interval[{min(availability_values)},{max(availability_values)}]"
            )

        max_gen = int(np.sum(capacity_values[capacity_values >= 0]))
        min_gen = int(np.sum(capacity_values[capacity_values < 0]))

        zero_idx = np.abs(
            min_gen
        )  # this is in case there are generators with negative generation
        pdf_length = max_gen + 1 - min_gen
        pdf = np.zeros((pdf_length,), dtype=np.float64)  # initialise pdf values
        pdf[zero_idx] = 1.0

        i = 0
        for c, p in zip(capacity_values, availability_values):
            if c >= 0:
                suffix = pdf[0 : pdf_length - c]
                preffix = np.zeros((c,))
            else:
                preffix = pdf[np.abs(c) : pdf_length]
                suffix = np.zeros((np.abs(c),))
            pdf = (1 - p) * pdf + p * np.concatenate([preffix, suffix])
            i += 1

        support = np.arange(min_gen, max_gen + 1)
        return Binned(support=support, pdf_values=pdf, data=None)

    @classmethod
    def from_generator_csv_file(cls, file_path: str, **kwargs) -> NonSequential:
        """Takes a csv file and builds the generation model

        Args:
            file_path (str): Path to csv file. It must have colums 'availability' and 'capacity', where the former is the probability of the generating unit being available and the latter the nameplate capacity; each row represents an individual generator
            **kwargs: additional arguments passed to pandas.read_csv

        Returns:
            NonSequential: fitted model
        """
        df = pd.read_csv(file_path, **kwargs)
        return cls.from_generator_df(df)


class Sequential(NonSequential):

    """Available conventional generation model in which generators are modelled as Markov chains and are assumed to be independent of each other. The methods `from_generator_df` and `from_generator_file` can be used to instantiate this class when 2-state Markov chains are used (on-off availability for each generating unit without de-rated states), see the cited methods for details.
    To simulate Markov chain models with a different set of statess (e.g. de-rated states), the class can be instantiated through its constructor by passing named parameters for the transition matrices and chain states. See the argument specification below.

    Args:
        transition_matrices (np.ndarray): three-dimensional array where axis 0 represents generation units, and axis 1 and 2 represent transition matrix dimensions
        chain_states (np.ndarray): two-dimensional array where axis 0 represents generation units and axis 1 represents the array of the unit's states

    """

    transition_matrices: np.ndarray
    chain_states: np.ndarray

    def __str__(self):
        max_cap = np.sum([s[0] for s in self.chain_states])
        return f"Markov chain generation model with {len(self.transition_matrices)} generators and {max_cap} maximum capacity"

    @property
    def stationary_distributions(self):
        return np.array(
            [self.get_stationary_dist(mat) for mat in self.transition_matrices]
        )

    @classmethod
    def build_chains(cls, df: pd.DataFrame) -> t.Tuple[np.ndarray, np.ndarray]:
        """Takes a dataframe with generator data (availability and mean time to repair) and returns the corresponding Markov chain states and transition matrices. Here, the availability probability determines the stationary distribution

        Args:
            df (pd.DataFrame): Generator data with columns 'availability' and 'mttr'

        Returns:
            t.Tuple[np.ndarray, np.ndarray]
        """

        def get_transition_matrix(generator_data):
            prob, mttr = generator_data
            alpha = 1 - 1 / mttr
            a11 = 1 - (1 - prob) * (1 - alpha) / prob
            mat = np.array([[a11, 1 - a11], [1 - alpha, alpha]], dtype=np.float32)
            mat = mat / np.sum(mat, axis=1)
            return mat

        states = np.array([np.array([x, 0.0]) for x in df["capacity"]], dtype=np.int32)
        transition_matrices = np.apply_along_axis(
            get_transition_matrix, 1, np.array(df[["availability", "mttr"]])
        )

        return states, transition_matrices

    @classmethod
    def sample_stationary_dists(
        cls, transition_matrices: np.ndarray, chain_states: np.ndarray
    ) -> np.ndarray:
        """Sample states from the stationary distribution of transition probability matrices

        Args:
            transition_matrices (np.ndarray): array of transition probability matrices
            chain_states (np.ndarray): two-dimensional array with state vectors for each transition matrix

        Returns:
            np.ndarray: array with sampled state for each transition matrix
        """
        sample = []
        for mat, states in zip(transition_matrices, chain_states):
            stationary_dist = cls.get_stationary_dist(mat)
            s = np.random.choice(states, size=1, p=stationary_dist)
            sample.append(s)

        return np.array(sample)

    @classmethod
    def get_stationary_dist(cls, mat: np.ndarray) -> np.ndarray:
        """Compute stationary probability distribution over states for a Markov chain

        Args:
            mat (np.ndarray): transition probability matrix

        Returns:
            np.ndarray: vector with stationary probability values for each state

        """
        # from somewhere in stackoverflow

        evals, evecs = np.linalg.eig(mat.T)
        evec1 = evecs[:, np.isclose(evals, 1)]

        # Since np.isclose will return an array, we've indexed with an array
        # so we still have our 2nd axis.  Get rid of it, since it's only size 1.
        if evec1.shape[1] == 0:
            raise Exception("Some generators might not have a stationary distribution")
        evec1 = evec1[:, 0]

        stationary = evec1 / evec1.sum()

        # eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
        stationary = stationary.real

        return stationary

    @classmethod
    def from_generator_df(cls, df: pd.DataFrame) -> Sequential:
        """Takes a dataframe object and builds the generation model from it.

        Args:
            df (pd.DataFrame): dataframe with colums 'availability' and 'capacity', where the former is the probability that the generating unit is available (i.e. stationary availability probability) and the latter is the unit's nameplate capacity; in additon, an 'mttr' column with estimated mean times to repair per unit should be present. Each row represents an individual generator.

        Returns:
            Sequential: fitted model

        """

        time_collapsed = super().from_generator_df(df)

        df["capacity"] = df["capacity"].astype(
            np.int32
        )  # for consistency between time-collapsed and time-dependent logic

        states, matrices = cls.build_chains(df)

        return cls(
            pdf_values=time_collapsed.pdf_values,
            support=time_collapsed.support,
            data=time_collapsed.data,
            transition_matrices=matrices,
            chain_states=states,
        )

    @classmethod
    def simulate_chains(
        cls,
        size: int,
        trace_length: int,
        transition_matrices: np.ndarray,
        chain_states: np.ndarray,
        initial_state: np.ndarray = None,
        simulate_escape_time: bool = True,
        output_array: np.ndarray = None,
        seed: int = None,
    ) -> np.Optional[np.ndarray]:
        """Simulate multiple traces in which each one represents the aggregate of multiple Markov chains. This method samples a single large sequential trace and then split it into multiple subtraces; trace endpoints are therefore dependent.

        Args:
            size (int): Number of traces to simulate
            trace_length (int): length of individual traces
            transition_matrices (np.ndarray): three-dimensional array containing transition probability matrices for all chains where the first dimension corresponds to generating units and the last two dimensions correspond to transition matrices.
            chain_states (np.ndarray): two-dimensional array with vectors of chain states for all chains where the first dimension corresponds to generating units and the second one to the state set. Note that every generating unit must have the same number of states.
            initial_state (np.ndarray, optional): one-dimensional array with the initial state indices for all chains. The indices must correspond to a row in the transition matrices. If None, initial states are sampled from the stationary distributions of each chain.
            simulate_escape_time (bool, optional): If True, simulate chains through time-of-escape simulations. If false, simulate each timestep individually.
            output_array (np.ndarray, optional): Array where results are to be stored. If not provided, one is created.
            seed (int, optional): Random seed passed to C backend. If not given, numpy's random numbers are used to initialise it.

        No Longer Returned:
            np.Optional[np.ndarray]: two-dimensional array in which each row represent an individual simulated trace. If an output array is passed as input, None is returned.


        No Longer Raises:
            ValueError: Description


        """
        n_chains = len(chain_states)
        n_states = len(chain_states[0])

        # set output array
        if output_array is not None:
            if (
                not isinstance(output_array, np.ndarray)
                or len(output_array.shape) != 2
                or output_array.shape != (size, trace_length)
                or output_array.dtype != np.float32
            ):
                raise ValueError(
                    "output_array must be a two-dimensional numpy array with shape (size, trace_length) of type numpy.float32"
                )
            return_output = False
        else:
            output_array = np.ascontiguousarray(
                np.zeros((size, trace_length)), dtype=np.float32
            )
            return_output = True

        # if seed is None:
        #   seed = np.random.randint(low=0,high=2**31-1)
        # else:
        #   np.random.seed(seed) #both numpy and C seeds are the same if provided; this is needed for the initial state which is computed in Python
        seed = np.random.randint(low=0, high=2**20 - 1) if seed is None else seed
        # print(f"C seed: {seed}")
        # set initial state array
        if initial_state is None:
            initial_state = cls.sample_stationary_dists(
                transition_matrices, chain_states
            ).reshape(-1)
        else:
            if (
                not isinstance(initial_state, np.ndarray)
                or len(initial_state.shape) != 1
                or len(initial_state) != n_chains
            ):
                # print(initial_state)
                # print(initial_state.shape)
                raise ValueError(
                    "Initial state vector must be a one-dimensional numpy array with as many entries as transition_matrices."
                )

        if np.any(np.array([len(s) for s in chain_states]) != n_states):
            raise ValueError("Number of states must be the same for all chains")

        if np.any(
            np.array([s.shape != (n_states, n_states) for s in transition_matrices])
        ):
            raise ValueError("Matrices must be square and of the same dimensions")

        # call C program
        if trace_length <= 1:
            raise ValueError("Trace length must be an integer larger than 1")

        # cast as float
        initial_state = np.ascontiguousarray(initial_state).astype(np.float32)
        float_chain_states = chain_states.astype(np.float32)

        C_API.simulate_mc_power_grid_py_interface(
            ffi.cast("float *", output_array.ctypes.data),
            ffi.cast("float *", transition_matrices.ctypes.data),
            ffi.cast("float *", float_chain_states.ctypes.data),
            ffi.cast("float *", initial_state.ctypes.data),
            np.int32(n_chains),
            np.int32(size),
            np.int32(
                trace_length - 1
            ),  # initial state is accounted for in trace length
            np.int32(n_states),
            np.int32(seed),
            np.int32(simulate_escape_time),
        )

        if return_output:
            return output_array

    def simulate(self, size: int, seed: int = None) -> np.ndarray:
        """Simulate a single trace of sequential observations

        Args:
            size (int): Number of samples, or equivalently, trace length.
            seed (int, optional): Random seed passed to C backend. If not given, numpy's random numbers are used to initialise it.

        Returns:
            np.ndarray
        """
        return self.simulate_chains(
            size=1,
            trace_length=size,
            transition_matrices=self.transition_matrices,
            chain_states=self.chain_states,
            initial_state=None,
            simulate_escape_time=True,
            seed=seed,
        ).reshape(-1)

    def simulate_seasons(
        self,
        size: int,
        season_length: int,
        seasons_per_trace: int = 1,
        burn_in: int = 100,
        seed: int = None,
    ) -> np.ndarray:
        """Simulate multiple traces of available conventional generation; each trace can have one or more peak seasons in it, depending on whether streaks of multiple years need to be sampled.

        Args:
            size (int): number of traces to sample
            season_length (int): peak season length
            seasons_per_trace (int, optional): Number of seasons per trace. The default is 1.
            burn_in (int, optional): burn-in period between individual peak season traces; this is needed because in order to sample them, a large sequence is generated and subsequently subdivided, thus making trace endpoints correlated if a burn-in period is not allowed.
            seed (int, optional): Random seed passed to C backend. If not given, numpy's random numbers are used to initialise it.

        No Longer Returned:
            np.ndarray: two-dimensional array where each row represent a sampled peak season of available conventional generation.
        """
        total_seasons = size * seasons_per_trace
        augmented_season_length = season_length + burn_in

        output_array = self.simulate_chains(
            size=total_seasons,
            trace_length=augmented_season_length,
            transition_matrices=self.transition_matrices,
            chain_states=self.chain_states,
            initial_state=None,
            simulate_escape_time=True,
            seed=seed,
        )

        # drop burn in periods and reshape
        return output_array[:, 0:season_length].reshape(
            (size, season_length * seasons_per_trace)
        )
