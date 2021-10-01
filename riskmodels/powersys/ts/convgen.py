"""
This module implements a sequential model for available conventional generation, by representing each generator as a Markov chain whose transition probabilities are based on their overall availability and mean time to repair values. Available generation is then the aggregate of all Markov chains at each timestep.
"""
from __future__ import annotations
import warnings
import typing as t

import numpy as np
import pandas as pd

from c_sequential_models_api import ffi, lib as C_API

from riskmodels.powersys.iid.convgen import IndependentFleetModel


class MarkovChainGenerationModel(IndependentFleetModel):

  transition_matrices: np.ndarray
  chain_states: np.ndarray

  """Available conventional generation model in which generators are assumed to follow a 2-state Markov chain and are statistically independent of each other, and each one can be either 100% or 0% available at any given time.
  """
  
  @property
  def stationary_distributions(self):
    return np.array([self.get_stationary_dist(mat) for mat in self.transition_matrices]) 
  
  @classmethod
  def build_chains(cls,df: pd.DataFrame) -> t.Tuple[np.ndarray, np.ndarray]:
    """Takes a dataframe with generator data (availability and mean time to repair) and returns the corresponding Markov chain states and transition matrices. Here, the availability probability determines the stationary distribution
    
    Args:
        df (pd.DataFrame): Generator data with columns 'availability' and 'mttr'
    
    Returns:
        t.Tuple[np.ndarray, np.ndarray]
    """
    def get_transition_matrix(generator_data):
      prob, mttr = generator_data
      alpha = 1 - 1/mttr
      a11 = 1 - (1-prob)*(1-alpha)/prob
      mat = np.array([[a11,1-a11],[1-alpha,alpha]],dtype=np.float32)
      mat = mat / np.sum(mat,axis=1)
      return mat

    states = np.array([np.array([x,0.0]) for x in df["capacity"]], dtype=np.int32)
    transition_matrices = np.apply_along_axis(get_transition_matrix,1,np.array(df[["availability","mttr"]]))

    return states, transition_matrices

  @classmethod
  def sample_stationary_dists(
    cls, 
    transition_matrices: np.ndarray, 
    chain_states: np.ndarray) -> np.ndarray:
    """Sample states from the stationary distribution of transition probability matrices
    
    Args:
        transition_matrices (np.ndarray): array of transition probability matrices
        chain_states (np.ndarray): two-dimensional array with state vectors for each transition matrix
    
    Returns:
        np.ndarray: array with sampled state for each transition matrix
    """
    sample = []
    for mat,states in zip(transition_matrices, chain_states):
      stationary_dist = cls.get_stationary_dist(mat)
      s = np.random.choice(states,size=1,p=stationary_dist)
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
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    if evec1.shape[1] == 0:
      raise Exception("Some generators might not have a stationary distribution")
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real

    return stationary

  @classmethod
  def from_generator_df(cls, df: pd.DataFrame) -> MarkovChainGenerationModel:
    """Takes a dataframe object and builds the generation model from it.
    
    Args:
        df (pd.DataFrame): dataframe with colums 'availability' and 'capacity', where the former is a probability and the latter the maximum generation capacity; in additon, an 'mttr' column with estimated mean times to repair per unit should be present. Each row represents an individual generator.
    
    Returns:
        MarkovChainGenerationModel: fitted model
    
    """

    time_collapsed = super().from_generator_df(df)

    df["capacity"] = df["capacity"].astype(np.int32) #for consistency between time-collapsed and time-dependent logic

    states, matrices = cls.build_chains(df) 

    return cls(
      pdf_values = time_collapsed.pdf_values,
      support = time_collapsed.support,
      data = time_collapsed.data,
      transition_matrices = matrices,
      chain_states = states)

  @classmethod
  def simulate_chains(
    cls, 
    size: int,
    trace_length: int,
    transition_matrices: np.ndarray,
    chain_states: np.ndarray,
    initial_state: np.ndarray = None,
    simulate_escape_time: bool = True,
    output_array: np.ndarray = None) -> np.Optional[np.ndarray]:
      """Simulate multiple traces in which each one represents the aggregate of multiple Markov chains. This method samples a single large sequential trace and then split it into multiple subtraces; trace endpoints are therefore dependent.
      
      Args:
          size (int): Number of traces to simulate
          trace_length (int): length of individual traces
          transition_matrices (np.ndarray): three-dimensional array containing transition probability matrices for all chains
          chain_states (np.ndarray): two-dimensional array with vectors of chain states for all chains
          initial_state (np.ndarray, optional): one-dimensional array with the initial state indices for all chains. The indices must correspond to a row in the transition matrices. If None, initial states are sampled from the stationary distributions of each chain.
          simulate_escape_time (bool, optional): If True, simulate chains through time-of-escape simulations. If false, simulate each timestep individually.
          output_array (np.ndarray, optional): Array where results are to be stored. If not provided, one is created.
      
      Returns:
          np.Optional[np.ndarray]: two-dimensional array in which each row represent an individual simulated trace. If an output array is passed as input, None is returned.
      
      
      """
      n_chains = len(chain_states)
      n_states = len(chain_states[0])

      # set output array
      if output_array is not None:
        if not isinstance(output_array, np.ndarray) or len(output_array.shape) != 2 or output_array.shape != (size,trace_length) or output_array.dtype != np.float32:
          raise ValueError("output_array must be a two-dimensional numpy array with shape (size, trace_length) of type numpy.float32")
        return_output = False
      else:
        output_array = np.ascontiguousarray(np.zeros((size,trace_length)),dtype=np.float32)
        return_output = True

      c_seed = np.random.randint(low=0,high=2**31-1) #reproducibility for C code based on an external numpy seed

      # set initial state array
      if initial_state is None:
        initial_state = cls.sample_stationary_dists(transition_matrices, chain_states).reshape(-1)
      else:
        if not isinstance(initial_state, np.ndarray) or len(initial_state.shape) != 1 or len(initial_state) != n_chains:
          #print(initial_state)
          #print(initial_state.shape)
          raise ValueError("Initial state vector must be a one-dimensional numpy array with as many entries as transition_matrices.")

      if np.any(np.array([len(s) for s in chain_states]) != n_states):
        raise ValueError("Number of states must be the same for all chains")

      if np.any(np.array([s.shape != (n_states, n_states) for s in transition_matrices])):
        raise ValueError("Matrices must be squared and of the same dimensions")

      # call C program
      if trace_length <= 1:
        raise ValueError("Trace length must be an integer larger than 1")

      # cast as float
      initial_state = np.ascontiguousarray(initial_state).astype(np.float32)
      float_chain_states = chain_states.astype(np.float32)

      C_API.simulate_mc_power_grid_py_interface(
        ffi.cast("float *",output_array.ctypes.data),
        ffi.cast("float *",transition_matrices.ctypes.data),
        ffi.cast("float *",float_chain_states.ctypes.data),
        ffi.cast("float *",initial_state.ctypes.data),
        np.int32(n_chains),
        np.int32(size),
        np.int32(trace_length - 1), #initial state is accounted for in trace length
        np.int32(n_states),
        np.int32(c_seed),
        np.int32(simulate_escape_time))

      if return_output:
        return output_array

  def simulate(self, size: int) -> np.ndarray:
    """Simulate a single trace of sequential observations
    
    Args:
        size (int): Number of samples, or equivalently, trace length.
    
    Returns:
        np.ndarray
    """
    return self.simulate_chains(
      size=1,
      trace_length=size,
      transition_matrices=self.transition_matrices,
      chain_states=self.chain_states,
      initial_state=None,
      simulate_escape_time=True).reshape(-1)

  def simulate_seasons(
    self, 
    size: int, 
    season_length: int,
    seasons_per_trace: int = 1,
    burn_in: int = 100) -> np.ndarray:
    """Simulate multiple traces of available conventional generation; each trace can have one or more peak seasons in it, depending on whether streaks of multiple years need to be sampled.
    
    Args:
        size (int): number of traces to sample
        season_length (int): peak season length
        seasons_per_trace (int, optional): Number of seasons per trace. The default is 1.
        burn_in (int, optional): burn-in period between individual peak season traces; this is needed because in order to sample them, a large sequence is generated and subsequently subdivided, thus making trace endpoints correlated if a burn-in period is not allowed.
    
    Returns:
        np.ndarray: two-dimensional array where each row represent a sampled peak season of available conventional generation.
    """
    total_seasons = size*seasons_per_trace
    augmented_season_length = season_length + burn_in

    output_array = self.simulate_chains(
      size=total_seasons,
      trace_length=augmented_season_length,
      transition_matrices=self.transition_matrices,
      chain_states=self.chain_states,
      initial_state=None,
      simulate_escape_time=True)

    # drop burn in periods and reshape
    return output_array[:,0:season_length].reshape((size, season_length*seasons_per_trace))



