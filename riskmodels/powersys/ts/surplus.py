"""
This module implements surplus distribution models for sequential conventional generation. Because Monte Carlo estimation is the only way to compute statistical properties of time series models for power surpluses, these models rely heavily on large-scale simulation and computation. The classes of this module implement multi-core processing following a map-reduce pattern on a large number of simulated traces that have been persisted in multiple files.
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
import typing as t

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

from c_sequential_models_api import ffi, lib as C_CALL

from scipy.optimize import bisect

import riskmodels.univariate as univar
from riskmodels.powersys.iid.surplus import BaseSurplus
from riskmodels.powersys.ts.convgen import MarkovChainGenerationModel

from tqdm import tqdm


class MarkovChainGenerationTraces(BaseModel):

  """Wrapper class for persisted available conventional generation traces
  """

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

  def __add__(self, other):
    return type(self)(traces=self.samples + other)

  def __mul__(self, other):
    return type(self)(traces=self.samples * other)

class UnivariateEmpiricalTraces(BaseSurplus, BaseModel):

  """Wrapper class for the workers of map-reduce computations; they use a file-based sequence of conventional generation traces in order to perform computations. 
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
    return MarkovChainGenerationTraces.from_file(self.gen_filepath).traces - (self.demand - self.renewables)

  def cdf(self, x: float) -> t.Tuple[float, int]:
    """Evaluates the surplus distribution's CDF. Also returns the number of seasons used to calculate it.
    
    Args:
        x (float): Description
    
    Returns:
        t.Tuple[float, int]: A tuple with the estimated value and the number of seasons used to calculate it.
    """
    trace = self.surplus_trace
    return np.mean(trace < 0), len(trace)

  def simulate(self) -> float:
    return self.surplus_trace

  def lole(self) -> t.Tuple[float, int]:
    """Evaluates the distribution's season-wise LOLE. Also returns the number of seasons used to calculate it.
    
    Returns:
        t.Tuple[float, int]: A tuple with the estimated value and the number of seasons used to calculate it.
    """

    # cdf_value, n = self.cdf(0.0)
    # return self.season_length * cdf_value, n
    trace = self.surplus_trace

    n_traces, trace_length = trace.shape
    if trace_length%self.season_length != 0:
      raise ValueError("Trace length is not a multiple of season length.")
    seasons_per_trace = int(trace_length/self.season_length)
    n_total_seasons = n_traces*seasons_per_trace

    return np.sum(trace < 0)/n_total_seasons, n_traces

  def eeu(self) -> t.Tuple[float, int]:
    """Evaluates the distribution's season-wise expected energy unserved. Also returns the number of seasons used to calculate it.
    
    
    Returns:
        t.Tuple[float, int]: A tuple with the estimate value and the number of seasons used to calculate it.
    """

    trace = self.surplus_trace
    
    n_traces, trace_length = trace.shape
    if trace_length%self.season_length != 0:
      raise ValueError("Trace length is not a multiple of season length.")
    seasons_per_trace = int(trace_length/self.season_length)
    n_total_seasons = n_traces*seasons_per_trace

    return np.sum(np.maximum(0.0, -trace))/n_total_seasons, n_traces

  def get_surplus_df(self, shorfalls_only: bool = True) -> pd.DataFrame:
    """Returns a data frame with time occurrence information of observed surplus values and shortfalls.
    
    Args:
        shortfalls_only (bool, optional): If True, only shortfall rows are returned

    Returns:
        pd.DataFrame: A data frame with the surplus values, a 'season_time' column with the within-season time of occurrence (0,1,...,season_length-1), a 'file_id' column that indicates which file was used to compute the value, and a 'season' column to indicate which season the value was observed in.
    
    """
    trace = self.surplus_trace
    df = pd.DataFrame({"surplus": trace.reshape(-1)})
    df["time"] = np.arange(len(df))
    df["file_id"] = Path(self.gen_filepath).name
    # filter by shortfall
    if shorfalls_only:
      df = df.query("surplus < 0")
    # add season features
    df["season_time"] = df["time"]%self.season_length
    df["season"] = (df["time"]/self.season_length).astype(np.int32)
    df = df.drop(columns=["time"])
    return df, len(trace)


class UnivariateEmpiricalMapReduce(BaseSurplus, BaseModel):

  """Univariate model for power surplus using a sequential available conventional generation model
  """
  
  gen_dir: str
  demand: np.ndarray
  renewables: np.ndarray
  season_length: int

  class Config:
    arbitrary_types_allowed = True

  @classmethod
  def init(
    cls,
    output_dir: str,
    n_traces: int,
    n_files: int,
    gen: MarkovChainGenerationModel,
    demand: np.ndarray,
    renewables: np.ndarray,
    season_length: int,
    n_threads: int = 4,
    burn_in: int = 100) -> UnivariateEmpiricalMapReduce:
    """Generate and persists traces of conventional generation in files, and use them to instantiate a surplus model.
    
    Args:
        output_dir (str): Output directory for trace files
        n_traces (int): Total number of season traces to simulate
        n_files (int): Number of files to create. Making this a multiple of the available number of cores and ensuring that each file is on the order of 500 MB (~ 125 million floats) is probably optimal.
        gen (MarkovChainGenerationModel): Sequential conventional generation instance.
        demand (np.ndarray): Demand data
        renwables (np.ndarray): Renewables data
        season_length (int): Peak season length. 
        n_threads (int, optional): Number of threads to use.
        burn_in (int, optional): Parameter passed to MarkovChainGenerationModel.simulate_seasons.
    
    Returns:
        UnivariateEmpiricalMapReduce: Sequential surplus model
    
    
    """
    def persist_gen_traces(args):
      gen, call_args, filename = args
      traces = gen.simulate_seasons(*call_args)
      np.save(filename, traces)

    # create dir if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if len(demand) != len(renewables):
      raise ValueError("demand and renewables must have the same length.")
    
    trace_length = len(demand)

    if trace_length%season_length!=0:
      raise ValueError("trace_length must be divisible by season_length.")

    if n_traces <= 0 or not isinstance(n_traces, int):
      raise ValueError("n_traces must be a positive integer")

    if n_files <= 0 or not isinstance(n_files, int):
      raise ValueError("n_files must be a positive integer")

    # compute file size (in terms of number of traces)
    file_sizes = [int(n_traces/n_files) for k in range(n_files)]
    file_sizes[-1] += n_traces - sum(file_sizes)

    # create argument list for multithreaded execution
    arglist = []
    seasons_per_trace = int(trace_length/season_length)
    for k, file_size in enumerate(file_sizes):
      output_path = Path(output_dir) / str(k)
      call_args = (file_size, season_length, seasons_per_trace, burn_in)
      arglist.append((gen, call_args, output_path))

    #create files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
      jobs = list(tqdm(executor.map(persist_gen_traces, arglist), total=len(arglist)))
    
    return cls(
      gen_dir=output_dir,
      demand=np.array(demand), 
      renewables=np.array(renewables), 
      season_length=season_length)


  def map_reduce(
    self, 
    mapper: t.Union[str, t.Callable], 
    reducer: t.Optional[t.Callable], 
    str_map_args: t.Tuple = tuple(),
    n_threads: int = 4) -> t.Any:
    """Performs map-reduce processing operations on each persisted generation trace file, given mapper and reducer functions
    
    Args:
        mapper (t.Union[str.t.Callable]): If a string, the method of that name is called on each UnivariateEmpiricalTraces worker instance. If a function, it must take as only argument an instance of UnivariateEmpiricalTraces.
        reducer (t.Optional[t.Callable]): This function must take a list of mapper outputs as only argument. If None, no reducer is applied.
        str_map_args (t.Tuple, optional): Unnamed arguments passed to the mapper function when it is passed as a string.
        n_threads (int, optional): Number of threads to use.
    
    """
    def execute_map(call_args):
      kwargs, map_func, str_map_args = call_args
      surplus = UnivariateEmpiricalTraces(**kwargs)
      if isinstance(map_func, str):
        return getattr(surplus, map_func)(*str_map_args)
      elif isinstance(map_func, t.Callable):
        return map_func(surplus)
      else:
        raise ValueError("map_func must be a string or a function.")

    arglist = []
    # create arglist for parallel execution
    for file in Path(self.gen_dir).iterdir():
      kwargs = {"gen_filepath": str(file), "demand": self.demand, "renewables": self.renewables, "season_length": self.season_length}
      arglist.append((kwargs, mapper, str_map_args))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
      mapped = list(tqdm(executor.map(execute_map, arglist), total=len(arglist)))
    
    if reducer is not None:
      return reducer(mapped)
    else:
      return mapped

  def cdf(self, x: float):
    def reducer(mapped):
      n_traces = np.sum([n for _, n in mapped])
      return np.array([n*val for val, n in mapped]).sum()/n_traces

    return self.map_reduce(mapper="cdf", reducer=reducer, str_map_args=(x,))

  def simulate(self):
    raise NotImplementedError("This class does not implement a simulate() method. Use get_surplus_df() to get the shortfalls or the full sequence of surplus values.")

  def lole(self):
    return self.season_length * self.cdf(0.0)

  def eeu(self):
    def reducer(mapped):
      n_traces = np.sum([n for _, n in mapped])
      return np.array([n*val for val, n in mapped]).sum()/n_traces
    return self.map_reduce(mapper="eeu", reducer=reducer)

  def get_surplus_df(self, shorfalls_only: bool = True) -> pd.DataFrame:
    """Returns a data frame with time occurrence information of observed surplus values and shortfalls.
      
      Args:
          shortfalls_only (bool, optional): If True, only shortfall rows are returned

      Returns:
          pd.DataFrame: A data frame with the surplus values, a 'season_time' column with the within-season time of occurrence (0,1,...,season_length-1), a 'file_id' column that indicates which file was used to compute the value, and a 'season' column to indicate which season the value was observed in.
      
      """

    def reducer(mapped):
      return pd.concat([df for df, n in mapped])

    return self.map_reduce(mapper="get_surplus_df",reducer=reducer, str_map_args=(shorfalls_only,))

  def __str__(self):
    return f"Map-reduce based sequential surplus model using trace files in {self.gen_dir}"
