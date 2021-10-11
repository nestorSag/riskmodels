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
from riskmodels.powersys.iid.surplus import BaseSurplus, BaseBivariateMonteCarlo
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


class BivariateEmpiricalTraces(BaseBivariateMonteCarlo):

  """Wrapper class for the workers of map-reduce computations; they use a file-based sequence of conventional generation traces in order to perform computations. This class takes advantage of riskmodels.powersys.iid.surplus.BaseBivariateMonteCarlo to avoid repeating code, and implements both veto and share policies.

  Args:
      univariate_traces (t.List[UnivariateEmpiricalTraces]): Univariate traces
      policy (str): Either 'veto' or 'share'
  """

  univariate_traces: t.List[UnivariateEmpiricalTraces]
  policy: str

  class Config:
    arbitrary_types_allowed = True

  @property
  def surplus_trace(self):
    """This returns the traces as a 3-dimensional array where the axes correspond to area, trace number and peak season time step respectively
    
    """
    return return np.array([t.surplus_trace for t in univariate_traces])

  def get_pre_itc_sample(self) -> np.ndarray:
    """Returns a pre-interconnection surplus sample as a two-dimensional array where realisations of different peak seasons have been concatenated for each area.
    
    Returns:
        np.ndarray: Sample
    """
    return np.concatente([t.surplus_trace.reshape((-1,1)) for t in univariate_traces], axis=1)

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
      flow = np.zeros((len(sample, )), dtype=np.float32)
      # split individual surplus traces
      s1, s2 = sample[:,0], sample[:,1]
      # market-driven shortfall-sharing conditions from a share policy only kick in when the following happens; in all other situations, the policy is identical to veto.
      # briefly, this is because of interconnector constraints
      share_cond = np.logical_and(s1 +s2 < 0, s1 < itc_cap, s2 < itc_cap)
      # market-driven flows are determined by both demand and surpluses; tile demand vector to perform flow calculations
      k = len(sample)/len(univariate_traces[0].demand) #tiling factor
      if k - int(k) != 0:
        raise ValueError("Length of surplus samples is not a multiple of demand array length.")
      k = int(k)
      # tiled demand ratio
      r = np.tile(univariate_traces[0].demand/(univariate_traces[0].demand + univariate_traces[1].demand), k)
      # compute share flow when applicable
      flow[share_cond] = np.minimum(itc_cap, np.maximum(-itc_cap, r[share_cond]*s2[share_cond] - (1-r[share_cond])*s1[share_cond]))
      # compute veto flow for all other entries
      flow[np.logical_not(share_cond)] = super().itc_flow(sample[np.logical_not(share_cond)], itc_cap)
      return flow
    else:
      raise ValueError("policy must be either 'veto' or 'share'")

  def get_surplus_df(self, shorfalls_only: bool = True, itc_cap: int = 1000) -> pd.DataFrame:
    """Returns a data frame with time occurrence information of observed post-interconnection surplus values and shortfalls.
    
    Args:
        shorfalls_only (bool, optional): Whether to return only rows corresponding to shortfalls.
        itc_cap (int, optional): Interconnector policy
    
    Returns:
        pd.DataFrame: A data frame with the surplus values, a 'season_time' column with the within-season time of occurrence (0,1,...,season_length-1), a 'file_id' column that indicates which file was used to compute the value, and a 'season' column to indicate which season the value was observed in.
    
    """
    trace = self.simulate(itc_cap)
    df = pd.DataFrame(trace, columns = ["surplus1", "surplus2"])
    df["time"] = np.arange(len(df))
    df["file_id"] = Path(self.gen_filepath).name
    # filter by shortfall
    if shorfalls_only:
      df = df.query("surplus1 < 0 or surplus2 < 0")
    # add season features
    df["season_time"] = df["time"]%self.season_length
    df["season"] = (df["time"]/self.season_length).astype(np.int32)
    df = df.drop(columns=["time"])
    return df, len(trace)


class UnivariateEmpiricalMapReduce(BaseSurplus, BaseModel):

  """Univariate model for power surplus using a sequential available conventional generation model, implementing Monte Carlo evaluations through map-reduce patterns. Worker instances are of type UnivariateEmpiricalTraces.
  """
  
  gen_dir: str
  demand: np.ndarray
  renewables: np.ndarray
  season_length: int

  _worker_class = UnivariateEmpiricalTraces

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


  def create_mapred_arglist(self, mapper: t.Union[str, t.Callable], str_map_kwargs: t.Dict) -> t.List[t.Dict]:
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
      kwargs = {"gen_filepath": str(file), "demand": self.demand, "renewables": self.renewables, "season_length": self.season_length}
      arglist.append((kwargs, mapper, str_map_kwargs))

    return arglist

  @classmethod
  def execute_map(cls, call_args: t.Tuple[t.Dict, t.Union[str, t.Callable], t.Tuple]) -> t.Any:
    """Instantiate a worker and execute mapper function on it.
    
    Args:
        call_args (t.Tuple[t.Dict, t.Union[str, t.Callable], t.Tuple]): A triplet with named arguments to instantiate the workers, the function to call on instantiated workers as a string or callable object, and additional unnamed arguments passed to the mapper if given as a string.
    
    Returns:
        t.Any: Mapper output
    
    """
    worker_kwargs, map_func, str_map_kwargs = call_args
    surplus = cls._worker_class(**worker_kwargs)
    if isinstance(map_func, str):
      return getattr(surplus, map_func)(**str_map_kwargs)
    elif isinstance(map_func, t.Callable):
      return map_func(surplus)
    else:
      raise ValueError("map_func must be a string or a function.")

  def map_reduce(
    self, 
    mapper: t.Union[str, t.Callable], 
    reducer: t.Optional[t.Callable], 
    str_map_kwargs: t.Dict = {},
    n_threads: int = 4) -> t.Any:
    """Performs map-reduce processing operations on each persisted generation trace file, given mapper and reducer functions
    
    Args:
        mapper (t.Union[str, t.Callable]): If a string, the method of that name is called on each worker instance. If a function, it must take as only argument a worker instance.
        reducer (t.Optional[t.Callable]): This function must take a list of mapper outputs as only argument. If None, no reducer is applied.
        str_map_kwargs (t.Dict, optional): Named arguments passed to the mapper function when it is passed as a string.
        n_threads (int, optional): Number of threads to use.
    
    Returns:
        t.Any: Map-reduce output
    
    """

    arglist = self.create_mapred_arglist(mapper, str_map_kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
      mapped = list(tqdm(executor.map(self.execute_map, arglist), total=len(arglist)))
    
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
      return np.array([n*val for val, n in mapped]).sum()/n_traces

    return self.map_reduce(mapper="cdf", reducer=reducer, str_map_args={"x": x})

  def simulate(self):
    raise NotImplementedError("This class does not implement a simulate() method. Use get_surplus_df() to get the shortfalls or the full sequence of surplus values.")

  def lole(self) -> float:
    """Computes the loss of load expectation
    
    Returns:
        float: lole estimate
    """
    return self.season_length * self.cdf(-1e-1) #tiny offset

  def eeu(self):
    """Computes the expected energy unserved
    
    Returns:
        float: eeu estimate
    """
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

    return self.map_reduce(
      mapper="get_surplus_df",
      reducer=reducer, 
      str_map_kwargs={"shortfalls_only": shortfalls_only})

  def __str__(self):
    return f"Map-reduce based sequential surplus model using trace files in {self.gen_dir}"



class BivariateEmpiricalMapReduce(UnivariateEmpiricalMapReduce):

  """Bivariate model for power surplus using a sequential available conventional generation model, implementing Monte Carlo evaluations through map-reduce patterns. Worker instances are of type BivariateEmpiricalTraces.
  """
  
  gen_dir: t.List[str]
  demand: np.ndarray
  renewables: np.ndarray
  season_length: int

  _worker_class = BivariateEmpiricalTraces

  class Config:
    arbitrary_types_allowed = True

  @classmethod
  def init(
    cls,
    output_dirs: t.List[str],
    n_traces: int,
    n_files: int,
    gens: t.List[MarkovChainGenerationModel],
    demand: np.ndarray,
    renewables: np.ndarray,
    season_length: int,
    n_threads: int = 4,
    burn_in: int = 100) -> UnivariateEmpiricalMapReduce:
    """Generate and persists traces of conventional generation in files, and use them to instantiate a surplus model.
    
    Args:
        output_dirs (str): List of output directories for trace files, with an entry per system.
        n_traces (int): Total number of season traces to simulate
        n_files (int): Number of files to create. Making this a multiple of the available number of cores and ensuring that each file is on the order of 500 MB (~ 125 million floats) is probably optimal.
        gens (MarkovChainGenerationModel): List of sequential conventional generation instances, one per system.
        demand (np.ndarray): Demand data
        renewables (np.ndarray): Renewables data
        season_length (int): Peak season length. 
        n_threads (int, optional): Number of threads to use.
        burn_in (int, optional): Parameter passed to MarkovChainGenerationModel.simulate_seasons.
    
    Returns:
        BivariateEmpiricalMapReduce: Sequential surplus model
    
    
    """
    area = 0
    for out_dir, gen, univar_demand, univar_renewables in zip(output_dirs, gens, demand.T, renewables.T):
      print(f"Creating files for area {area}..")
      UnivariateEmpiricalMapReduce.init(
        output_dir = out_dir,
        n_traces = n_traces,
        n_files = n_files,
        gen = gen,
        demand = univar_demand,
        renewables = univar_renewables,
        season_length = season_length,
        n_threads = n_threads,
        burn_in = burn_in)
      area += 1

    return cls(
      gen_dirs=output_dirs,
      demand=np.array(demand), 
      renewables=np.array(renewables), 
      season_length=season_length)

  def create_mapred_arglist(self, mapper: t.Union[str, t.Callable], str_map_kwargs: t.Dict, policy: str) -> t.List[t.Dict]:

    arglist = []
    
    dir1, dir2 = self.gen_dirs

    # for each pair of files, create a list with two instances of univariate surplus models
    for file1, file2 in zip(Path(dir1).iterdir(), Path(dir2).iterdir()):
      files = [file1, file2]
      univariate_traces = []
      # create each univariate surplus model
      for file, demand, renewables in zip(files, self.demand.T, self.renewables.T):
        univar_args = {
          "gen_filepath": str(file), 
          "demand": demand, 
          "renewables": renewables, 
          "season_length": self.season_length
        }

        univariate_traces.append(UnivariateEmpiricalTraces(**univar_args))

      # policy is passed here at worker instantiation level. This is to take advantage of bivariate surplus code from iid module.
      bivariate_args = {
        "univariate_traces": univariate_traces, 
        "season_length": self.season_length. 
        "policy": policy
      }
      arglist.append(bivariate_args, mapper, str_map_kwargs)
    return arglist

  def map_reduce(
    self, 
    mapper: t.Union[str, t.Callable], 
    reducer: t.Optional[t.Callable], 
    str_map_kwargs: t.Tuple = {},
    policy: str = "veto",
    itc_cap: float = 1000.0,
    n_threads: int = 4) -> t.Any:
    """Performs map-reduce processing operations on each persisted generation trace file, given mapper and reducer functions
    
    Args:
        mapper (t.Union[str, t.Callable]): If a string, the method of that name is called on each worker instance. If a function, it must take as only argument a worker instance.
        reducer (t.Optional[t.Callable]): This function must take a list of mapper outputs as only argument. If None, no reducer is applied.
        str_map_args (t.Tuple, optional): Unnamed arguments passed to the mapper function when it is passed as a string.
        n_threads (int, optional): Number of threads to use.
    
    """

    # itc_cap is passed at the mapper function level as an extra argument; overwrite if necessary
    str_map_kwargs["itc_cap"] = itc_cap
    arglist = self.create_mapred_arglist(mapper, str_map_kwargs, policy)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
      mapped = list(tqdm(executor.map(self.execute_map, arglist), total=len(arglist)))
    
    if reducer is not None:
      return reducer(mapped)
    else:
      return mapped

  def cdf(self, x: np.ndarray, itc_cap: float = 1000.0, policy = "veto"):
    """Evaluates the bivariate post-interconnection power surplus distribution's cumulative distribution function
    
    Args:
        x (np.ndarray): value at which to evaluate the cdf
        itc_cap (int, optional): interconnection capacity
        policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.

    """
    def reducer(mapped):
      n_traces = np.sum([n for _, n in mapped])
      return np.array([n*val for val, n in mapped]).sum()/n_traces

    return self.map_reduce(mapper="cdf", reducer=reducer, itc_cap = itc_cap, policy=policy)

  def simulate(self):
    raise NotImplementedError("This class does not implement a simulate() method. Use get_surplus_df() to get the shortfalls or the full sequence of surplus values.")

  def lole(self, itc_cap: float = 1000.0, policy = "veto", area: int = 0):
    """Computes the post-interconnection loss of load expectation.
    
    Args:
        itc_cap (int, optional): interconnection capacity
        policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.
        area (int, optional): Area for which to evaluate LOLE; if area=-1, system-wide lole is returned
    """
    x = np.zeros((2,), dtype=np.float32) - 1e-1 #tiny offset
    x[1-area] = np.Inf
    return self.season_length * self.cdf(x, itc_cap = itc_cap, policy=policy, str_map_kwargs={"area": area})

  def eeu(self itc_cap: float = 1000.0, policy = "veto", area: int = 0):
    """Computes the post-interconnection expected energy unserved.
    
    Args:
        itc_cap (int, optional): interconnection capacity
        policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.
        area (int, optional): Area for which to evaluate eeu; if area=-1, systemwide eeu is returned
    """

    def reducer(mapped):
      n_traces = np.sum([n for _, n in mapped])
      return np.array([n*val for val, n in mapped]).sum()/n_traces

    return self.map_reduce(
      mapper="eeu", 
      reducer=reducer, 
      itc_cap = itc_cap, 
      policy=policy, 
      str_map_kwargs={"area": area})

  def get_surplus_df(self, shorfalls_only: bool = True) -> pd.DataFrame:

    def reducer(mapped):
      return pd.concat([df for df, n in mapped])

    return self.map_reduce(
      mapper="get_surplus_df",
      reducer=reducer, 
      str_map_kwargs={"shortfalls_only": shortfalls_only}, 
      itc_cap = itc_cap, 
      policy=policy)

  def __str__(self):
    return f"Map-reduce based sequential surplus model using trace files in {self.gen_dir}"
