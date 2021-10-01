import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from _c_ext_timedependence import ffi, lib as C_CALL

from scipy.optimize import bisect

import riskmodels.univariate as univar
from riskmodels.powersys.iid.surplus import BaseSurplus
from riskmodels.powersys.convgen import MarkovChainGenerationModel

class UnivariateEmpiricalMapReduce(BaseSurplus, BaseModel):

  class MarkovChainGenerationTraces(BaseModel):

    traces: np.ndarray

    @classmethod
    def from_file(cls, trace_filepath: str):
      cls(np.load(trace_filepath, allow_pickle=True))

    def __add__(self, other):
      return type(self)(self.samples + other)

    def __mul__(self, other):
      return type(self)(self.samples * other)

  class UnivariateEmpiricalTraces(BaseSurplus, BaseModel):

    gen_filepath: str
    demand: np.ndarray
    renewables: np.ndarray
    season_length: t.Optional[Int]

    @validator("season_length")
    def season_length_validator(cls, season_length):
      if season_length is None:
        return len(self.demand)

    @property
    def surplus_trace(self):
      return MarkovChainGenerationTraces.from_file(self.gen_filepath).traces - (self.demand - self.renewables)

    def cdf(self, x: float):
      trace = self.surplus_trace
      return np.mean(trace <= x, axis=1), len(trace)

    def simulate(self):
      return self.surplus_trace

    def lole(self):
      cdf_value, n = self.cdf(0.0)
      return self.season_length * cdf_value, n

    def eeu(self):
      trace = self.surplus_trace
      return self.season_length * np.mean(np.maximum(0.0, -trace), axis=1), len(trace)

    def get_shortfall_data(self):
      df = pd.DataFrame({"surplus": self.surplus_trace.reshape(-1)})
      df["time"] = np.arange(len(df.index))
      df["file_id"] = Path(self.gen_filepath).name
      # filter by shortfall
      df = df.query("surplus < 0")
      # add season features
      df["season_time"] = df["time"]%self.season_length
      df["season"] = (df["time"]/self.season_length).astype(np.int32)
      return df


  gen_dir: str
  demand: np.ndarray
  renewables: np.ndarray
  season_length: int

  @classmethod
  def create_gen_trace_files(
    cls,
    output_dir: str,
    n_traces: int,
    max_traces_per_file: int,
    gen: MarkovChainGenerationModel,
    demand: np.ndarray,
    renewables: np.ndarray,
    season_length: int,
    n_threads: int = 4,
    burn_in: int = 100):

    def persist_gen_traces(args):
      gen, call_args, filename = args
      traces = gen.simulate_seasons(*call_args)
      np.save(filename, traces)

    if len(demand) != len(renewables) or len(demand)%season_length!=0:
      raise ValueError("Demand and renewables arrays must have same length, and the length must be divisible by season_length.")

    # compute size (in terms of number of traces) and number of files
    n_files = int(np.ceil(n_traces/max_traces_per_file))
    file_sizes = [max_traces_per_file for k in range(n_files)]
    remainder = n_traces%max_traces_per_file
    if remainder > 0:
      file_sizes[-1] = remainder

    # create argument list for multithreaded execution
    arglist = []
    seasons_per_trace = len(demand)/season_length
    for k, file_size in enumerate(file_sizes):
      output_path = Path(output_dir) / str(k)
      call_args = (file_size, season_length, seasons_per_trace, burn_in)
      arglist.append((gen, call_args, output_path))

    #create files in parallel
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
      jobs = executor.map(persist_gen_traces, arglist)
    concurrent.futures.wait(executor, timeout=None, return_when=concurrent.futures.ALL_COMPLETED)
    t1 = time.time()
    print(f"{np.round(t1-t0,2)} seconds ellapsed.")


  def map_reduce(
    self, 
    mapper: t.Union[str. t.Callable], 
    reducer: t.Optional[t.Callable], 
    str_map_args: t.Tuple = None):

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
      arglist.append(((str(file), self.demand, self.renewables, self.season_length), map_func, str_map_args))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
      mapped = executor.map(execute_map, arglist)
    concurrent.futures.wait(executor, timeout=None, return_when=concurrent.futures.ALL_COMPLETED)
    
    if reducer is not None:
      return reducer(mapped)
    else:
      return mapped

  def cdf(self, x: float):
    def reducer(mapped):
      n_traces = np.sum([n for _, n in mapped])
      return np.array([n*val for val, n in mapped])/n_traces

    return self.map_reduce(mapper="cdf", reducer=reducer, str_map_args=(x,))

  def simulate(self):
    raise NotImplementedError("This class does not implement a simulate() method.")

  def lole(self):
    self.season_length * self.cdf(0.0)

  def eeu(self):
    def reducer(mapped):
      n_traces = np.sum([n for _, n in mapped])
      return np.array([n*val for val, n in mapped])/n_traces
    self.map_reduce(mapper="eeu", reducer=reducer)

  def get_shortfall_data(self):

    def reducer(mapped):
      return pd.concat(mapped)

    return self.map_reduce(mapper="get_shortfall_data",reducer=reducer)
