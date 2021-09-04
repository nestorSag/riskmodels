from __future__ import annotations
import warnings

from riskmodels.base.marginals import Binned

import numpy as np
import pandas as pd

class IndependentFleetModel(Binned):

  """Available conventional generation model in which generators are assumed statistically independent of each other, and each one can be either 100% or 0% available at any given time.
  """
  
  @classmethod
  def from_generator_df(cls, df: pd.DataFrame) -> IndependentFleetModel:
    """Takes a dataframe object and builds the generation model from it.
    
    Args:
        df (pd.DataFrame): dataframe with colums 'availability' and 'capacity', where the former is a probability and the latter the maximum capacity; each row represents an individual generator
    
    Returns:
        IndependentFleetModel: fitted model
    
    """
    warnings.warn("Coercing capacity values to integers")
    
    capacity_values = np.array(df["capacity"], dtype=np.int32)
    availability_values = np.array(df["availability"])

    if np.any(availability_values < 0) or np.any(availability_values > 1):
      raise Exception(f"Availabilities must be in the interval [0,1]; found interval[{min(availability_values)},{max(availability_values)}]")

    max_gen = int(np.sum(capacity_values[capacity_values>=0]))
    min_gen = int(np.sum(capacity_values[capacity_values<0]))

    zero_idx = np.abs(min_gen) #this is in case there are generators with negative generation
    f_length = max_gen+1 - min_gen
    f = np.zeros((f_length,),dtype=np.float64)
    f[zero_idx] = 1.0

    i = 0
    for c,p in zip(capacity_values,availability_values):
      if c >= 0:
        suffix = f[0:f_length-c]
        preffix = np.zeros((c,))
      else:
        preffix = f[np.abs(c):f_length]
        suffix = np.zeros((np.abs(c),))
      f = (1-p) * f + p * np.concatenate([preffix,suffix])
      i += 1

    support = np.arange(min_gen, max_gen+1)
    return Binned(support=support, pdf_values=f, data=None)

  @classmethod
  def from_generator_csv_file(cls, file_path: str, **kwargs) -> IndependentFleetModel:
    """Takes a csv file and builds the generation model
    
    Args:
        file_path (str): Path to csv file. It must have colums 'availability' and 'capacity', where the former is a probability and the latter the maximum capacity; each row represents an individual generator
        **kwargs: additional arguments passed to pandas.read_csv
    
    Returns:
        IndependentFleetModel: fitted model
    """
    df = pd.read_csv(gen_data,**kwargs)
    return cls.from_generator_df(df)

    
    
