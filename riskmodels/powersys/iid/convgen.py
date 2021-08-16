from riskmodels.base.models import Integer

import numpy as np
import pndas as pd

class IndependentFleetModel(Integer):

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
    logger.info("Coercing capacity values to integers")
    
    capacity_values = np.array(data["capacity"], dtype=np.int32)
    availability_values = np.array(data["availability"])

    if np.any(availabilities < 0) or np.any(availabilities > 1):
      raise Exception("Availabilities ,ust between 0 and 1")

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
    return cls.from_data(support, f)

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

    
    
