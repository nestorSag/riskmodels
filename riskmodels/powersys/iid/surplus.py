"""This module implements models to calculate risk metrics relevant to energy procurement for the case of an interconnected 2-area system, specifically loss of load expectation (LOLE) and expected energy unserved (EEU). As these models assume a time-collapsed setting in which serial dependence does not exist, only metrics based on expected values like the above can be validly calculated. Exact calculations for empirical demand and renewable models are available for both veto and share policies (see below), and Monte Carlo estimation for arbitrary net demand models are available for a veto policy only.

In a share policy, power flow through the interconnection is driven by market prices, even in the event of a shortfall; this can create situations in which shortfalls spread to other areas by excessive imports or exports. In a veto policy on the other hand, only spare available generation can flow through the interconnector, and areas never divert generation they are already using somewhere else. 
"""

from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
import copy

from riskmodels.bivariate import Independent, BaseDistribution
import riskmodels.univariate as univar

import numpy as np
import pandas as pd

from c_bivariate_surplus_api import ffi, lib as C_API

from pydantic import BaseModel

class BaseSurplus(ABC):

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
   

class BaseBivariateMonteCarlo(BaseModel, BaseSurplus):

  """Base class for calculating time-collapsed risk indices for bivariate power surplus distributions using Monte Carlo. Implements calculations based on an assumed surplus trace, but crucially leaves unimplemented the method to compute this surplus trace; subclasses inheriting from this one can implement different ways to calculate this, such as simulation from bivariate distribution objects or loading traces from a file in the case of sequential Monte Carlo models. Only veto policies are implemented.
  
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
    flow = np.zeros((len(sample, )), dtype=np.float32)

    if itc_cap == 0:
      return flow

    flow_from_area_1_idx = np.logical_and(sample[:,0] > 0, sample[:,1] < 0)
    flow_to_area_1_idx = np.logical_and(sample[:,0] < 0, sample[:,1] > 0)

    # flows are bounded by interconnection capacity, shortfall size and spare available capacity in each area.
    flow[flow_from_area_1_idx] = -np.minimum(itc_cap, np.minimum(sample[:,0][flow_from_area_1_idx], -sample[:,1][flow_from_area_1_idx]))
    flow[flow_to_area_1_idx] = np.minimum(itc_cap, np.minimum(-sample[:,0][flow_to_area_1_idx], sample[:,1][flow_to_area_1_idx]))

    return flow

  def simulate(self, itc_cap: int = 1000):
    """Simulate from post-interconnection surplus distribution
    
    Args:
        itc_cap (int, optional): Interconnection capacity
    
    """
    pre_itc_sample = self.get_pre_itc_sample()
    flow = self.itc_flow(pre_itc_sample, itc_cap)
    # add flow to pre itc sample
    pre_itc_sample[:,0] += flow
    pre_itc_sample[:,1] -= flow
    return pre_itc_sample

  def cdf(self, x: np.ndarray, itc_cap: int = 1000):
    """Estimate the CDF of bivariate post-interconnection surplus evaluated at x
    
    Args:
        x (np.ndarray): point to be evaluated
        itc_cap (int, optional): interconnection capacity
    
    """
    samples = self.simulate(itc_cap)
    u = samples <= x.reshape((1,2)) # componentwise comparison
    v = u.dot(np.ones((2,1))) >= 2 #equals 1 if and only if both components fulfill the above condition
    return np.mean(v) #return empirical CDF estimate

  def lole(self, itc_cap: int = 1000, area: int = 0):
    """Calculates loss of load expectation for one of the areas in the system    
    Args:
        itc_cap (int, optional): Interconnection capacity
        area (int, optional): Area index (0 or 1); if area=-1, systemwide lole is returned.

    """
    # take as loss of load when shortfalls are at least 0.1MW in size; this induces a negligible amount of bias but solves numerical issues when comparing post-itc surpluses to 0 to flag shortfalls.
    x = np.array([-1e-1,-1e-1], dtype=np.float32)
    if area in [0,1]:
      x[1-area] = np.Inf
      return self.season_length * self.cdf(x, itc_cap)
    elif area == -1:
      return self.season_length * (self.cdf(np.array([np.Inf,0]), itc_cap) + self.cdf(np.array([0,np.Inf]), itc_cap) - self.cdf(np.array([0,0]), itc_cap))
    else:
      raise ValueError("area must be in [-1,0,1]")

  def eeu(self, itc_cap: int = 1000, area: int = 0):
    """Calculates expected energy unserved for one of the areas in the system
    
    Args:
        itc_cap (int, optional): Interconnection capacity
        area (int, optional): Area index (0 or 1).

    """
    samples = self.simulate(itc_cap)
    if area in [0,1]:
      return -self.season_length * np.mean(np.minimum(samples[:,area], 0))
    elif area == -1:
      return -self.season_length * (np.mean(np.minimum(samples[:,0], 0)) + np.mean(np.minimum(samples[:,1], 0)))
    else:
      raise ValueError("area must be in [-1,0,1]")

  @abstractmethod
  def get_pre_itc_sample(self) -> np.ndarray:
    """Returns a pre-interconnection surplus sample
    
    Returns:
        np.ndarray: Sample
    """
    pass



class BivariateMonteCarlo(BaseBivariateMonteCarlo):

  """General bivariate power surplus distribution formed by a power generation distribution and a net demand distribution. It calculates risk metrics by simulation and only implements a veto policy between areas, this is, areas will only export spare available capacity. 
  
  Args:
      gen_distribution (BaseDistribution): available conventional generation distribution
      net_demand (BaseDistribution): net demand distribution
      size (BaseDistribution): Sample size for Monte Carlo estimation
      
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
    return self.gen_distribution.simulate(self.size) - self.net_demand.simulate(self.size)



class BivariateEmpirical(BaseSurplus):

  """Computes statistics for the power surpluses in a 2-area power system with a single interconnector, given the distributions of available conventional generation and data for demand and renewable generation in the two areas; this uses the empirical distributions induced by the data. The interconnector is assumed to always work. This class interfaces to C for performance.
  """
  
  def __repr__(self):
    return f"Bivariate empirical surplus model with {len(self.demand_data)} observations"
  def __init__(
    self,
    demand_data: np.ndarray,
    renewables_data: np.ndarray,
    gen_distribution: Independent,
    season_length: int = None):
    """
    
    Args:
      demand_data (np.ndarray): Demand data matrix with two columns
      gen_distribution (Independent): A bivariate distribution with independent components, where each component is a univar.Binned instance representing the distribution of available conventional generation for the corresponding area
      renewables_data (np.ndarray): Renewable generation data matrix with two columns
      season_length (int, optional): length of peak season. If None, it is set as the length of demand data
    
    """
    warnings.warn("Coercing data to integer values.", stacklevel=2)

    self.demand_data = np.ascontiguousarray(demand_data, dtype = np.int32)
    self.renewables_data = np.ascontiguousarray(renewables_data, dtype = np.int32)
    self.net_demand_data = np.ascontiguousarray(self.demand_data - self.renewables_data)

    if not isinstance(gen_distribution.x, univar.Binned) or not isinstance(gen_distribution.y, univar.Binned):
      raise TypeError("Marginal generation distributions must be instances of Binned (i.e. integer support).")

    # save hard copy of relevant arrays from generation
    # this is needed because strange things happened when copying arrays directly from the Independent instance. This somehow solves the issue
    self.convgen1 = {
      "min": gen_distribution.x.min,
      "max": gen_distribution.x.max,
      "cdf_values": np.ascontiguousarray(np.copy(gen_distribution.x.cdf_values, order="C")),
      "expectation_vals": np.ascontiguousarray(np.cumsum(gen_distribution.x.support * gen_distribution.x.pdf_values))}

    self.convgen2 = {
      "min": gen_distribution.y.min,
      "max": gen_distribution.y.max,
      "cdf_values": np.ascontiguousarray(np.copy(gen_distribution.y.cdf_values, order = "C")),
      "expectation_vals": np.ascontiguousarray(np.cumsum(gen_distribution.y.support * gen_distribution.y.pdf_values))}

    self.gen_distribution = gen_distribution
    self.MARGIN_BOUND = int(np.iinfo(np.int32).max / 2)

    if season_length is None:
      warnings.warn("Using length of demand data as season length.", stacklevel=2)
    self.season_length = len(self.demand_data) if season_length is None else season_length

  def cdf(
    self, 
    x: np.ndarray,
    itc_cap: int = 1000, 
    policy: str = "veto"):
    """Evaluates the bivariate post-interconnection power surplus distribution's cumulative distribution function
    
    Args:
        x (np.ndarray): value at which to evaluate the cdf
        itc_cap (int, optional): interconnection capacity
        policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.

    """

    # if x is an n x 2 matrix
    if len(x.shape) == 2 and len(x) > 1:
      return np.array([self.cdf(v,itc_cap,policy) for v in x])

    #bound and unbunble component values
    x = np.clip(x,a_min=-self.MARGIN_BOUND,a_max=self.MARGIN_BOUND)
    x1, x2 = x.reshape(-1)

    #convgen1, convgen2 = self.gen_distribution.x, self.gen_distribution.y
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
        ffi.cast("double *",self.convgen1["cdf_values"].ctypes.data),
        ffi.cast("double *",self.convgen2["cdf_values"].ctypes.data),
        np.int32(x1),
        np.int32(x2),
        np.int32(net_demand1),
        np.int32(net_demand2),
        np.int32(demand1),
        np.int32(demand2),
        np.int32(itc_cap),
        np.int32(policy == "share"))

      cdf += point_cdf
    
    return cdf/self.season_length

  def system_lolp(
    self,
    itc_cap: int=1000):
    """Computes the system-wide post-interconnection loss of load probability. This is, the probability that at least one area will experience a shortfall.
    
    Args:
        itc_cap (int, optional): Interconnector capacity
    
    """
    def trapezoid_prob(ulc,c):

      ulc1, ulc2 = ulc
      return C_API.trapezoid_prob_py_interface(
        np.int32(ulc1),
        np.int32(ulc2),
        np.int32(c),
        np.int32(self.convgen1["min"]),
        np.int32(self.convgen2["min"]),
        np.int32(self.convgen1["max"]),
        np.int32(self.convgen2["max"]),
        ffi.cast("double *",self.convgen1["cdf_values"].ctypes.data),
        ffi.cast("double *",self.convgen2["cdf_values"].ctypes.data))
      
    n = len(self.net_demand_data)
    gen = self.gen_distribution
    lolp = 0
    
    c = itc_cap
    for k in range(n):
      net_demand1, net_demand2 = self.net_demand_data[k]
      # system-wide lolp does not depend on the policy
      point_lolp = gen.cdf(np.array([net_demand1-c-1,np.Inf])) + gen.cdf(np.array([np.Inf,net_demand2-c-1])) - gen.cdf(np.array([net_demand1+c,net_demand2-c-1])) + trapezoid_prob((net_demand1-c-1,net_demand2+c),2*c)
      lolp += point_lolp

    return lolp/self.season_length


  def lole(
    self,
    itc_cap: int = 1000,
    policy: str = "veto",
    area: int=0):
    """Computes the post-interconnection loss of load expectation.
    
    Args:
        itc_cap (int, optional): interconnection capacity
        policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.
        area (int, optional): Area for which to evaluate LOLE; if area=-1, system-wide lole is returned
    """
    if area == -1:
      return len(self.net_demand_data) * self.system_lolp(itc_cap)

    x = np.array([np.Inf,np.Inf])
    x[area] = -1
    #m = (-1,np.Inf)
    lolp = self.cdf(
      x=x, 
      itc_cap=itc_cap,
      policy=policy) 
    
    return self.season_length * lolp


  def swap_axes(self):
    """Utility method to flip components in bivariate distribution objects
    """
    self.demand_data = np.flip(self.demand_data,axis=1)
    self.renewables_data = np.flip(self.renewables_data,axis=1)
    self.net_demand_data = np.flip(self.net_demand_data,axis=1)
    
    aux = copy.deepcopy(self.convgen1)
    self.convgen1 = copy.deepcopy(self.convgen2)
    self.convgen2 = aux

    self.gen_distribution = Independent(x=self.gen_distribution.y, y=self.gen_distribution.x)
    
  def eeu(
    self,
    itc_cap: int = 1000,
    policy: str = "veto",
    area: int=0):
    """Computes the post-interconnection expected energy unserved.
    
    Args:
        itc_cap (int, optional): interconnection capacity
        policy (str, optional): one of 'veto' or 'share'; in a 'veto' policy, areas only export spare available capacity, while in a 'share' policy, exports are market-driven, i.e., by power scarcity at both areas. Shortfalls can extend from one area to another by diverting power.
        area (int, optional): Area for which to evaluate eeu; if area=-1, systemwide eeu is returned
    """
    
    if area == -1:
      return self.eeu(itc_cap, policy,0) + self.eeu(itc_cap, policy, 1)

    if area == 1:
      self.swap_axes()
    
    n = len(self.net_demand_data)
    eeu = 0

    for k in range(n):
      #print(i)
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
          ffi.cast("double *",self.convgen1["cdf_values"].ctypes.data),
          ffi.cast("double *",self.convgen2["cdf_values"].ctypes.data),
          ffi.cast("double *",self.convgen1["expectation_vals"].ctypes.data))
      elif policy == "veto":
        point_EPU = C_API.cond_eeu_veto_py_interface(
          np.int32(net_demand1),
          np.int32(net_demand2),
          np.int32(itc_cap),
          np.int32(self.convgen1["min"]),
          np.int32(self.convgen2["min"]),
          np.int32(self.convgen1["max"]),
          np.int32(self.convgen2["max"]),
          ffi.cast("double *",self.convgen1["cdf_values"].ctypes.data),
          ffi.cast("double *",self.convgen2["cdf_values"].ctypes.data),
          ffi.cast("double *",self.convgen1["expectation_vals"].ctypes.data))
      else:
        raise ValueError(f"Policy name ({policy}) not recognised.")

      eeu += point_EPU

    if area == 1:
      self.swap_axes()

    return eeu

  def get_pointwise_risk(self, x: np.ndarray, itc_cap: int = 1000, policy:str = "veto"):
    """Calculates the post-interconnection shortfall probability for each one of the net demand observations
    
    Args:
        x (np.ndarray): point to evaluate CDF at
        itc_cap (int, optional): interconnection capacity
        policy (str): one of 'veto' or 'share'
    
    """
    pointwise_cdfs = np.empty((len(self.demand_data),))
    for k, (demand_row, renewables_row) in enumerate(zip(self.demand_data, self.renewables_data)):
      pointwise_cdfs[k] = type(self)(
        demand_data=demand_row.reshape((1,2)),
        renewables_data=renewables_row.reshape((1,2)),
        gen_distribution=self.gen_distribution).cdf(x=x, itc_cap=itc_cap, policy=policy)

    return pointwise_cdfs

  def simulate(self, size: int, itc_cap: int = 1000, policy = "veto"):
    """Simulate the post-interconnection bivariate surplus distribution
    
    Args:
        size (int): Sample size
        itc_cap (int, optional): Interconnector capacity
        policy (str): one of 'veto' or 'share'
    
    """
    return self.simulate_region(size, np.array([np.Inf,np.Inf]), itc_cap, policy, False)

  def simulate_region(
    self,
    size: int,
    upper_bounds: np.ndarray,
    itc_cap: int = 1000,
    policy: str = "veto",
    shortfall_region: bool = False):

    """Simulate post-interconnection bivariate surplus distribution conditioned to a rectangular region bounded above.
    
    Args:
        size (int): Sample size
        upper_bounds (np.ndarray): region's upper bounds
        itc_cap (int, optional): Interconnector capacity
        policy (str): one of 'veto' or 'share'
        shortfall_region (bool, optional): If True, upper bounds are ignored and the sampling region becomes the shortfall region, this is, min(S_1, S_2) < 0, or equivalently, that in which at least one area has a shortfall.
    
    """
    seed = np.random.randint(low=0,high=1e8)
    upper_bounds = np.clip(upper_bounds,a_min=-self.MARGIN_BOUND,a_max=self.MARGIN_BOUND)
    m1, m2 = upper_bounds
    n = len(self.demand_data)

    simulated = np.ascontiguousarray(np.zeros((size,2)),dtype=np.int32)
    #convgen1, convgen2 = self.gen_distribution.x, self.gen_distribution.y
    ### calculate conditional probability of each historical observation conditioned to the region of interest
    if shortfall_region:
      warnings.warn("Simulating from shortfall region; ignoring passed upper bounds.", stacklevel=2)
      pointwise_cdfs = self.get_pointwise_risk(x=np.array([0,np.Inf]),itc_cap=itc_cap,policy=policy) + \
        self.get_pointwise_risk(x=np.array([np.Inf,0]),itc_cap=itc_cap,policy=policy) - \
        self.get_pointwise_risk(x=np.array([0,0]),itc_cap=itc_cap,policy=policy) 
      intersection = False
    else:
      pointwise_cdfs = self.get_pointwise_risk(x=upper_bounds,itc_cap=itc_cap,policy=policy)
      intersection = True
    
    # numerical rounding error sometimes output negative probabilities of the order of 1e-30
    pointwise_cdfs = np.clip(pointwise_cdfs, a_min=0.0, a_max=np.Inf)

    total_prob = np.sum(pointwise_cdfs)
    if total_prob <= 1e-8:
      if fixed_area == 1:
        self.swap_axes()
      raise Exception(f"Region has probability {total_prob}; too small to simulate accurately")
    else:
      probs = pointwise_cdfs/total_prob
      
      samples_per_row = np.random.multinomial(n=size,pvals=probs,size=1).reshape((len(probs),))
      nonzero_samples = samples_per_row > 0
      ## only pass rows which induce at least one simulated value
      row_weights = np.ascontiguousarray(samples_per_row[nonzero_samples],dtype=np.int32)

      net_demand = np.ascontiguousarray(self.net_demand_data[nonzero_samples,:],dtype=np.int32)

      demand = np.ascontiguousarray(self.demand_data[nonzero_samples,:],dtype=np.int32)

      C_API.region_simulation_py_interface(
        np.int32(size),
        ffi.cast("int *",simulated.ctypes.data),
        np.int32(self.convgen1["min"]),
        np.int32(self.convgen2["min"]),
        np.int32(self.convgen1["max"]),
        np.int32(self.convgen2["max"]),
        ffi.cast("double *",self.convgen1["cdf_values"].ctypes.data),
        ffi.cast("double *",self.convgen2["cdf_values"].ctypes.data),
        ffi.cast("int *",net_demand.ctypes.data),
        ffi.cast("int *",demand.ctypes.data),
        ffi.cast("int *",row_weights.ctypes.data),
        np.int32(net_demand.shape[0]),
        np.int32(m1),
        np.int32(m2),
        np.int32(itc_cap),
        int(seed),
        int(intersection),
        int(policy == "share"))

      return simulated


  def simulate_conditional(
    self,
    size: int,
    fixed_value: int,
    fixed_area: int,
    itc_cap: int = 1000,
    policy: str = "veto"):
    """Simulate post-interconnection surplus distribution conditioned to a value in the other area's surplus
    
    Args:
        size (int): Sample size
        fixed_value (int): Surplus value conditioned on
        fixed_area (TYPE): Area conditioned on
        itc_cap (int, optional): Interconnector capacity
        policy (str): one of 'veto' or 'share'
    
    """
    seed = np.random.randint(low=0,high=1e8)
    m1 = np.clip(fixed_value,a_min=-self.MARGIN_BOUND,a_max=self.MARGIN_BOUND)
    m2 = self.MARGIN_BOUND

    simulated = np.ascontiguousarray(np.zeros((size,2)),dtype=np.int32)
    
    ### calculate conditional probability of each historical observation given
    ### margin value tuple m

    if fixed_area == 0:
      x = np.array([fixed_value,np.Inf])
      y = x - 1
    else:
      x = np.array([np.Inf, fixed_value])
      y = x - 1

    pointwise_cdfs = self.get_pointwise_risk(x=x,itc_cap=itc_cap,policy=policy) - \
      self.get_pointwise_risk(x=y,itc_cap=itc_cap,policy=policy)

    pointwise_cdfs = np.clip(pointwise_cdfs, a_min=0.0, a_max=np.Inf)
  
    ## rounding errors can make probabilities negative of the order of 1e-60
    total_prob = np.sum(pointwise_cdfs)
    
    if total_prob <= 1e-12:
      raise Exception(f"Region has low probability ({total_prob}); too small to simulate accurately")
    else:
      probs = pointwise_cdfs/total_prob
      
      samples_per_row = np.random.multinomial(n=size,pvals=probs,size=1).reshape((len(probs),))
      nonzero_samples = samples_per_row > 0
      ## only pass rows which induce at least one simulated value
      row_weights = np.ascontiguousarray(samples_per_row[nonzero_samples],dtype=np.int32)

      if fixed_area == 1:
        self.swap_axes()

      net_demand = np.ascontiguousarray(self.net_demand_data[nonzero_samples,:],dtype=np.int32)

      demand = np.ascontiguousarray(self.demand_data[nonzero_samples,:],dtype=np.int32)

      C_API.conditioned_simulation_py_interface(
        np.int32(size),
        ffi.cast("int *",simulated.ctypes.data),
        np.int32(self.convgen1["min"]),
        np.int32(self.convgen2["min"]),
        np.int32(self.convgen1["max"]),
        np.int32(self.convgen2["max"]),
        ffi.cast("double *",self.convgen1["cdf_values"].ctypes.data),
        ffi.cast("double *",self.convgen2["cdf_values"].ctypes.data),
        ffi.cast("int *",net_demand.ctypes.data),
        ffi.cast("int *",demand.ctypes.data),
        ffi.cast("int *",row_weights.ctypes.data),
        np.int32(net_demand.shape[0]),
        np.int32(m1),
        np.int32(itc_cap),
        int(seed),
        int(policy == "share"))

      if fixed_area == 1:
        self.swap_axes()

    return simulated[:,1] #first column has variable conditioned on (constant value)
