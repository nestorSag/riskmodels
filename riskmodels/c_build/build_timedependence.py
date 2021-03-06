from cffi import FFI
import os
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

header = f'#include "{project_dir / "riskmodels" / "c" / "libtimedependence.h"}"'

libs = ["m"] if os.name != "nt" else None

sources = [
  f'{Path("riskmodels") /"c"/"libtimedependence.c"}', 
  f'{Path("riskmodels") /"c"/ "mtwist-1.5" / "mtwist.c"}'
  ]

ffibuilder = FFI()

ffibuilder.cdef(
    """ 

  void calculate_post_itc_share_margins_py_interface(
    float* margin_series,
    float* dem_series,
    int period_length,
    int series_length,
    int n_areas,
    float c);

  void calculate_post_itc_veto_margins_py_interface(
    float* margin_series,
    int series_length,
    int n_areas,
    float c);


  void calculate_pre_itc_margins_py_interface(
    float* gen_series,
    float* netdem_series,
    int period_length,
    int series_length,
    int n_areas);

  void simulate_mc_power_grid_py_interface(
      float *output, 
      float *transition_probs,
      float *states,
      float *initial_values,
      int n_generators,
      int n_simulations, 
      int n_transitions, 
      int n_states,
      int random_seed,
      int simulate_streaks);

	"""
)

ffibuilder.set_source(
    "c_sequential_models_api",  # name of the output C extension
    header,
    sources=sources,
    libraries=libs,
)  # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
