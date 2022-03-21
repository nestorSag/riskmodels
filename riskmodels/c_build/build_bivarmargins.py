from cffi import FFI
import os

from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]

ffibuilder = FFI()

ffibuilder.cdef(
    """ 

	double triangle_prob_py_interface(
	    int origin_x,
	    int origin_y,
	    int triangle_length,
	    int min_gen1,
	    int min_gen2,
	    int max_gen1,
	    int max_gen2,
	    double* gen1_cdf_array,
	    double* gen2_cdf_array);

	double cond_eeu_veto_py_interface(
	  int v1,
	  int v2,
	  int c,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  double* gen1_expectation);

	double cond_eeu_share_py_interface(
	  int d1, 
	  int d2,
	  int v1,
	  int v2,
	  int c,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  double* gen1_expectation);

	double trapezoid_prob_py_interface(
	  int ul_x,
	  int ul_y,
	  int width,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array);

	void region_simulation_py_interface(
	  int n,
	  int* simulations,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  int* net_demand,
	  int* demand,
	  int* row_weights,
	  int n_rows,
	  int m1,
	  int m2,
	  int c,
	  int seed,
	  int intersection,
	  int share_policy);

	void conditioned_simulation_py_interface(
	  int n,
	  int* simulations,
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  int* net_demand,
	  int* demand,
	  int* row_weights,
	  int n_rows,
	  int m1,
	  int c,
	  int seed,
	  int share_policy);

	double cond_bivariate_power_margin_cdf_py_interface(
	  int min_gen1,
	  int min_gen2,
	  int max_gen1,
	  int max_gen2,
	  double* gen1_cdf_array,
	  double* gen2_cdf_array,
	  int v1,
	  int v2,
	  int d1,
	  int d2,
	  int m1,
	  int m2,
	  int c,
	  int share_policy);

	"""
)

# with open('riskmodels/_c/libbivarmargins.h','r') as f:
# 	ffibuilder.cdef(f.read())

header = f'#include "{project_dir / "riskmodels" / "c" / "libbivarmargins.h"}"'

ffibuilder.set_source(
    "c_bivariate_surplus_api",  # name of the output C extension
    # """
    # #include "../../riskmodels/_c/libbivarmargins.h"
    # """,
    header,
    sources=[
        "riskmodels/c/libbivarmargins.c",
        "riskmodels/c/libunivarmargins.c",
        "riskmodels/c/mtwist-1.5/mtwist.c",
    ],
    libraries=["m"],
)  # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
