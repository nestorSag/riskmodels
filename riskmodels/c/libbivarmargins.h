//#ifndef BIVAR_MARGINS_H_INCLUDED
//#define BIVAR_MARGINS_H_INCLUDED

#include "libunivarmargins.h"

/**
 * @brief This object represents a vector of integer observations
 *
 * @param value data
 * @param size data size (length)
 */
/*typedef struct DoubleVector{
  double* value;
  int size;
} DoubleVector;*/

/**
 * @brief This object represents a bivariate discrete distribution with independent components
 *
 * @param x first component
 * @param y second component
 */

typedef struct BivariateDiscreteDistribution{
  DiscreteDistribution* x;
  DiscreteDistribution* y;
} BivariateDiscreteDistribution;

/**
 * @brief Wrapper around observed demand and net demand values for a particular time
 *
 */

typedef struct ObservedData{
  int net_demand1;
  int net_demand2;
  int demand1;
  int demand2;
} ObservedData;

/**
 * @brief wrapper that represents a general 2-dimensional lattice point
 */

typedef struct Coord{
  int x;
  int y;
} Coord;

/**
 * @brief Characterises a the polygon with a single segment line with a slope of -1. This polygon is characterised by 2 points
 * which represent the left and right ends of the slope. 
 *
 */

typedef struct Polygon{
  Coord* p1;
  Coord* p2;
} Polygon;

/**
 * @brief Wrapper around a 2-dimensional int array
 *
 */

typedef struct IntMatrix{
  int* value;
  int n_rows;
  int n_cols;
} IntMatrix;


typedef struct SimulationParameters{
  int x_bound;
  int y_bound;
  int* obs_weights;
  int seed;
  int intersection;
  int share_policy;
  int c;
  int n;
} SimulationParameters;

int int_max(int a, int b);

int int_min(int a, int b);

int get_element(IntMatrix* m, int i, int j);

void set_element(IntMatrix* m, int i, int j, int x);


/**
 * @brief Parameters needed to simulate rare events from a bivariate power margin distribution
 *
 * @param x_bound x axis bound for simulated points
 * @param y_bound y axis bound for simulated points
 * @param obs_weights number of simulated samples to get using each historic observation
 * @param seed random seed
 * @param intersection whether to simulate from the intersection or union of inequalities of the type x <= m1, y <= m2
 * @param share_policy whether a share policy is used
 * @param c interconnection capacity
 * @param n number of simulated points to get
 */


double sign(double x);
/**
 * @brief Returns the CDF of a bivariate distribution on a lattice in, evaluated (x,y)
 *
 */

double bivariate_gen_cdf(BivariateDiscreteDistribution* F, int x, int y);


/**
 * @brief Returns the PDF of a bivariate distribution on a lattice in, evaluated (x,y)
 *
 */
double bivariate_gen_pdf(BivariateDiscreteDistribution* F, int x, int y);


/**
 * @brief Calculate the probability mass of the inner lattice of a straight triangular segment of the plane. It does not take the hypothenuse points into account.
 *
 * @param F probability distribution object
 * @param x x coordinate of lower left corner
 * @param y y coordinate of lower left corner
 * @param triangle_length  length of of cathethuses
 */
double triangle_prob(BivariateDiscreteDistribution* F, int x, int y, int triangle_length);
/**
 * @brief Calculates the probability of a trapezoidal region in conventional generation space.
 * The region is formed by stacking a right triangle where the hypotenuse facing right, on
 * top of a rectangular segment of the same width.
 *
 * @param F probability distribution object
 * @param ul_x x coordinate of upper left corner
 * @param ul_y y coordinate of upper y corner
 * @param width width of trapezoid
 */

double trapezoid_prob(BivariateDiscreteDistribution* F, int ul_x, int l_y, int width);

/**
 * @brief Calculate conditional EPU given demand and net demand values, under a share policy for a 2-area system
 *
 * @param F probability distribution object
 * @param obs Object with observed demands and net demands at a given time
 * @param c Interconnection capacity
 */

double cond_epu_share(BivariateDiscreteDistribution* F, ObservedData* obs, int c);

/**
 * @brief Calculate conditional EPU given demand and net demand values, under a veto policy for a 2-area system
 *
 * @param F probability distribution object
 * @param obs Object with observed demands and net demands at a given time
 * @param c Interconnection capacity
 */
double cond_epu_veto(BivariateDiscreteDistribution* F, ObservedData* obs, int c);

/**
 * @brief Computes points for that characterises the polygon in conventional generation space that has to be integrated 
 * in order to calculate the probability P(M<=x), for a given interconnector size, a share policy and demand and net demand values
 *
 * @param p Polygon object where points will be saved
 * @param obs Object with observed demands and net demands at a given time
 * @param x Margin value
 * @param c Interconnection capacity
 * @param area area to which the polygon will correspond
 */
void get_share_polygon_points(Polygon* p, ObservedData* obs, int x, int c, int area);

/**
 * @brief Computes points for that characterises the polygon in conventional generation space that has to be integrated 
 * in order to calculate the probability P(M<=x), for a given interconnector size, a veto policy and demand and net demand values
 *
 * @param p Polygon object where points will be saved
 * @param obs Object with observed demands and net demands at a given time
 * @param x Margin value
 * @param c Interconnection capacity
 * @param area area to which the polygon will correspond
 */
void get_veto_polygon_points(Polygon* p, ObservedData* obs, int x, int c, int area);

/**
 * @brief Returns max(x2) such that (x,x2) is part of the polygon
 *
 * @param p Polygon object where points will be saved
 * @param x x axis value to evaluate at
 */
int axis1_polygon_upper_bound(Polygon* p, int x);

/**
 * @brief Returns max(x1) such that (x1,x) is part of the polygon
 *
 * @param p Polygon object where points will be saved
 * @param x y axis value to evaluate at
 */
int axis2_polygon_upper_bound(Polygon* p, int x);

/**
 * @brief Maps a uniform variable in [0,1] to the quantile of a distribution conditioned to be below a threshold
 *
 * @param F distribution object
 * @param upper_bound given upper bound
 */
int get_bounded_quantile(DiscreteDistribution* F, int upper_bound, double u);

/**
 * @brief Returns veto flow to/from area 1 given power margins and interconnector capacity
 *
 * @param m1 Power margin at area 1
 * @param m2 Power margin at area 2
 * @param c Interconnection capacity
 */
int veto_flow(int m1,int m2,int c);

/**
 * @brief Returns share flow to/from area 1 given power margins, demands and interconnector capacity
 *
 * @param m1 Power margin at area 1
 * @param m2 Power margin at area 2
 * @param d1 demand at area 2
 * @param d2 demand at area 2
 * @param c Interconnection capacity
 */
double share_flow(int m1,int m2,int d1,int d2,int c);


/**
 * @brief Calculate probability mass of the intersection of 2 given polygons; such polygons are given by the given values of net demand and demand.
 *
 * @param F distribution object
 * @param plg1 Polygon 1
 * @param plg2 Polygon 2
 */
double cond_bivariate_power_margin_cdf(BivariateDiscreteDistribution* F, Polygon* plg1, Polygon* plg2);

/**
 * @brief get the rightmost X coordinate of the intersection or union of both polygons
 *
 * @param F distribution object
 * @param plg1 Polygon1
 * @param plg2 Polygon2
 * @param intersection whether to apply union or intersection to both polygons
 */
int get_joint_polygon_x_bound(
  BivariateDiscreteDistribution* F, 
  Polygon* plg1, 
  Polygon* plg2, 
  int intersection);

/**
 * @brief get the highest Y coordinate of the intersection or union of both polygons at a particular X coordinate
 *
 * @param F distribution object
 * @param plg1 Polygon1
 * @param plg2 Polygon2
 * @param x x axis coordinate
 * @param intersection whether to apply union or intersection to both polygons
 */
int get_joint_polygon_y_bound_given_x(
  BivariateDiscreteDistribution* F, 
  Polygon* plg1, 
  Polygon* plg2, 
  int x, 
  int intersection);


/**
 * @brief Simulate bivariate values for conventional generation, 
 * such that bivariate post interconnector margins fall into the region specified by the bounds in the parameters' object
 *
 * @param F distribution object
 * @param data observed demand and net demand data at a given time
 * @param net_demand matrix of net demand values
 * @param demand matrix of demand values
 * @param results matrix where results will be saved
 * @param parameters Set of simulation parameters
*/

void region_simulation(
  BivariateDiscreteDistribution* F, 
  IntMatrix* net_demand,
  IntMatrix* demand,
  IntMatrix* results,
  SimulationParameters* parameters);


/**
 * @brief Simulates power margin values in one axis conditioned on margin values on the other axis
 *
 * @param F distribution object
 * @param net_demand matrix of net demand values
 * @param demand matrix of demand values
 * @param results matrix where results will be saved
 * @param parameters Set of simulation parameters
*/

void conditioned_simulation(
  BivariateDiscreteDistribution* F,
  IntMatrix* net_demand,
  IntMatrix* demand,
  IntMatrix* results,
  SimulationParameters* parameters);

// Returns bivariate empirical CDF values from a matrix of observations
void bivariate_empirical_cdf(
  DoubleVector* ecdf,
  IntMatrix* X);

/**
 * @brief Returns filled polygon objects
 *
 * @param F distribution object
 * @param obs wrapper that contains observed demand and net demand values for a particular time
 * @param parameters Set of simulation parameters
*/

void get_polygons(Polygon* p1, Polygon* p2, ObservedData* obs, SimulationParameters* pars);



void get_double_vector_from_py_objs(DoubleVector* vector, double* value, int size);

void get_observed_data_from_py_objs(ObservedData* obs, int d1, int d2, int nd1, int nd2);

void get_int_matrix_from_py_objs(IntMatrix* m, int* value, int n_rows, int n_cols);

void get_sim_pars_from_py_objs(
  SimulationParameters* pars,
  int x_bound,
  int y_bound,
  int* obs_weights,
  int seed,
  int intersection,
  int share_policy,
  int c,
  int n);
// py interfaces

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

/*void bivariate_empirical_cdf_py_interface(
  double* ecdf,
  int* X,
  int n);*/


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