//#ifndef TIMEDEPENDENCE_H_INCLUDED
//#define TIMEDEPENDENCE_H_INCLUDED

/**
 * @brief Wrapper around a 2-dimensional float array
 *
 */

/*typedef struct FloatMatrix{
  float* value;
  int n_rows;
  int n_cols;
} FloatMatrix;*/

/**
 * @brief Wrapper around a 2-dimensional float array
 *
 */

typedef struct FloatMatrix{
  float* value;
  int n_rows;
  int n_cols;
} FloatMatrix;

/**
 * @brief Markov chain wrapper
 *
 * @param states pointer to array of states
 * @param transition_probs pointer to array of transition probabilities
 * @param initial_state state at time t=0
 * @param n_states number of states
 */
typedef struct MarkovChain{
  float* states;
  float* transition_probs;
  float initial_state;
  int n_states;
} MarkovChain;

/**
 * @brief This object represents a bivariate discrete distribution with independent components
 *
 * @param x first component
 * @param y second component
 */
typedef struct MarkovChainArray{
  MarkovChain* chains;
  int size;
} MarkovChainArray;

/**
 * @brief This object represents a bivariate discrete distribution with independent components
 *
 * @param x first component
 * @param y second component
 */
typedef struct TimeSimulationParameters{
  int n_simulations;
  int n_transitions;
  int seed;
  int simulate_streaks;

} TimeSimulationParameters;

/**
 * @brief simulate a time series of availability states for a single generator, simulating each step separately
 *
 * @param output 1D array where time series is going to be stored
 * @param chain wrapper for a Markov Chain data
 * @param n_transitions number of transitions to simulate
 * @return @c void
 */
void simulate_mc_generator_steps(float *output, MarkovChain* chain, int n_transitions);

float* get_float_element_pointer(FloatMatrix* m, int i, int j);

float get_float_element(FloatMatrix* m, int i, int j);

void set_float_element(FloatMatrix* m, int i, int j, float x);

int simulate_geometric_dist(float p);

float float_min(float num1, float num2) ;

float float_max(float num1, float num2);

/**
 * @brief simulate a time series of availability states for a single generator, simulating escape time: the time it takes for the generator to switch to a different state. 
 *
 * @param output 1D array where time series is going to be stored
 * @param chain wrapper for a Markov Chain data
 * @param n_transitions number of transitions to simulate
 * @return @c void
 */
void simulate_mc_generator_streaks(float* output, MarkovChain* chain, int n_transitions);

int get_next_state_idx(
  float* prob_row, int current_state_idx, int n_states);
/**
 * @brief simulate a time series of aggregated availabilities for a set of generators
 *
 * @param output 1D array where each time series simulation is going to be stores
 * @param mkv_chains Pointer to MarkovChainArray object
 * @param pars Wrapper of simulation parameter values
 * @return @c void
 */
void simulate_mc_power_grid(FloatMatrix* output, MarkovChainArray* mkv_chains, TimeSimulationParameters* pars);

/**
 * @brief Calculate bivariate pre-interconnection margins
 *
 * @param gen_series struct that wraps a generation time series simulation
 * @param netdem_series struct that wraps net demand time series
 */

void calculate_pre_itc_margins(FloatMatrix* gen_series, FloatMatrix* netdem_series);

// gets the minimum between 3 values

float min3(float a, float b, float c);

/**
 * @brief get power flow to area 1 under a veto policy
 *
 * @param m1 margin of area 1
 * @param m2 margin of area2
 * @param c interconnection capacity
 */

float get_veto_flow(float m1, float m2, float c);


/**
 * @brief get power flow to area 1 under a share policy 
 *
 * @param m1 margin of area 1
 * @param m2 margin of area2
 * @param d1 demand at area 1
 * @param d2 demand at area 2
 * @param c interconnection capacity
 */

float get_share_flow(
  float m1,
  float m2,
  float d1,
  float d2,
  float c);

/**
 * @brief calculate post-interconnection margins under a veto policy
 *
 * @param power_margin_matrix struct for power margin matrix values
 * @param c interconnection capacity
 */

void calculate_post_itc_veto_margins(FloatMatrix* power_margin_matrix, float c);


/**
 * @brief calculate post-interconnection margins under a share policy
 *
 * @param power_margin_matrix struct for power margin matrix values
 * @param power_margin_matrix struct for demand matrix values
 * @param c interconnection capacity
 */

void calculate_post_itc_share_margins(FloatMatrix* power_margin_matrix, FloatMatrix* demand_matrix, int c);

void get_float_matrix_from_py_objs(FloatMatrix* m, float* value, int n_rows, int n_cols);

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

//#endif