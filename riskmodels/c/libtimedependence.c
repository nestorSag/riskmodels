#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mtwist-1.5/mtwist.h"

#include "libtimedependence.h"

// documentation is in .h files 

/*float get_double_element(DoubleMatrix* m, int i, int j){
  return m->value[i*m->n_cols + j];
}
void set_double_element(DoubleMatrix* m, int i, int j, float x){
  m->value[i*m->n_cols + j] = x;
}*/

float* get_float_element_pointer(FloatMatrix* m, int i, int j){
  return &(m->value[i*m->n_cols + j]);
}

float get_float_element(FloatMatrix* m, int i, int j){
  return m->value[i*m->n_cols + j];
}

void set_float_element(FloatMatrix* m, int i, int j, float x){
  m->value[i*m->n_cols + j] = x;
}


void simulate_mc_generator_steps(float *output, MarkovChain* chain, int n_transitions){
  // simulate each step in a power availability time series

  int current_state_idx = 0, current_timestep = 0, k = 0, n_timesteps=n_transitions+1;
  float cdf;
  double u;

  //find index of initial state to get initial transition probability row
  while(chain->states[current_state_idx] != chain->initial_state){
    ++current_state_idx;
  }
  output[current_timestep] = chain->states[current_state_idx]; //initial state output

  float *prob_row = &chain->transition_probs[chain->n_states*current_state_idx];


  // simulate n hours and save inplace in output
  for(current_timestep = 1; current_timestep < n_timesteps; ++current_timestep){
    cdf = 0;
    k = 0;
    u = mt_drand();

    while(cdf < u){
      cdf += prob_row[k];
      ++k;
    }


    prob_row = &chain->transition_probs[chain->n_states*(k-1)];
    //output[current_timestep] = chain->states[k-1];
    output[current_timestep] += chain->states[k-1];

  }

}


int simulate_geometric_dist(
  float p){
  // p = probability of looping back to same state
  double u = mt_drand();
  int x;

  //printf("p: %f, u: %f\n",p,u);

  if(u<=p){
    return 0;
  }else{
    x = (int) ceil(log(1.0-u)/log(1.0-p)-1.0);
    return x;
  }
}


int get_next_state_idx(
  float* prob_row, int current_state_idx, int n_states){

  //float cdf; //total_prob = probabiliti mass f(x)
  float  cdf=0.0, u, escape_prob = 0.0;
  int j = 0;

  // compute total escape probability
  for(j=0;j<n_states;j++){
    if(j!= current_state_idx){
      escape_prob += prob_row[j];
    }
  }

  //ensure u is feasible (rounding errors can make it unfeasible)
  u = escape_prob * ((float) mt_drand());
  while(u > escape_prob){
    u = escape_prob * ((float) mt_drand()); 
  }

  j = 0;
  while(cdf < u){
    if(j != current_state_idx){
      cdf += prob_row[j];
    }
    j += 1;
  }

  return j-1;

}

/**
 * Find minimum between two numbers.
 */
float float_min(float num1, float num2) 
{
    return (num1 > num2 ) ? num2 : num1;
}

float float_max(float num1, float num2) 
{
    return (num1 > num2 ) ? num1 : num2;
}


void simulate_mc_generator_streaks(float* output, MarkovChain* chain, int n_transitions){
  // simulate 'escape time': number of steps before leaving a state
  // if one state has a large stationary probability (> 0.5) this method might be faster
  // than simply simulating each step

  //find index of initial state to get initial transition probability row
  int current_state_idx = 0, current_timestep = 0, k = 0, streak, next_state_idx, n_timesteps = n_transitions+1;
  float current_state;

  while(chain->states[current_state_idx] != chain->initial_state){
    ++current_state_idx;
  }

  // get initial row and loop probability
  float *prob_row = &chain->transition_probs[chain->n_states*current_state_idx];

  float prob_loop = prob_row[current_state_idx];

  while(current_timestep < n_timesteps){
    current_state = chain->states[current_state_idx];
    //printf("current timestep: %d\n",current_timestep);
    //printf("current state id: %d\n",current_state_idx);
    streak = float_min(simulate_geometric_dist(1.0-prob_loop), n_transitions - current_timestep);
    //printf("streak: %d\n",streak);
    next_state_idx = get_next_state_idx(prob_row,current_state_idx,chain->n_states);
    //printf("next state id: %d\n",next_state_idx);

    for(k=current_timestep;k<=current_timestep+streak;++k){
      output[k] += current_state;
      //output[k] = current_state;
    }

    // update objects
    current_timestep += streak+1;
    current_state_idx = next_state_idx;
    prob_row = &chain->transition_probs[chain->n_states*current_state_idx];
    prob_loop = prob_row[current_state_idx];

  }

}

void simulate_mc_power_grid(FloatMatrix* output, MarkovChainArray* mkv_chains, TimeSimulationParameters* pars){

  mt_seed32(pars->seed);

  int i = 0, j = 0;

  //int output_length = pars->n_timesteps+1; //number of simulated timesteps + initial state

  //float aggregate[output_length], gen_output[pars->n_timesteps];
  float* current_trace;
  MarkovChain* chains = mkv_chains->chains;

  for(i=0;i<pars->n_simulations;++i){

    // initialise auxiliary aggregator

    /*for(k=0;k<output_length;++k){
      aggregate[k] = 0;
    }*/

    current_trace = get_float_element_pointer(output,i,0);

    float current_value = 0.0, last_value = 0.0, gencap;

    for(j=0;j<mkv_chains->size;++j){
      // get generators' output
      
      if(pars->simulate_streaks > 0){

        //simulate_mc_generator_streaks(gen_output, &chains[j], pars->n_timesteps);
        simulate_mc_generator_streaks(current_trace, &chains[j], pars->n_transitions);

      }else{

        //simulate_mc_generator_steps(gen_output, &chains[j], pars->n_timesteps);
        simulate_mc_generator_steps(current_trace, &chains[j], pars->n_transitions);

      }

      /*if(i==4643){
        //printf("output[7526] = %f, gen. cap. %f, %d\n",current_trace[7526], (&chains[j])->states[0],j);
        gencap = (&chains[j])->states[0];
        last_value = current_value;
        current_value = current_trace[7526];
        if(current_value - last_value > gencap){
          printf("gencap %f, diff %f, j %d\n",gencap,current_value-last_value,j);
        }
      }*/

      /*for(k = 0; k < output_length; ++k){
        aggregate[k] += gen_output[k];
      }*/

    }

    /*for(k = 0; k < output_length; ++k){
      set_float_element(output,i,k,(float) aggregate[k]);
      
    }*/

  }

}


float min3(float a, float b, float c){
  if(a < b && a < c){
    return a;
  }else{
    if(b < c){
      return b;
    }else{
      return c;
    }
  }
}


float get_veto_flow(float m1, float m2, float c){
  float transfer = 0.0;
  if(m1 < 0 && m2 > 0){
    transfer = min3(-m1,m2,c);
  }else{
    if(m1 > 0 && m2 < 0){
      transfer = -min3(m1,-m2,c);
    }
  }

  return transfer;
}

float get_share_flow(
  float m1,
  float m2,
  float d1,
  float d2,
  float c){

  float alpha = d1/(d1+d2);
  float unbounded_flow = alpha*m2 - (1-alpha)*m1;

  if (m1+m2 < 0 && m1 < c && m2 < c){
    return float_min(float_max( unbounded_flow,-c),c);
  }else{
    return get_veto_flow(m1,m2,c);
  }
}

void calculate_post_itc_veto_margins(FloatMatrix* power_margin_matrix, float c){

  int i;
  float m1, m2, flow;
  for(i=0;i<power_margin_matrix->n_rows;++i){
    m1 = get_float_element(power_margin_matrix,i,0);
    m2 = get_float_element(power_margin_matrix,i,1);
    flow = get_veto_flow(m1, m2,c);

    set_float_element(power_margin_matrix,i,0, m1 + flow);
    set_float_element(power_margin_matrix,i,1, m2 - flow);

  }
}

void calculate_post_itc_share_margins(FloatMatrix* power_margin_matrix, FloatMatrix* demand_matrix, int c){

  float flow, m1, m2;
  int i, j=0, period_length = demand_matrix->n_rows;
  for(i=0;i<power_margin_matrix->n_rows;++i){
    m1 = get_float_element(power_margin_matrix, i, 0);
    m2 = get_float_element(power_margin_matrix, i, 1);

    flow = get_share_flow(
      m1,
      m2,
      get_float_element(demand_matrix, j, 0),
      get_float_element(demand_matrix, j, 1),
      c);

    set_float_element(power_margin_matrix,i,0, m1 + flow);
    set_float_element(power_margin_matrix,i,1, m2 - flow);
    // this is to avoid using integer remainder operators, which is expensive
    j += 1;
    if(j==period_length){
      j = 0;
    }

  }
}

void calculate_pre_itc_margins(FloatMatrix* gen_series, FloatMatrix* netdem_series){
  // Assuming row-major order

  int i = 0, j=0, k=0, period_length = netdem_series->n_rows, series_length = gen_series->n_rows, n_areas = gen_series->n_cols;
  for(k=0;k<n_areas;++k){
    for(i = 0; i < series_length; ++i){
      set_float_element(gen_series, i, k, get_float_element(gen_series, i, k) - get_float_element(netdem_series, j, k));
      //gen_series[k + n_areas*i] += (-netdem_series[k + n_areas*j]);
      // this is to avoid using integer remainder operators, which is expensive
      j += 1;
      if(j==period_length){
        j = 0;
      }
    }
  }
}








// interfaces for Python

void get_float_matrix_from_py_objs(FloatMatrix* m, float* value, int n_rows, int n_cols){

  m->value = value;
  m->n_rows = n_rows;
  m->n_cols = n_cols;
}

void calculate_post_itc_share_margins_py_interface(
  float* margin_series,
  float* dem_series,
  int period_length,
  int series_length,
  int n_areas,
  float c){

  FloatMatrix power_margin_matrix;
  FloatMatrix demand_matrix;

  get_float_matrix_from_py_objs(&power_margin_matrix, margin_series, series_length, n_areas);
  get_float_matrix_from_py_objs(&demand_matrix, dem_series, period_length, n_areas);

  calculate_post_itc_share_margins(&power_margin_matrix, &demand_matrix, c);

}

void calculate_post_itc_veto_margins_py_interface(
  float* margin_series,
  int series_length,
  int n_areas,
  float c){

  FloatMatrix power_margin_matrix;

  get_float_matrix_from_py_objs(&power_margin_matrix, margin_series, series_length, n_areas);

  calculate_post_itc_veto_margins(&power_margin_matrix, c);

}


void calculate_pre_itc_margins_py_interface(
  float* gen_series,
  float* netdem_series,
  int period_length,
  int series_length,
  int n_areas){

  FloatMatrix generation_matrix;
  FloatMatrix net_demand_matrix;

  get_float_matrix_from_py_objs(&generation_matrix, gen_series, series_length, n_areas);
  get_float_matrix_from_py_objs(&net_demand_matrix, netdem_series, period_length, n_areas);

  calculate_pre_itc_margins(&generation_matrix, &net_demand_matrix);

}

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
  int simulate_streaks){

  int i=0;

  TimeSimulationParameters pars;

  pars.n_simulations = n_simulations;
  pars.n_transitions = n_transitions;
  pars.seed = random_seed;
  pars.simulate_streaks = simulate_streaks;

  //MarkovChain chains[n_generators];
  MarkovChain *chains = (MarkovChain *)malloc(n_generators * sizeof(MarkovChain));

  MarkovChainArray mkv_chains;

  for(i=0;i<n_generators;i++){
    MarkovChain current;
    current.n_states = n_states;
    current.initial_state = initial_values[i];
    current.states = &states[n_states*i];
    current.transition_probs = &transition_probs[n_states*n_states*i];
    chains[i] = current;
  }

  mkv_chains.chains = &chains[0];
  mkv_chains.size = n_generators;

  FloatMatrix m;
  get_float_matrix_from_py_objs(&m,output,n_simulations,n_transitions+1);

  simulate_mc_power_grid(&m, &mkv_chains, &pars);
  free(chains);
}