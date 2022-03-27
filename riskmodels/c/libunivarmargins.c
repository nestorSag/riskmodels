#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libunivarmargins.h"

double cumulative_value(DiscreteDistribution* F, double* array, int x){
  if(x < F->min){
    return 0.0;
  }else{
    if(x >= F->max){
      return array[F->max - F->min];
    }else{
      return array[x - F->min];
    }
  } 
}

double gen_cdf(DiscreteDistribution* F, int x){
  return cumulative_value(F,F->cdf,x);
}

double gen_pdf(DiscreteDistribution* F, int x){
  return gen_cdf(F,x) - gen_cdf(F,x-1);
}

double cumulative_expectation(DiscreteDistribution* F, int x){
  return cumulative_value(F,F->expectation,x);
}

// documentation is in .h files 

double double_max(double num1, double num2){
    return (num1 > num2 ) ? num1 : num2;
}

double double_min(double num1, double num2){
    return (num1 > num2 ) ? num2 : num1;
}

double empirical_power_margin_cdf(DiscreteDistribution* F, IntVector* net_demand, int x){

  double cdf_val = 0;
  int i=0;

  for(i=0;i<net_demand->size;++i){
    cdf_val += gen_cdf(F,(int) net_demand->value[i]+x);
  }

  return cdf_val/net_demand->size;
}

double empirical_cvar(DiscreteDistribution* F, IntVector* net_demand, int q){
  double cvar = 0;
  int upper_bound = q + 1, i, current;

  for(i=0;i<net_demand->size;++i){
    current = (int) net_demand->value[i];
    cvar += current*gen_cdf(F,current-upper_bound) - cumulative_expectation(F,current-upper_bound);
  }

  return cvar/net_demand->size;
}

/*double empirical_eeu(DiscreteDistribution* F, IntVector* net_demand){
  double eeu = 0;
  int i, current;
  for(i=0;i<net_demand->size;++i){
    current = (int) net_demand->value[i];
    eeu += current*gen_cdf(F,current-1) - cumulative_expectation(F,current-1);
  }
  return eeu/net_demand->size;
}*/

double gpdist_cdf(GPModel* gp, double x){

  double xi = gp->xi, u = gp->u, sigma = gp->sigma, val = 0;

  if(xi>=0){
    val = 1.0 - pow(1.0 + xi*(x-u)/sigma,-1.0/xi);
  }else{
    if(x<u - sigma/xi){
      val = 1.0 - pow(1.0 + xi*(x-u)/sigma,-1.0/xi);
    }else{
      val = 1.0;
    }
  }

  return val;

}

double expdist_cdf(GPModel* gp, double x){
  return 1.0 - exp(-(x-gp->u)/gp->sigma); 
}


double tail_model_cdf(GPModel* gp, double x){
  if(gp->xi != 0){
    return gpdist_cdf(gp, x);
  }else{
    return expdist_cdf(gp, x);
  }
}





/*void bayesian_tail_model_cdf_trace(PosteriorGPTrace* gpt, DoubleVector* output, double x){
  double estimator=0;
  int i;
  GPModel current;
  for(i=0;i<gpt->size;++i){
    
    current.xi = gpt->xi[i];
    current.u = gpt->u;
    current.p = 1; //irrelevant
    current.sigma = gpt->sigma[i];
    output->value[i] = tail_model_cdf(&current, x);
  }
}*/


/*double bayesian_tail_model_cdf(PosteriorGPTrace* gpt, double x){
  double estimator=0;
  int i;
  GPModel current;
  for(i=0;i<gpt->size;++i){
    
    current.xi = gpt->xi[i];
    current.u = gpt->u;
    current.p = 1; //irrelevant
    current.sigma = gpt->sigma[i];
    estimator += tail_model_cdf(&current, x);
  }
  return estimator/gpt->size;
}*/




double empirical_net_demand_cdf(IntVector* net_demand, double x){

  int i;

  double nd_below_x = 0;

  for(i=0;i<net_demand->size;++i){
    if(net_demand->value[i]<=x){
      nd_below_x += 1;
    }
  }
  return nd_below_x/net_demand->size;
}

double empirical_net_demand_pdf(IntVector* net_demand, double x){

  int i;

  double p = 0;

  for(i=0;i<net_demand->size;++i){
    if(net_demand->value[i]==x){
      p += 1;
    }
  }
  return p/net_demand->size;
}


double semiparametric_net_demand_cdf(GPModel* gp, IntVector* net_demand, double x){

  if(x <= gp->u){
    return empirical_net_demand_cdf(net_demand,x);
  }else{
    return gp->p + (1.0-gp->p)*tail_model_cdf(gp, x);
  }
}


double semiparametric_net_demand_pdf(GPModel* gp, IntVector* net_demand, double x){

  return semiparametric_net_demand_cdf(gp,net_demand,x) - semiparametric_net_demand_cdf(gp,net_demand,x-1);
}





/*void bayesian_semiparametric_net_demand_cdf_trace(PosteriorGPTrace* gpt, IntVector* net_demand, DoubleVector* output, double x){
  int i;
  GPModel current;
  current.u = gpt->u;
  current.p = gpt->p;
  for(i=0;i<gpt->sizeli++){
    current.sigma = gpt->sigma[i];
    current.xi = gpt->xi[i];
    output->value[i] = semiparametric_net_demand_cdf(&current, net_demand, x);
  }
}
double bayesian_semiparametric_net_demand_cdf(PosteriorGPTrace* gpt, IntVector* net_demand, double x){
  if(x <= gpt->u){
    return empirical_net_demand_cdf(net_demand,x);
  }else{
    return gpt->p + (1.0-gpt->p)*bayesian_tail_model_cdf(gpt,x);
  }
}
double bayesian_semiparametric_net_demand_pdf_trace(PosteriorGPTrace* gpt, IntVector* net_demand, DoubleVector* output, double x){
  int i;
  GPModel current;
  current.u = gpt->u;
  current.p = gpt->p;
  for(i=0;i<gpt->sizeli++){
    current.sigma = gpt->sigma[i];
    current.xi = gpt->xi[i];
    output->value[i] = semiparametric_net_demand_pdf(&current, net_demand, x);
  }
}*/

/*double bayesian_semiparametric_net_demand_pdf(PosteriorGPTrace* gpt, IntVector* net_demand, double x){
  return bayesian_semiparametric_net_demand_cdf(gpt,net_demand,x) - bayesian_semiparametric_net_demand_cdf(gpt,net_demand,x-1);
}*/






double semiparametric_power_margin_cdf(GPModel* gp, IntVector* net_demand, DiscreteDistribution* F, double x){

  int y;

  double cdf_val = 0;
  double pdf_val, g_cdf;

  for(y=0;y<net_demand->size;++y){
    if(net_demand->value[y] < gp->u){
      //cdf += get_gen_array_val(nd_vals[y]+x,gen_cdf,gen_min,gen_max+1)/nd_length;
      cdf_val += gen_cdf(F, (int) net_demand->value[y]+x);
    }
  }
  cdf_val /=net_demand->size;

  for(y=(int) ceil(gp->u);y<F->max-x+1;++y){
    pdf_val = semiparametric_net_demand_pdf(gp,net_demand,y);
    g_cdf = gen_cdf(F,(int) (y+x));
    cdf_val += g_cdf*pdf_val;
  }

  cdf_val += 1.0 - semiparametric_net_demand_cdf(gp,net_demand,F->max-x);

  return cdf_val;
}




void bayesian_semiparametric_power_margin_cdf_trace(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F, DoubleVector* output, double x) {

  int i, y;

  double constant_factor = 0;

  GPModel current;
  current.u = gpt->u;
  current.p = gpt->p;

  for(y=double_max((double) F->min,-x);y<gpt->u;++y){

    constant_factor = empirical_net_demand_pdf(net_demand,y)*gen_cdf(F,(int) (y+x));
  }

  for(i=0;i<gpt->size;i++){

    output->value[i] = constant_factor;

    current.sigma = gpt->sigma[i];
    current.xi = gpt->xi[i];
    for(y=(int) ceil(gpt->u);y<=F->max-x;++y){

      output->value[i] += gen_cdf(F,(int) (y+x))*semiparametric_net_demand_pdf(&current,net_demand,y);
    }

    output->value[i] += 1.0 - semiparametric_net_demand_cdf(&current,net_demand,F->max - x);
  }

}

/*double bayesian_semiparametric_power_margin_cdf(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F, double x) {
  int y;
  double cdf_val = 0;
  double pdf_val;
  for(y=double_max((double) F->min,-x);y<F->max-x+1;++y){
    pdf_val = bayesian_semiparametric_net_demand_pdf(gpt,net_demand,y);
    cdf_val += gen_cdf(F,(int) (y+x))*pdf_val;
  }
  cdf_val += 1.0 - bayesian_semiparametric_net_demand_cdf(gpt,net_demand,F->max - x);
  return cdf_val;
}*/




double semiparametric_cvar(GPModel* gp, IntVector* net_demand, DiscreteDistribution* F, int q){

  double cvar = 0, semiparametric_cvar = 0, pdf_val, remainder;

  int upper_bound = q + 1, sum_split =  F->max+q-1, i, y, current;

  if(gp->xi>=1){
    cvar = -1.0; //infinite expectation
  }else{

    for(i=0;i<net_demand->size;++i){
      current = (int) net_demand->value[i];
      if(current < gp->u){
        cvar += current*gen_cdf(F,current-upper_bound) - cumulative_expectation(F,current-upper_bound);
        //printf("current: %d, cvar: %f\n",current,cvar);
      }
    }
    cvar /= net_demand->size;

    //genpar_expectation = (1-gp->p)*(gp->u + gp->sigma/(1-gp->xi));

    for(y=(int) ceil(gp->u);y<=sum_split;++y){

      pdf_val = semiparametric_net_demand_pdf(gp,net_demand,y);

      semiparametric_cvar += pdf_val*(y*gen_cdf(F,y-upper_bound) - cumulative_expectation(F,y-upper_bound));
    }

    remainder = (1 - semiparametric_net_demand_cdf(gp,net_demand,sum_split))*(sum_split + 1 + gp->sigma/(1-gp->xi));

    cvar += semiparametric_cvar + remainder - cumulative_expectation(F,F->max)*(1-semiparametric_net_demand_cdf(gp,net_demand,sum_split));

  }

  return cvar;
}

/*double semiparametric_eeu(GPModel* gp, IntVector* net_demand, DiscreteDistribution* F){
  double eeu = 0, semiparametric_eeu = 0, pdf_val, remainder;
  int i, y, current;
  if(gp->xi>=1){
    eeu = -1.0; //infinite expectation
  }else{
    for(i=0;i<net_demand->size;++i){
      current = (int) net_demand->value[i];
      if(current < gp->u){
        eeu += current*gen_cdf(F,current-1) - cumulative_expectation(F,current-1);
        //printf("current: %d, eeu: %f\n",current,eeu);
      }
    }
    eeu /= net_demand->size;
    //genpar_expectation = (1-gp->p)*(gp->u + gp->sigma/(1-gp->xi));
    for(y=(int) ceil(gp->u);y<F->max+1;++y){
      pdf_val = semiparametric_net_demand_pdf(gp,net_demand,y);
      semiparametric_eeu += pdf_val*(y*gen_cdf(F,y-1) - cumulative_expectation(F,y-1));
    }
    remainder = (1 - semiparametric_net_demand_cdf(gp,net_demand,F->max))*(F->max + gp->sigma/(1-gp->xi));
    eeu += semiparametric_eeu + remainder - cumulative_expectation(F,F->max)*(1-semiparametric_net_demand_cdf(gp,net_demand,F->max));
  }
  return eeu;
}*/

/*double bayesian_semiparametric_eeu(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F){
  double eeu = 0, semiparametric_eeu = 0, remainder =0, pdf_val;
  int i, y, current;
  int infinite_expectation = 0;
  GPModel current_gp;
  current_gp.u = gpt->u;
  current_gp.p = gpt->p;
  for(y=0;y<gpt->size;++y){
    if(gpt->xi[y]>=1){
      infinite_expectation = 1;
    }
  }
  if(infinite_expectation==1){
    eeu = -1.0;
  }else{
    for(i=0;i<net_demand->size;++i){
      current = (int) net_demand->value[i];
      if(current < gpt->u){
        eeu += current*gen_cdf(F,current-1) - cumulative_expectation(F,current-1);
        //printf("current: %d, eeu: %f\n",current,eeu);
      }
    }
    eeu /= net_demand->size;
    for(y=(int) ceil(gpt->u);y<F->max+1;++y){
      pdf_val = bayesian_semiparametric_net_demand_pdf(gpt,net_demand,y);
      semiparametric_eeu += pdf_val*(y*gen_cdf(F,y-1) - cumulative_expectation(F,y-1));
    }
    for(i=0;i<gpt->size;++i){
      current_gp.xi = gpt->xi[i];
      current_gp.sigma = gpt->sigma[i]; 
      remainder += (1.0 - semiparametric_net_demand_cdf(&current_gp,net_demand,F->max))*(F->max + current_gp.sigma/(1.0-current_gp.xi));
      //genpar_expectation += gpt->u + gpt->sigma[i]/(1-gpt->xi[i]);
    }
    //genpar_expectation = (1-gpt->p)*genpar_expectation/gpt->size;
    remainder /= gpt->size;
    eeu += semiparametric_eeu + remainder - cumulative_expectation(F,F->max)*(1-bayesian_semiparametric_net_demand_cdf(gpt,net_demand,F->max));
  }
  return eeu;
}*/


void bayesian_semiparametric_cvar_trace(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F, DoubleVector* output, int q){

  double constant_factor = 0, pdf_val;

  int upper_bound = q + 1, sum_split =  F->max+q-1, i, y, current;
  
  int infinite_expectation = 0;

  GPModel current_gp;
  current_gp.u = gpt->u;
  current_gp.p = gpt->p;

  for(y=0;y<gpt->size;++y){
    if(gpt->xi[y]>=1){
      infinite_expectation = 1;
    }
  }
  if(infinite_expectation==1){
    output->value[0] = -1.0;
  }else{

    for(i=0;i<net_demand->size;++i){
      current = (int) net_demand->value[i];
      if(current < gpt->u){
        constant_factor += current*gen_cdf(F,current-upper_bound) - cumulative_expectation(F,current-upper_bound);
        //printf("current: %d, cvar: %f\n",current,cvar);
      }
    }
    constant_factor /= net_demand->size;

    for(i=0;i<gpt->size;i++){
      current_gp.xi = gpt->xi[i];
      current_gp.sigma = gpt->sigma[i]; 

      output->value[i] = constant_factor;

      for(y=(int) ceil(gpt->u);y<=sum_split;++y){

        pdf_val = semiparametric_net_demand_pdf(&current_gp,net_demand,y);

        output->value[i] += pdf_val*(y*gen_cdf(F,y-upper_bound) - cumulative_expectation(F,y-upper_bound));
      }

      output->value[i] += (1.0 - semiparametric_net_demand_cdf(&current_gp,net_demand,sum_split))*(sum_split + 1 + current_gp.sigma/(1.0-current_gp.xi));

      output->value[i] -= cumulative_expectation(F,F->max)*(1-semiparametric_net_demand_cdf(&current_gp,net_demand,sum_split));

    }

  }

}


/*double bayesian_semiparametric_cvar(PosteriorGPTrace* gpt, IntVector* net_demand, DiscreteDistribution* F, int q){
  double eeu = 0, semiparametric_eeu = 0, remainder =0, pdf_val;
  int upper_bound = q + 1, sum_split =  F->max+q-1, i, y, current;
  
  int infinite_expectation = 0;
  GPModel current_gp;
  current_gp.u = gpt->u;
  current_gp.p = gpt->p;
  for(y=0;y<gpt->size;++y){
    if(gpt->xi[y]>=1){
      infinite_expectation = 1;
    }
  }
  if(infinite_expectation==1){
    eeu = -1.0;
  }else{
    for(i=0;i<net_demand->size;++i){
      current = (int) net_demand->value[i];
      if(current < gpt->u){
        eeu += current*gen_cdf(F,current-upper_bound) - cumulative_expectation(F,current-upper_bound);
        //printf("current: %d, eeu: %f\n",current,eeu);
      }
    }
    eeu /= net_demand->size;
    for(y=(int) ceil(gpt->u);y<=sum_split;++y){
      pdf_val = bayesian_semiparametric_net_demand_pdf(gpt,net_demand,y);
      semiparametric_eeu += pdf_val*(y*gen_cdf(F,y-upper_bound) - cumulative_expectation(F,y-upper_bound));
    }
    for(i=0;i<gpt->size;++i){
      current_gp.xi = gpt->xi[i];
      current_gp.sigma = gpt->sigma[i]; 
      remainder += (1.0 - semiparametric_net_demand_cdf(&current_gp,net_demand,sum_split))*(sum_split + 1 + current_gp.sigma/(1.0-current_gp.xi));
      //genpar_expectation += gpt->u + gpt->sigma[i]/(1-gpt->xi[i]);
    }
    //genpar_expectation = (1-gpt->p)*genpar_expectation/gpt->size;
    remainder /= gpt->size;
    eeu += semiparametric_eeu + remainder - cumulative_expectation(F,F->max)*(1-bayesian_semiparametric_net_demand_cdf(gpt,net_demand,sum_split));
  }
  return eeu;
}
*/













// ************** Python interfaces

void get_discrete_dist_from_py_objs(DiscreteDistribution* F, double* cdf, double* expectation, int min, int max){
  F->cdf = cdf;
  F->expectation = expectation;
  F->max = max;
  F->min = min;
}

void get_int_vector_from_py_objs(IntVector* vector, int* value, int size){

  vector -> value = value;
  vector ->size = size;
}

void get_double_vector_from_py_objs(DoubleVector* vector, double* value, int size){

  vector -> value = value;
  vector ->size = size;
}

void get_gp_from_py_objs(GPModel* gp, double xi, double sigma, double u, double p){
  gp->sigma = sigma;
  gp->u = u;
  gp->xi = xi;
  gp->p = p;
}

void get_gpt_from_py_objs(PosteriorGPTrace* gpt, double* xi, double* sigma, double u, double p, int size){
  gpt->sigma = sigma;
  gpt->u = u;
  gpt->xi = xi;
  gpt->p = p;
  gpt->size = size;
}

double empirical_power_margin_cdf_py_interface(
  int x, 
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals, 
  double* gen_cdf){

  DiscreteDistribution F;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_cdf, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);

  return empirical_power_margin_cdf(&F,&net_demand,x);

}

double empirical_net_demand_cdf_py_interface(
  double x,
  int nd_length,
  int* nd_vals){

  IntVector net_demand;

  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);

  return empirical_net_demand_cdf(&net_demand,x);
}

double semiparametric_power_margin_cdf_py_interface(
  int x,
  double u,
  double p,
  double sigma,
  double xi,
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals,
  double* gen_cdf
  ){

  DiscreteDistribution F;
  GPModel gp;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_cdf, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);
  get_gp_from_py_objs(&gp,xi,sigma,u,p);

  return semiparametric_power_margin_cdf(&gp, &net_demand, &F, x);
}

void bayesian_semiparametric_power_margin_cdf_trace_py_interface(
  int x,
  double u,
  double p,
  int n_posterior,
  double* sigma,
  double* xi,
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals,
  double* gen_cdf,
  double* py_output){

  PosteriorGPTrace gpt;
  DiscreteDistribution F;
  IntVector net_demand;
  DoubleVector output;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_cdf, gen_min, gen_max);
  get_double_vector_from_py_objs(&output,py_output,n_posterior);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);
  get_gpt_from_py_objs(&gpt, xi, sigma, u, p, n_posterior);

  bayesian_semiparametric_power_margin_cdf_trace(&gpt, &net_demand, &F, &output, x);
}

double empirical_cvar_py_interface(
  int q,
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals, 
  double* gen_cdf,
  double* gen_expectation){

  DiscreteDistribution F;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_expectation, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);

  return empirical_cvar(&F,&net_demand,q);

}

double semiparametric_cvar_py_interface(
  int q,
  double u,
  double p,
  double sigma,
  double xi,
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals,
  double* gen_cdf,
  double* gen_expectation){

  DiscreteDistribution F;
  GPModel gp;
  IntVector net_demand;

  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_expectation, gen_min, gen_max);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);
  get_gp_from_py_objs(&gp,xi,sigma,u,p);

  return semiparametric_cvar(&gp, &net_demand, &F,q);
}

void bayesian_semiparametric_cvar_trace_py_interface(
  int q,
  double u,
  double p,
  int n_posterior,
  double *sigma,
  double *xi,
  int nd_length,
  int gen_min,
  int gen_max,
  int* nd_vals,
  double* gen_cdf,
  double* gen_expectation,
  double* py_output){

  PosteriorGPTrace gpt;
  DiscreteDistribution F;
  IntVector net_demand;
  DoubleVector output;


  get_discrete_dist_from_py_objs(&F, gen_cdf, gen_expectation, gen_min, gen_max);
  get_double_vector_from_py_objs(&output,py_output,n_posterior);
  get_int_vector_from_py_objs(&net_demand,nd_vals,nd_length);
  get_gpt_from_py_objs(&gpt, xi, sigma, u, p, n_posterior);

  bayesian_semiparametric_cvar_trace(&gpt, &net_demand, &F, &output, q);

}