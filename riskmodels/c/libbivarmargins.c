#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include "mtwist-1.5/mtwist.h"

#include "libbivarmargins.h"

// documentation is in .h files 

int int_min(int a, int b){
  return (a > b ) ? b : a;
}

int int_max(int a, int b){
  return (a > b ) ? a : b;
}

int get_element(IntMatrix* m, int i, int j){
  return m->value[i*m->n_cols + j];
}

void set_element(IntMatrix* m, int i, int j, int x){
  m->value[i*m->n_cols + j] = x;
}

double sign(double x){
  if(x > 0){
    return 1.0;
  }else if(x < 0){
    return -1.0;
  }else{
    return 0.0;
  }
}

double bivariate_gen_cdf(BivariateDiscreteDistribution* F, int x, int y){
  return gen_cdf(F->x,x)*gen_cdf(F->y,y);
}

double bivariate_gen_pdf(BivariateDiscreteDistribution* F, int x, int y){
  return bivariate_gen_cdf(F,x,y) + bivariate_gen_cdf(F,x-1,y-1) - bivariate_gen_cdf(F,x-1,y) - bivariate_gen_cdf(F,x,y-1);
}

double triangle_prob(BivariateDiscreteDistribution* F,int x,int y, int triangle_length){
  // by interior lattice below, I mean points not in any triangle side
  // except maybe the hypotenuse

  double val, gen1_pdf, gen2_pdf, square_prob;

  int l = (int) floor(triangle_length/2);

  int displaced_righthand_x = x + l, displaced_righthand_y = y;
  int displaced_upper_x = x, displaced_upper_y = y + l;
  //#int displaced_triangle_origin_h[2] = {origin_x + l, origin_y};
  //int displaced_triangle_origin_v[2] = {origin_x, origin_y + l};
  
  if(triangle_length <= 1){
    // interior lattice of length-1 triangle is empty
    val = 0.0;
  }else{
    // interior lattice of length 2 triangle consist in a single point
    if(triangle_length == 2){

      gen1_pdf = gen_pdf(F->x,x+1);
      gen2_pdf = gen_pdf(F->y,y+1);

      val = gen1_pdf * gen2_pdf;
    }else{

      square_prob = bivariate_gen_cdf(F,x+l,y+l) + bivariate_gen_cdf(F,x,y) - bivariate_gen_cdf(F,x,y+l) - bivariate_gen_cdf(F,x+l,y);
      
      val = square_prob +  triangle_prob(F,displaced_upper_x,displaced_upper_y,triangle_length - l) + triangle_prob(F,displaced_righthand_x,displaced_righthand_y,triangle_length - l);
    }
  }

  return val;
}


double trapezoid_prob(BivariateDiscreteDistribution* F, int ul_x, int ul_y, int width){
  // ulc = upper left
  int ur_x = ul_x + width, ur_y = ul_y - width;

  double prob = 0;

  prob =  bivariate_gen_cdf(F,ur_x,ur_y) - bivariate_gen_cdf(F,ul_x,ur_y) + triangle_prob(F,ul_x,ur_y,width);

  return prob;

}

double cond_eeu_share(BivariateDiscreteDistribution* F, ObservedData* obs, int c){

  int v1 = obs->net_demand1, v2 = obs->net_demand2, d1 = obs->demand1, d2=obs->demand2;
  double eeu = 0;
  double d1_div_d2 = ((double)d1)/d2;
  double beta0 =  (double) v1 - d1_div_d2*v2 + ((double)d1+d2)/d2*c;
  double alpha0 = (double) v1 - d1_div_d2*v2 - ((double)d1+d2)/d2*c;
  double r = ((double) d1)/(d1+d2);

  int x2, beta, alpha ;
  double gen2_pdf;

  for(x2=F->y->min;x2<v2+c;++x2){

    //EPU += -r*FX2.pdf(x2)*((x2-v1-v2)*(FX1.cdf(beta)-FX1.cdf(alpha-1)) + FX1.expectation(fro=alpha,to=beta ) )
    
    beta = (int) double_min(beta0 + d1_div_d2*x2,v1+v2-x2);
    alpha = (int) ceil(alpha0 + d1_div_d2*x2);
    //gen2_pdf = (get_gen_array_val(x2,gen2_cdf_array,min_gen2,max_gen2) - get_gen_array_val(x2-1,gen2_cdf_array,min_gen2,max_gen2));
    gen2_pdf = gen_pdf(F->y,x2);
    eeu += -r*gen2_pdf*((x2-v1-v2)*(gen_cdf(F->x,beta) - gen_cdf(F->x,alpha-1)) + cumulative_expectation(F->x,beta) - cumulative_expectation(F->x,alpha-1));
  }

  for(x2=F->y->min;x2<v2-c;++x2){

    beta = (int) floor(beta0 + d1_div_d2*x2);

    //EPU += FX2.pdf(x2)*((v1+c)*(FX1.cdf(v1+c)-FX1.cdf(beta))-FX1.expectation(fro=beta+1,to=v1+c))
    gen2_pdf = gen_pdf(F->y,x2);
    eeu += gen2_pdf*((v1+c)*(gen_cdf(F->x,v1+c) - gen_cdf(F->x,beta)) - (cumulative_expectation(F->x,v1+c) - cumulative_expectation(F->x,beta)));
  }

  for(x2=(int) ceil(v2+c-((double)d2)/d1*(v1-c));x2<v2+c;++x2){
    alpha = (int) ceil(alpha0 + d1_div_d2*x2);

    //EPU += FX2.pdf(x2)*((v1-c)*FX1.cdf(alpha-1)-FX1.expectation(to=alpha-1))
    gen2_pdf = gen_pdf(F->y,x2);
    eeu += gen2_pdf*((v1-c)*gen_cdf(F->x,alpha-1) - cumulative_expectation(F->x,alpha-1));
  }

  if(v1-c >= F->x->min){
    //EPU += (1-FX2.cdf(v2+c-1))*((v1-c)*FX1.cdf(v1-c)-FX1.expectation(to=v1-c))
    eeu += (1-gen_cdf(F->y,v2+c))*((v1-c)*gen_cdf(F->x,v1-c)-cumulative_expectation(F->x,v1-c));
  }

  return eeu;
}

double cond_eeu_veto(BivariateDiscreteDistribution* F, ObservedData* obs, int c){

  int v1 = obs->net_demand1, v2 = obs->net_demand2;
  double eeu = 0;
  int x2;
  double gen2_pdf;

  //EPU = (1 - FX2.cdf(v2+c))*((v1-c)*FX1.cdf(v1-c) - FX1.expectation(to=v1-c)) + FX2.cdf(v2-1)*(v1*FX1.cdf(v1) - FX1.expectation(to=v1))
  eeu = (1 - gen_cdf(F->y,v2+c))*((v1-c)*gen_cdf(F->x,v1-c) - cumulative_expectation(F->x,v1-c)) + gen_cdf(F->y,v2-1)*(v1*gen_cdf(F->x,v1)-cumulative_expectation(F->x,v1));
  
  for(x2=v2;x2<v2+c+1;++x2){

    //EPU += FX2.pdf(x2) * ((v1+v2-x2)*FX1.cdf(v1+v2-x2) - FX1.expectation(to=v1+v2-x2))
    gen2_pdf = gen_pdf(F->y,x2);
    eeu += gen2_pdf*((v1+v2-x2)*gen_cdf(F->x,v1+v2-x2) - cumulative_expectation(F->x,v1+v2-x2));
  }

  return eeu;
}


void get_share_polygon_points(Polygon* p, ObservedData* obs, int x, int c, int area){

  int v1 = obs->net_demand1, v2 = obs->net_demand2, d1 = obs->demand1, d2 = obs->demand2;
  int nd1, nd2, q1, q2;
  int x1, x2, y1, y2;
  int delta;
  double res;

  if(area==1){
    nd1 = v2;
    nd2 = v1;
    q1 = d2;
    q2 = d1;
  }else{
    nd1 = v1;
    nd2 = v2;
    q1 = d1;
    q2 = d2;
  }

  if(x >= 0){
    x1 = nd1 + x;
    x2 = nd2;
    delta = c;
  }else{
    x1 = nd1 - c + x;
    // this commented line fix a woird behaviour due to rounding to 1MW:
    // as c -> inf, the risk in the two areas do not fully converge because 
    // rounding causes that some cases go unacounted for, for example: if the systemwide
    // shortfall is 1MW, a fractional portion is going to be beared by each area,
    // but rounding makes this fractional part disappear and become zero.
    

    // floor function is necessary to prune lattice correctly
    res = ((double) q2)/q1*x;
    x2 = (int) (nd2 + c + res);
    delta = 2*c;
  }
  y1 = x1 + delta;
  y2 = x2 - delta;
  if(area==1){
    p->p1->x = y2;
    p->p1->y = y1;
    p->p2->x = x2;
    p->p2->y = x1;
  }else{
    p->p1->x = x1;
    p->p1->y = x2;
    p->p2->x = y1;
    p->p2->y = y2;
  }
}

void get_veto_polygon_points(Polygon* p, ObservedData* obs, int x, int c, int area){

  int v1 = obs->net_demand1, v2 = obs->net_demand2;
  int nd1, nd2;
  int x1, x2, y1, y2;

  if(area == 1){
    nd1 = v2;
    nd2 = v1;
  }else{
    nd1 = v1;
    nd2 = v2;
  }

  if(x >= 0){
    x1 = nd1 + x;
    x2 = nd2;
  }else{
    x1 = nd1 - c + x;
    x2 = nd2 + c;
  }
  y1 = x1 + c;
  y2 = x2 - c;

  if(area == 1){
    p->p1->x = y2;
    p->p1->y = y1;
    p->p2->x = x2;
    p->p2->y = x1;
  }else{
    p->p1->x = x1;
    p->p1->y = x2;
    p->p2->x = y1;
    p->p2->y = y2;
  }
}

void get_polygons(Polygon* p1, Polygon* p2, ObservedData* obs, SimulationParameters* pars){

  if(pars->share_policy > 0){
    get_share_polygon_points(p1, obs, pars->x_bound, pars->c, 0);
    get_share_polygon_points(p2, obs, pars->y_bound, pars->c, 1);
  }else{
    get_veto_polygon_points(p1, obs, pars->x_bound, pars->c, 0);
    get_veto_polygon_points(p2, obs, pars->y_bound, pars->c, 1);
  }
}

int axis1_polygon_upper_bound(Polygon* p, int x){

  int res;
  if(x <= p->p1->x){
    res = (int) (INT_MAX/2);
  }else if(x <= p->p2->x){
    res = p->p1->y - (x - p->p1->x);
  }else{
    res = (int) (-INT_MAX/2);
  }
  return res;
}

int axis2_polygon_upper_bound(Polygon* p, int x){

  int res;
  if(x <= p->p1->x){
    res = p->p1->y;
  }else if(x <= p->p2->x){
    res = p->p1->y - (x - p->p1->x);
  }else{
    res = p->p2->y;
  }
  return res;
}

int get_bounded_quantile(DiscreteDistribution* F, int upper_bound, double u){
  int lb = F->min, i=0;
  int ub = (int) int_min(upper_bound,F->max);

  double p_lb = gen_cdf(F,lb-1);
  double box_prob = gen_cdf(F,ub);

  while(u > (gen_cdf(F,lb+i) - p_lb)/box_prob) {
    i +=1;
  }

  return lb+i;
}


int veto_flow(int m1,int m2,int c){

  int res;
  if(m1 > 0 && m2 < 0){
      res = -int_min(c,int_min(m1,-m2));
  }else if(m1 < 0 && m2 > 0){
      res = int_min(c,int_min(m2,-m1));
  }else{
      res = 0;
  }
  return res;
}

double share_flow(int m1,int m2,int d1,int d2,int c){

  double flow;
  if(m1+m2 < 0 && m1 < c && m2 < c){
    //res = min(c,double_max(-c,((float) d1)/(d1+d2)*m2 - ((float) d2)/(d1+d2)*m1));
    flow = ((float) d1)/(d1+d2)*m2 - ((float) d2)/(d1+d2)*m1;
    flow = (flow >= c ? c : (flow <= -c ? -c: flow));
  }else{
    flow = (double) veto_flow(m1,m2,c);
  }
  return flow;
}

double cond_bivariate_power_margin_cdf(BivariateDiscreteDistribution* F, Polygon* plg1, Polygon* plg2){ 
  
  double cdf_vals = 0;
  int diff = 0;
  int c1 = plg1->p2->x - plg1->p1->x, c2 = plg2->p2->x - plg2->p1->x;

  //#if P2 segment is inside marginal polygon of area 1
  if(plg2->p2->x <= plg1->p1->x || (plg2->p2->x <= plg1->p2->x && plg2->p2->y <= plg1->p1->y + plg1->p1->x - plg2->p2->x)) {

    cdf_vals = bivariate_gen_cdf(F,plg2->p1->x,plg2->p1->y) + trapezoid_prob(F,plg2->p1->x,plg2->p1->y,c2);

    //# if segment is inside lower rectangle
    if(plg2->p2->y <= plg1->p2->y){

      cdf_vals += bivariate_gen_cdf(F,plg1->p2->x,plg2->p2->y) - bivariate_gen_cdf(F,plg2->p2->x,plg2->p2->y);
     
    //# if it's in middle trapezoidal section
    }else if(plg2->p2->y <= plg1->p1->y){

      diff = plg1->p1->y - plg2->p2->y;

      cdf_vals += bivariate_gen_cdf(F,plg1->p1->x - diff,plg2->p2->y) - \
        bivariate_gen_cdf(F,plg2->p2->x,plg2->p2->y) + \
        trapezoid_prob(F,plg1->p1->x + diff,plg1->p1->y - diff,c1-diff);

    }else{
      //# if it's in uppermost rectangle

      cdf_vals += bivariate_gen_cdf(F,plg1->p1->x,plg2->p2->y) - \
      bivariate_gen_cdf(F,plg2->p2->x,plg2->p2->y) + \
      trapezoid_prob(F,plg1->p1->x,plg1->p1->y,c1);

    }
  //#if P2 segment is completely outside of marginal polygon of area 1
  }else if(plg2->p1->x >= plg1->p2->x || (plg2->p1->x >= plg1->p1->x && plg2->p1->y >= plg1->p1->y + plg1->p1->x - plg2->p1->x)){
    
    if(plg2->p1->y <= plg1->p2->y){
      //# if it's not higher than lowermost quadrant

      cdf_vals += bivariate_gen_cdf(F,plg1->p2->x, plg2->p1->y);

    }else if(plg2->p1->y <= plg1->p1->y){
      //# if it's no higher than middle trapezoid
      diff = plg1->p1->y - plg2->p1->y;

      cdf_vals += bivariate_gen_cdf(F,plg1->p1->x+diff,plg2->p1->y) + trapezoid_prob(F,plg1->p1->x + diff,plg2->p1->y,c1-diff);

    }else{
      // if it's higher than trapezoid
      cdf_vals += bivariate_gen_cdf(F,plg1->p1->x,plg2->p1->y) + trapezoid_prob(F,plg1->p1->x,plg1->p1->y,c1);
    }
    
  }else{

    if(plg2->p1->x <= plg1->p1->x){
      //# if the crossing is at the upper rectangle

      cdf_vals += bivariate_gen_cdf(F,plg2->p1->x,plg2->p1->y) + \
      trapezoid_prob(F,plg2->p1->x, plg2->p1->y,plg1->p1->x-plg2->p1->x) +\
      trapezoid_prob(F,plg1->p1->x,plg1->p1->y,c1);

    }else{
      // if crossing is at the bottom rectangle
      cdf_vals += bivariate_gen_cdf(F,plg2->p1->x,plg2->p1->y) + trapezoid_prob(F,plg2->p1->x,plg2->p1->y,plg1->p2->x-plg2->p1->x);
    }
  }

  return cdf_vals;
}

/*double bivariate_power_margin_cdf(BivariateDiscreteDistribution* F, IntMatrix* demand, IntMatrix* net_demand, SimulationParameters* pars){
  int i;
  double cdf_val = 0;
  ObservedData current_obs;
  Polygon plg1, plg2;
  for(i=0;i<demand->n_rows;++i){
    current_obs.net_demand1 = get_element(net_demand,i,0);
    current_obs.net_demand1 = get_element(net_demand,i,1);
    current_obs.demand1 = get_element(demand,i,0);
    current_obs.demand2 = get_element(demand,i,1);
    get_polygons(&plg1, &plg2, &current_obs, &pars);
    cdf_vals += cond_bivariate_power_margin_cdf(F, &plg1, &plg2);
  }
  return cdf_vals/demand->n_rows;
}*/


int get_joint_polygon_x_bound(
  BivariateDiscreteDistribution* F, 
  Polygon* plg1, 
  Polygon* plg2, 
  int intersection){

  // find where the polygon derived from P2 crosses the x axis
  int compared;
  int p2_crossing, p1_crossing;
  if(plg2->p2->y > 0){
    p2_crossing = INT_MAX/2;
  }else if(plg2->p1->y <= 0){
    p2_crossing = plg2->p1->x;
  }else{
    p2_crossing = plg2->p2->x + plg2->p2->y;
  }

  // find where the polygon derived from P1 crosses the x axis
  if(plg1->p2->y >= 0){
    p1_crossing = plg1->p2->x;
  }else if(plg1->p1->y <= 0){
    p1_crossing = plg1->p1->x;
  }else{
    p1_crossing = plg1->p2->x + plg1->p2->y;
  }

  compared = intersection > 0 ? int_min(p1_crossing,p2_crossing) : int_max(p1_crossing,p2_crossing);
  return(int_min(F->x->max,compared));

}

int get_joint_polygon_y_bound_given_x(
  BivariateDiscreteDistribution* F, 
  Polygon* plg1, 
  Polygon* plg2, 
  int x, 
  int intersection){

  int compared, p1_y_bound, p2_y_bound;

  p1_y_bound = axis1_polygon_upper_bound(plg1,x);
  p2_y_bound = axis2_polygon_upper_bound(plg2,x);

  compared = intersection > 0 ? int_min(p1_y_bound,p2_y_bound) : int_max(p1_y_bound,p2_y_bound);
  
  return(int_min(F->y->max,compared));

}

void region_simulation(
  BivariateDiscreteDistribution* F, 
  IntMatrix* net_demand,
  IntMatrix* demand,
  IntMatrix* results,
  SimulationParameters* parameters){

  mt_seed32(parameters->seed);

  int row, x1, x2, x1_ub, x2_ub, m1_s, m2_s;
  double flow, u;

  Coord p11 = {0,0}, p12 = {0,0}, p21 = {0,0}, p22 = {0,0};
  Polygon plg1 = {&p11,&p12}, plg2 = {&p21,&p22};
  ObservedData current_obs;

  //double x1_cond_cdf_array[F->x->max+1]; //not all entries will be filled, but it works
  double *x1_cond_cdf_array = (double *)malloc(F->x->max+1 * sizeof(double));

  DiscreteDistribution F_cond = {F->x->min, F->x->max, &x1_cond_cdf_array[0], F->x->expectation};

  int n_sim = 0, i, j;
  //printf("x1_cond_cdf_array: %f\n",F_cond.cdf[0]);
  // iterate over the provided rows
  for(row=0;row<net_demand->n_rows;row++){

    if(parameters->obs_weights[row] > 0){

      for(i=0;i<=F->x->max;i++){
        x1_cond_cdf_array[i] = 1;
      }
      //printf("x1_cond_cdf_array: %f\n",F_cond.cdf[0]);
      current_obs.net_demand1 = get_element(net_demand,row,0);
      current_obs.net_demand2 = get_element(net_demand,row,1);
      current_obs.demand1 = get_element(demand,row,0);
      current_obs.demand2 = get_element(demand,row,1);

      get_polygons(&plg1, &plg2, &current_obs, parameters);

      x1_ub = get_joint_polygon_x_bound(F, &plg1, &plg2, parameters->intersection);

      //x1_ub = get_joint_polygon_x_bound(P1,P2,min_gen1,max_gen1,intersection);

      for(x1=0;x1<=x1_ub;x1++){

        x2_ub = get_joint_polygon_y_bound_given_x(F, &plg1, &plg2, x1, parameters->intersection);

        x1_cond_cdf_array[x1] = bivariate_gen_cdf(F,x1,x2_ub) - bivariate_gen_cdf(F,x1-1,x2_ub);

        if(x1 > 0){
          x1_cond_cdf_array[x1] += x1_cond_cdf_array[x1-1];
        }
      }
      //printf("x1_cond_cdf_array: %f\n",F_cond.cdf[0]);

      for(j=0;j<parameters->obs_weights[row];j++){

        u = mt_drand();

        x1 = get_bounded_quantile(&F_cond,x1_ub,u);

        x2_ub = get_joint_polygon_y_bound_given_x(F, &plg1, &plg2, x1, parameters->intersection);

        u = mt_drand();

        x2 = get_bounded_quantile(F->y,x2_ub,u);
        
        m1_s = x1 - current_obs.net_demand1;
        m2_s = x2 - current_obs.net_demand2;

        if(parameters->share_policy>0){
          flow = share_flow(m1_s, m2_s, current_obs.demand1, current_obs.demand2, parameters->c);

        }else{
          flow = veto_flow(m1_s, m2_s, parameters->c);

        }

        //printf("x1: %d, x2: %d, x1_ub: %d, x2_ub:%d, nd1: %d, nd2: %d, m1_s: %d, flow: %f\n",x1, x2, x1_ub, x2_ub, current_obs.net_demand1, current_obs.net_demand2, m1_s, flow);
        set_element(results,n_sim,0, m1_s + (int) flow);
        set_element(results,n_sim,1, m2_s - (int) flow);

        n_sim++;

      }
    }
  }
  free(x1_cond_cdf_array);
}

void conditioned_simulation(
  BivariateDiscreteDistribution* F,
  IntMatrix* net_demand,
  IntMatrix* demand,
  IntMatrix* results,
  SimulationParameters* parameters){

  mt_seed32(parameters->seed);

  int row, x1, x2, m1_s, m2_s;
  double flow, u;

  Coord p11 = {0,0}, p12 = {0,0}, p21 = {0,0}, p22 = {0,0};
  Polygon plg1 = {&p11,&p12};//, plg2 = {&p21,&p22};
  
  //double cond_x2_cdf[F->y->max+1]; //not all entries will be filled, but it works
  double *cond_x2_cdf = (double *)malloc(F->y->max+1 * sizeof(double));

  DiscreteDistribution F_cond = {F->y->min, F->y->max, &cond_x2_cdf[0], F->y->expectation};
/*  F_cond->min = F->y->min;
  F_cond->max = F->y->max;
  F_cond->expectation = F->y->expectation;
  F_cond->cdf = &cond_x2_cdf[0];*/

  int n_sim = 0, j;

  ObservedData current_obs;

  for(row=0;row<net_demand->n_rows;row++){

    if(parameters->obs_weights[row]>0){

      current_obs.net_demand1 = get_element(net_demand,row,0);
      current_obs.net_demand2 = get_element(net_demand,row,1);
      current_obs.demand1 = get_element(demand,row,0);
      current_obs.demand2 = get_element(demand,row,1);

      if(parameters->share_policy > 0){
        get_share_polygon_points(&plg1,&current_obs,parameters->x_bound,parameters->c,0);
      }else{
        get_veto_polygon_points(&plg1,&current_obs,parameters->x_bound,parameters->c,0);
      }

      // initialize cond prob array
      for(x2=0;x2<=F->y->max;x2++){
        if(x2 < plg1.p2->y){
          cond_x2_cdf[x2] = bivariate_gen_pdf(F,plg1.p2->x,x2);
        }else if(x2 <= plg1.p1->y){
          cond_x2_cdf[x2] = bivariate_gen_pdf(F,plg1.p2->x - (x2 - plg1.p2->y),x2);
        }else{
          cond_x2_cdf[x2] = bivariate_gen_pdf(F,plg1.p1->x,x2);
        }
        if(x2>0){
          cond_x2_cdf[x2] += cond_x2_cdf[x2-1];
        }
      }

      for(j=0;j<parameters->obs_weights[row];j++){

        u = mt_drand();

        x2 = get_bounded_quantile(&F_cond,F_cond.max,u);
        if(x2 >= plg1.p1->y){
          x1 = plg1.p1->x;
        }else if(x2 <= plg1.p2->y){
          x1 = plg1.p2->x;
        }else{
          x1 = plg1.p1->x + (plg1.p1->y - x2);
        }
        m1_s = x1 - current_obs.net_demand1;
        m2_s = x2 - current_obs.net_demand2;

        //printf("x1: %d, m1_s: %d, nd1: %d, row: %d\n",x1,m1_s,current_obs.net_demand1,row);

        if(parameters->share_policy>0){
          flow = share_flow(m1_s, m2_s, current_obs.demand1, current_obs.demand2, parameters->c);

        }else{
          flow = veto_flow(m1_s, m2_s, parameters->c);
        }
        set_element(results,n_sim,0,m1_s + (int) flow);
        set_element(results,n_sim,1,m2_s - (int) flow);

        n_sim++;
      }

    }
  }
  free(cond_x2_cdf);
}

void bivariate_empirical_cdf(
  DoubleVector* ecdf,
  IntMatrix* X){

  int i,j;
  double iter_ecdf;
  for(i=0;i<X->n_rows;i++){
    iter_ecdf = 0.0;
    //printf("(%f,%f) larger than: \n", x[i],y[i]);
    for(j=0;j<X->n_rows;j++){
      if(get_element(X,j,0) <= get_element(X,i,0) && get_element(X,j,1) <= get_element(X,i,1)){
        iter_ecdf ++;
      }
    }
    //printf("ECDF value: %f \n", iter_ecdf);
    ecdf->value[i] = iter_ecdf/(X->n_rows+1);
  }

}



















// ***************** Interfaces to Python

void get_observed_data_from_py_objs(ObservedData* obs, int d1, int d2, int nd1, int nd2){

  obs->demand1 = d1;
  obs->demand2 = d2;
  obs->net_demand1 = nd1;
  obs->net_demand2 = nd2;
}

void get_int_matrix_from_py_objs(IntMatrix* m, int* value, int n_rows, int n_cols){

  m->value = value;
  m->n_rows = n_rows;
  m->n_cols = n_cols;
}

void get_sim_pars_from_py_objs(
  SimulationParameters* pars,
  int x_bound,
  int y_bound,
  int* obs_weights,
  int seed,
  int intersection,
  int share_policy,
  int c,
  int n){

  pars->x_bound = x_bound;
  pars->y_bound = y_bound;
  pars->obs_weights = obs_weights;
  pars->seed = seed;
  pars->intersection = intersection;
  pars->share_policy = share_policy;
  pars->c = c;
  pars->n = n;
}

double triangle_prob_py_interface(
    int origin_x,
    int origin_y,
    int triangle_length,
    int min_gen1,
    int min_gen2,
    int max_gen1,
    int max_gen2,
    double* gen1_cdf_array,
    double* gen2_cdf_array){

  DiscreteDistribution X;
  DiscreteDistribution Y;
  BivariateDiscreteDistribution F;

  get_discrete_dist_from_py_objs(&X, gen1_cdf_array, gen1_cdf_array, min_gen1, max_gen1);
  get_discrete_dist_from_py_objs(&Y, gen2_cdf_array, gen2_cdf_array, min_gen2, max_gen2);
  F.x = &X;
  F.y = &Y;

  return triangle_prob(&F, origin_x, origin_y, triangle_length);

}

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
  double* gen1_expectation){

  DiscreteDistribution X;
  DiscreteDistribution Y;
  BivariateDiscreteDistribution F;
  ObservedData obs;

  get_discrete_dist_from_py_objs(&X, gen1_cdf_array, gen1_expectation, min_gen1, max_gen1);
  get_discrete_dist_from_py_objs(&Y, gen2_cdf_array, gen2_cdf_array, min_gen2, max_gen2);
  F.x = &X;
  F.y = &Y;

  get_observed_data_from_py_objs(&obs,v1,v2,v1,v2);

  return cond_eeu_veto(&F,&obs,c);
}

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
  double* gen1_expectation){

  DiscreteDistribution X;
  DiscreteDistribution Y;
  BivariateDiscreteDistribution F;
  ObservedData obs;

  get_discrete_dist_from_py_objs(&X, gen1_cdf_array, gen1_expectation, min_gen1, max_gen1);
  get_discrete_dist_from_py_objs(&Y, gen2_cdf_array, gen2_cdf_array, min_gen2, max_gen2);
  F.x = &X;
  F.y = &Y;

  get_observed_data_from_py_objs(&obs,d1,d2,v1,v2);

  return cond_eeu_share(&F,&obs,c);

}

double trapezoid_prob_py_interface(
  int ul_x,
  int ul_y,
  int width,
  int min_gen1,
  int min_gen2,
  int max_gen1,
  int max_gen2,
  double* gen1_cdf_array,
  double* gen2_cdf_array){

  DiscreteDistribution X;
  DiscreteDistribution Y;
  BivariateDiscreteDistribution F;

  get_discrete_dist_from_py_objs(&X, gen1_cdf_array, gen1_cdf_array, min_gen1, max_gen1);
  get_discrete_dist_from_py_objs(&Y, gen2_cdf_array, gen2_cdf_array, min_gen2, max_gen2);
  F.x = &X;
  F.y = &Y;

  return trapezoid_prob(&F,ul_x,ul_y,width);

}

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
  int share_policy){

  DiscreteDistribution X;
  DiscreteDistribution Y;
  BivariateDiscreteDistribution F;

  get_discrete_dist_from_py_objs(&X, gen1_cdf_array, gen1_cdf_array, min_gen1, max_gen1);
  get_discrete_dist_from_py_objs(&Y, gen2_cdf_array, gen2_cdf_array, min_gen2, max_gen2);
  F.x = &X;
  F.y = &Y;

  IntMatrix net_demand_matrix;
  IntMatrix demand_matrix;
  IntMatrix results_matrix;

  get_int_matrix_from_py_objs(&net_demand_matrix, net_demand, n_rows, 2);
  get_int_matrix_from_py_objs(&demand_matrix, demand, n_rows, 2);
  get_int_matrix_from_py_objs(&results_matrix, simulations, n_rows, 2);

  SimulationParameters pars;

  get_sim_pars_from_py_objs(&pars,m1,m2,row_weights,seed,intersection,share_policy,c,n);

  region_simulation(&F,&net_demand_matrix,&demand_matrix,&results_matrix,&pars);

}

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
  int share_policy){

  DiscreteDistribution X;
  DiscreteDistribution Y;
  BivariateDiscreteDistribution F;

  get_discrete_dist_from_py_objs(&X, gen1_cdf_array, gen1_cdf_array, min_gen1, max_gen1);
  get_discrete_dist_from_py_objs(&Y, gen2_cdf_array, gen2_cdf_array, min_gen2, max_gen2);
  F.x = &X;
  F.y = &Y;

  IntMatrix net_demand_matrix;
  IntMatrix demand_matrix;
  IntMatrix results_matrix;

  get_int_matrix_from_py_objs(&net_demand_matrix, net_demand, n_rows, 2);
  get_int_matrix_from_py_objs(&demand_matrix, demand, n_rows, 2);
  get_int_matrix_from_py_objs(&results_matrix, simulations, n_rows, 2);

  SimulationParameters pars;

  get_sim_pars_from_py_objs(&pars,m1,m1,row_weights,seed,1,share_policy,c,n); //intersection = 1 (placeholder)

  conditioned_simulation(&F,&net_demand_matrix,&demand_matrix,&results_matrix,&pars);
}

/*void bivariate_empirical_cdf_py_interface(
  double* ecdf,
  double* X,
  int n){
  DoubleVector ecdf_vector;
  DoubleMatrix X_matrix; 
  get_int_matrix_from_py_objs(&X_matrix, X, n, 2);
  get_double_vector_from_py_objs(&ecdf_vector,ecdf,n);
  bivariate_empirical_cdf(&ecdf_vector,&X_matrix);
}*/

/*double bivariate_power_margin_cdf_py_interface(
  int min_gen1,
  int min_gen2,
  int max_gen1,
  int max_gen2,
  double* gen1_cdf_array,
  double* gen2_cdf_array,
  int* demand,
  int* net_demand,
  int n_rows,
  int m1,
  int m2,
  int c,
  int share_policy){
  int i;
  DiscreteDistribution X;
  DiscreteDistribution Y;
  BivariateDiscreteDistribution F;
  get_discrete_dist_from_py_objs(&X, gen1_cdf_array, gen1_cdf_array, min_gen1, max_gen1);
  get_discrete_dist_from_py_objs(&Y, gen2_cdf_array, gen2_cdf_array, min_gen2, max_gen2);
  F.x = &X;
  F.y = &Y;
  IntMatrix net_demand_matrix;
  IntMatrix demand_matrix;
  get_int_matrix_from_py_objs(&net_demand_matrix, net_demand, n_rows, 2);
  get_int_matrix_from_py_objs(&demand_matrix, demand, n_rows, 2);
  SimulationParameters pars;
  // valid parameters: x_bound, y_bound, share_policy and c. All other are placeholders
  get_sim_pars_from_py_objs(&pars,m1,m2,gen1_cdf_array,1,1,share_policy,c,m1);
  return bivariate_power_margin_cdf(&F, &demand, &net_demand, &pars);
}*/


double cond_bivariate_power_margin_cdf_py_interface(
  int min_gen1,
  int min_gen2,
  int max_gen1,
  int max_gen2,
  double* gen1_cdf_array,
  double* gen2_cdf_array,
  int m1,
  int m2,
  int v1,
  int v2,
  int d1,
  int d2,
  int c,
  int share_policy){

  DiscreteDistribution X;
  DiscreteDistribution Y;
  BivariateDiscreteDistribution F;

  get_discrete_dist_from_py_objs(&X, gen1_cdf_array, gen1_cdf_array, min_gen1, max_gen1);
  get_discrete_dist_from_py_objs(&Y, gen2_cdf_array, gen2_cdf_array, min_gen2, max_gen2);
  F.x = &X;
  F.y = &Y;

  ObservedData obs;
  obs.net_demand1 = v1, obs.net_demand2 = v2, obs.demand1 = d1, obs.demand2 = d2;

  SimulationParameters pars;

  int placeholder = 0;

  // valid parameters: x_bound, y_bound, share_policy and c. All others are placeholders
  get_sim_pars_from_py_objs(&pars,m1,m2,&placeholder,1,1,share_policy,c,m1);

  Coord p11 = {0,0}, p12 = {0,0}, p21 = {0,0}, p22 = {0,0};
  Polygon plg1 = {&p11,&p12}, plg2 = {&p21,&p22};

  get_polygons(&plg1, &plg2, &obs, &pars);

  return cond_bivariate_power_margin_cdf(&F, &plg1, &plg2);

}