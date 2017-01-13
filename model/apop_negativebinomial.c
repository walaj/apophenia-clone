/* The negative binomial distribution
  
 Modified by Jeremiah Wala (2016) from apop_poisson.c
    Copyright (c) 2006--2007, 2010 by Ben Klemens.  Licensed under the GPLv2; see COPYING.  
  
\amodel apop_negativebinomial

Negative Binomial: \f$p(k|p,r) = \nchoose{k+r-1, k} (1-p)^r p^k
Poission: \f$p(k|mu) = {\mu^k \over k!} \exp(-\mu). \f$

\adoc    Input_format One scalar observation per row (in the \c matrix or \c vector).  
##\adoc    Parameter_format  One parameter, the zeroth element of the vector (<tt>double mu = apop_data_get(estimated_model->parameters)</tt>).
##\adoc    settings   \ref apop_parts_wanted_settings, for the \c .want_cov element.  */

#include "apop_internal.h"

static double psi_apply(double in, void* param) {
  double r = *(double*)param;
  return gsl_sf_psi(in + r);
}

static double apply_me_negb(double x, void *in){
    if (x < 0) return -INFINITY;
    if ((x - (int)x) > 1e-4) return -INFINITY;
    gsl_vector * params = (gsl_vector*)in;
    double ln_p = log(1 - gsl_vector_get(params, 0));
    double r = gsl_vector_get(params, 1);

    double return_val = x==0 ? 0 : gsl_sf_lngamma(x + r) - gsl_sf_lngamma(x+1) + x * ln_p;
    return return_val;
}

static long double negativebinomial_log_likelihood(apop_data * d, apop_model * p) {  
  Nullcheck_mpd(d, p, GSL_NAN)
  Get_vmsizes(d) //tsize
  gsl_vector* params = p->parameters->vector;
  double negb_p = gsl_vector_get(params, 0);
  double negb_r = gsl_vector_get(params, 1);
  double ll = apop_map_sum(d, .fn_dp = apply_me_negb, .param=params);
  double return_val = ll - tsize * gsl_sf_lngamma(negb_r) + tsize * negb_r * log(negb_p);
  //static size_t ccc = 0;
  //if (++ccc % 10 == 0)
  //  fprintf(stderr, "LL %f num-data points %d p %f r %f iteration %d\n", return_val, tsize, negb_p, negb_r, ccc);
  return return_val; 
}

static long double positive_r_beta_p_constraint(apop_data *returned_beta, apop_model *v) {

  static apop_data *constraint = NULL;
  //p and r. p {0,1} and r > 0
  if (!constraint) constraint= apop_data_falloc((3,3,2), 0,   1,  0,   //0  <  p
  						         -1,  -1,  0,  //-1 < -p
  						         0,   0,  1);  //0  <    + r (r > 0)
  return apop_linear_constraint(v->parameters->vector, constraint, 1e-4);
}

// derivative of log likelihood wrt params p and r
static void negativebinomial_dlog_likelihood(apop_data* d, gsl_vector *gradient, apop_model* p) {
  Get_vmsizes(d) //tsize
    Nullcheck_mpd(d, p, )
    
  gsl_vector* params = p->parameters->vector;  
  double negb_p = gsl_vector_get(params, 0);
  double negb_r = gsl_vector_get(params, 1);
  gsl_vector *data = d->vector;
 
  double dsum_p = apop_vector_sum(data);
  double dsum_r = apop_map_sum(d, .fn_dp = psi_apply, .param=&negb_r);

  // d(log(gamma(x))) is gamma'(x)/gamma(x). gamma'(x) = psi_0(x)gamma(x)
  // therefore, d(log(gamma(x))) = psi_0(x)
  double negb_p_dlog = dsum_p / (1-negb_p) - tsize * negb_r / (negb_p);
  double negb_r_dlog = dsum_r - tsize * gsl_sf_psi(negb_r) + tsize * log(negb_p);
    
  gsl_vector_set(gradient, 0, negb_p_dlog);
  gsl_vector_set(gradient, 1, negb_r_dlog);
}

static int negativebinomial_rng(double* out, gsl_rng* r, apop_model* p) {
  double negb_p = gsl_vector_get(p->parameters->vector, 0);
  double negb_r = gsl_vector_get(p->parameters->vector, 1);

  *out = gsl_ran_negative_binomial(r, negb_p, negb_r);
  return 0;
}

static void negativebinomial_prep(apop_data *data, apop_model *m){
  apop_score_vtable_add(negativebinomial_dlog_likelihood, apop_negativebinomial);
  apop_model_clear(data, m);

  m->parameters = apop_data_alloc(2, 0, 0, NULL, NULL, NULL); // NB has two parameters, store in vec
  gsl_vector_set(m->parameters->vector, 0, 0.5); //give initial value to p [0,1]
  gsl_vector_set(m->parameters->vector, 1, 10); //give initial value to r [> 0]
}

static void nb_estimate(apop_data * data, apop_model * m) {

  double vmean = apop_mean(data->vector);
  double vvar = apop_var(data->vector);

  // set an initial guess based on mean and variance of data
  // p = mean / var       r = mean^2 / (var - mean)
  double p = vmean / vvar;
  double r = gsl_pow_2(vmean)/(vvar - vmean);
  fprintf(stderr, "Initial estimate -- p: %f  r: %f  mean: %f  var: %f\n", p, r, vmean, vvar);
  Apop_model_add_group(m, apop_mle, .verbose=1, .starting_pt=(double[]){p, r});
  apop_maximum_likelihood(data, m); // call the default 
}

apop_model *apop_negativebinomial = &(apop_model){"Negative binomial distribution", 1, 0, 0, .dsize=1,
     .log_likelihood = negativebinomial_log_likelihood, 
						  .estimate = nb_estimate, 						  
     .prep = negativebinomial_prep, .constraint = positive_r_beta_p_constraint, 
     .draw = negativebinomial_rng};
