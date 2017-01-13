/** \file */
/* The kernel density estimate (meta-)model.

Copyright (c) 2007, 2010, 2013 by Ben Klemens.  Licensed under the GPLv2; see COPYING.  
*/

#include "apop_internal.h"
#include <gsl/gsl_math.h>


/*\amodel apop_kernel_density The kernel density smoothing of a PMF or histogram.

At each point along the histogram, put a distribution (default: Normal(0,1)) on top
of the point. Sum all of these distributions to form the output distribution.

Setting up a kernel density consists of setting up a model with the base data and the
information about the kernel model around each point. This can be done using the \ref
apop_model_set_settings function to get a copy of the base \ref apop_kernel_density model
and add a \ref apop_kernel_density_settings group with the appropriate information;
see the \c main function of the example below.

\adoc Input_format  One observation per line. Each row in turn will be passed through to the elements of <tt>kernelbase</tt> and optional <tt>set_params</tt> function, so follow the format of the base model.
\adoc Parameter_format  None
\adoc Estimated_parameters None
\adoc Estimated_settings  The estimate method basically just runs
                          <tt>apop_model_add_group(your_data, apop_kernel_density);</tt>
\adoc Settings  \ref apop_kernel_density_settings, including:

\li \c data a data set, which, if  not \c NULL and \c base_pmf is \c NULL, will be converted to an \ref apop_pmf model.
\li \c base_pmf This is the preferred format for input data. It is the histogram to be smoothed.
\li \c kernelbase The kernel to use for smoothing, with all parameters set and a \c p method. Popular favorites are \ref apop_normal and \ref apop_uniform.
\li \c set_params A function that takes in a single number and the model, and sets
the parameters accordingly. The function will call this for every point in the data
set. Here is the default, which is used if this is \c NULL. It simply sets the first
element of the model's parameter vector to the input number; this is appropriate for a
Normal distribution, where we want to center the distribution on each data point in turn.

\code
static void apop_set_first_param(apop_data *in, apop_model *m){
    apop_data_set(m->parameters, .val= apop_data_get(in));
}
\endcode

See the sample code for for a Uniform[0,1] recentered around the first element of the PMF matrix.

\adoc Examples
This example sets up and uses KDEs based on Normal and Uniform distributions.

\include kernel.c
*/

static void apop_set_first_param(apop_data *in, apop_model *m){
    apop_data_set(m->parameters, .val= apop_data_get(in));
}

Apop_settings_init(apop_kernel_density, 
    //If there's a PMF associated with the model, run with it.
    //else, generate one from the data.
    Apop_varad_set(base_pmf, apop_estimate(in.base_data, apop_pmf));
    Apop_varad_set(kernel, apop_model_set_parameters(apop_normal, 0, 1));
    Apop_varad_set(set_fn, apop_set_first_param);
    out->own_pmf = !in.base_pmf;
    out->own_kernel = !in.kernel;
    if (!out->kernel->parameters) apop_prep(out->base_data, out->kernel);
)

Apop_settings_copy(apop_kernel_density,
    out->own_pmf    =
    out->own_kernel = 0;
)

Apop_settings_free(apop_kernel_density,
    if (in->own_pmf)    apop_model_free(in->base_pmf);
    if (in->own_kernel) apop_model_free(in->kernel);
)

static void apop_kernel_estimate(apop_data *d, apop_model *m){
    Nullcheck_d(d, );
    if (!apop_settings_get_group(m, apop_kernel_density))
        Apop_settings_add_group(m, apop_kernel_density, .base_data=d);
}

/* \adoc    CDF Sums the CDF to the given point of all the sub-distributions.*/
static long double kernel_cdf(apop_data *d, apop_model *m){
    Nullcheck_m(m, GSL_NAN);
    long double total = 0;
    apop_kernel_density_settings *ks = apop_settings_get_group(m, apop_kernel_density);
    apop_data *pmf_data = apop_settings_get(m, apop_kernel_density, base_pmf)->data;
    Get_vmsizes(pmf_data); //maxsize
    for (size_t k = 0; k < maxsize; k++){
        apop_data *r = Apop_r(pmf_data, k);
        double wt = r->weights ? *r->weights->data : 1;
        OMP_critical(kernel_p_cdf)
        {
        (ks->set_fn)(r, ks->kernel);
        total += apop_cdf(d, ks->kernel)*wt;
        }
    }
    long double weight = pmf_data->weights ? apop_sum(pmf_data->weights) : maxsize;
    total /= weight;
    return total;
}

static long double kernel_ll(apop_data *d, apop_model *m){
    Nullcheck_m(m, GSL_NAN);
    size_t datasize;
    {Get_vmsizes(d); datasize=maxsize;}
    apop_kernel_density_settings *ks = apop_settings_get_group(m, apop_kernel_density);
    apop_data *pmf_data = apop_settings_get(m, apop_kernel_density, base_pmf)->data;
    Get_vmsizes(pmf_data); //maxsize
    long double ll = 0;
    OMP_for_reduce(+:ll,    int i=0; i< datasize; i++){
        long double lls[maxsize];
        apop_data *datapt = Apop_r(d, i);
        for(int k=0; k< maxsize; k++){
            apop_data *r = Apop_r(pmf_data, k);
            OMP_critical(kernel_p_cdf)
            {
            (ks->set_fn)(r, ks->kernel);
            lls[k] = apop_log_likelihood(datapt, ks->kernel);
            }
        }

        //let p_m w_m be the largest value among the p_i w_is. Then
        //log (Σp_i w_i) = log(p_m w_m) + log(Σ(p_i w_i/p_m w_m).
        //This gives us a little more numeric accuracy.
        double max_ll = -INFINITY;
        double total = 0;
        #define getwt(i) (pmf_data->weights ? gsl_vector_get(pmf_data->weights, i) : 1);
        for (int i=0; i< maxsize; i++) if (lls[i]>max_ll) max_ll = lls[i];
        if (max_ll==-INFINITY) {ll=-INFINITY; continue;}
        for (int i=0; i< maxsize; i++) lls[i]-=max_ll;
        for (int i=0; i< maxsize; i++) lls[i]= exp(lls[i]) * getwt(i);
        for (int i=0; i< maxsize; i++) total += lls[i];
        ll += max_ll + log(total);
    }
    ll -= datasize * log(pmf_data->weights ? apop_sum(pmf_data->weights) : maxsize);
    return ll;
}

/* \adoc    RNG  Randomly selects a data point, then randomly draws from that sub-distribution.
 Returns 0 on success, 1 if unable to pick a sub-distribution (meaning the weights over the distributions are somehow broken), and 2 if unable to draw from the sub-distribution.
 */
static int kernel_draw(double *d, gsl_rng *r, apop_model *m){
    //randomly select a point, using the weights.
    apop_kernel_density_settings *ks = apop_settings_get_group(m, apop_kernel_density);
    apop_model *pmf = apop_settings_get(m, apop_kernel_density, base_pmf);
    apop_data *point = apop_data_alloc(1, pmf->dsize);
    Apop_stopif(apop_draw(Apop_rv(point, 0)->data, r, pmf), return 1, 0, "Unable to use the PMF over kernels to select a kernel from which to draw.");
    (ks->set_fn)(point, ks->kernel);
    //Now draw from the distribution around that point.
    Apop_stopif(apop_draw(d, r, ks->kernel), return 2, 0, "unable to draw from a single selected kernel.");
    apop_data_free(point);
    return 0;
}

apop_model *apop_kernel_density = &(apop_model){"kernel density estimate", .dsize=1,
    .estimate = apop_kernel_estimate, .log_likelihood = kernel_ll,
	.cdf=kernel_cdf, .draw=kernel_draw};
