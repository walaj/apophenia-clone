/* Test three methods for the entries in the conjugate prior table:

--using the conjugate priors
--using Metropolis-Hastings
--using draws from the prior

There are many complications.

Selecting the method:
The second method happens if the lookup in the update vtable fails, which is why the likelihood gets overwritten with fake_ll.

The third method happens when there is no p or log_likelihood at all.

Comparing to the `truth':
The parameters from updating on the c.p. table are very sharp---it is unlikely that
a search in the time tolerable for a unit test will get anywhere near them. However,
the CDF should still be about the same regardless of method. The deciles function goes through many slices of the CDF (more than ten as of this writing), and checks that the M-H CDF or the draw-based CDF are within 9% of the c.p. CDF.

Generating something for the CDF:
The apop_pmf method is limited at the momemt: if a point is not in the CMF, then
apop_cdf(intermeidate_point, the_pmf) returns zero. The rationale is that the data is
not necessarily ordered, but that seriously needs to be revised. In the mean time, I estimate a parameterized distribution from the PMF, and use that for the deciles tests.

Estimating from a weighted CMF:
None of the estimation routines use the weights in a data set. So, the
apop_data_pmf_expand function replicates the data according to the weights.
*/
#include <apop.h>
#include <assert.h>

long double fake_ll (apop_data *d, apop_model *m){
    return ((apop_model*)(m->more))->log_likelihood(d, m);
}

apop_data *apop_data_pmf_expand(apop_data *in, int factor){
    apop_data *expanded = apop_data_alloc();
    apop_vector_normalize(in->weights);
    for (int i=0; i< in->weights->size;i++){
        int wt = gsl_vector_get(in->weights, i)* factor;
        if (wt){
            apop_data *next = apop_data_alloc(wt);
            gsl_vector_set_all(next->vector, apop_data_get(in, i));
            apop_data_stack(expanded, next, .inplace='y');
        }
    }
    if (expanded->vector) return expanded;
    else return NULL;
}

void deciles(apop_model *m1, apop_model *m2, double max){
    double width = 30;
    for (double i=0; i< max; i+=1/width){
        apop_data *x = apop_data_falloc((1), i);
        double L = apop_cdf(x, m1);
        double R = apop_cdf(x, m2);
        assert(fabs(L-R) < 0.18); //wide, I know.
    }
}

void betabinom(){
    apop_model *beta = apop_model_set_parameters(apop_beta, 10, 5);

    apop_model *drawfrom = apop_model_copy(apop_multinomial);
    drawfrom->parameters = apop_data_falloc((2), 30, .4);
    drawfrom->dsize = 2;
    int draw_ct = 80;
    apop_data *draws = apop_model_draws(drawfrom, draw_ct);

    apop_model *betaup = apop_update(draws, beta, apop_binomial);
    apop_model_show(betaup);

    beta->more = apop_beta;
    beta->log_likelihood = fake_ll;
    apop_model *bi = apop_model_fix_params(apop_model_set_parameters(apop_binomial, 30, NAN));
    apop_model *upd = apop_update(draws, beta, bi);
    apop_model *betaed = apop_estimate(upd->data, apop_beta);
    deciles(betaed, betaup, 1);

    //now via a non-MCMC updating routine: draw from fixed prior, evaluate using likelihood.
    beta->log_likelihood = NULL;
    apop_model *upd_r = apop_update(draws, beta, bi);
    betaed = apop_estimate(apop_data_pmf_expand(upd_r->data, 2000), apop_beta);
    deciles(betaed, betaup, 1);
}

void gammafish(){
    printf("gamma/poisson\n");
    apop_model *gamma = apop_model_set_parameters(apop_gamma, 1.5, 2.2);

    apop_model *drawfrom = apop_model_set_parameters(apop_poisson, 3.1);
    int draw_ct = 90;
    apop_data *draws = apop_model_draws(drawfrom, draw_ct);

    apop_model *gammaup = apop_update(draws, gamma, apop_poisson);
    apop_model_show(gammaup);

    gamma->more = apop_gamma;
    gamma->log_likelihood = fake_ll;
    Apop_settings_add_group(gamma, apop_mcmc, .burnin=.1, .periods=1e4);
    apop_model *upd = apop_update(draws, gamma, apop_poisson);
    apop_model *gammafied = apop_estimate(upd->data, apop_gamma);
    deciles(gammafied, gammaup, 5);

    gamma->log_likelihood = NULL;
    apop_model *upd_r = apop_update(draws, gamma, apop_poisson);
    apop_model *gammafied2 = apop_estimate(apop_data_pmf_expand(upd_r->data, 2000), apop_gamma);
    deciles(gammafied2, gammaup, 5);
    deciles(gammafied, gammafied2, 5);
}

void make_draws(){
    apop_model *multinom = apop_model_copy(apop_multivariate_normal);
    multinom->parameters = apop_data_falloc((2, 2, 2), 
                                        1,  1, .1,
                                        8, .1,  1);
    multinom->dsize = 2;

    apop_model *d1 = apop_estimate(apop_model_draws(multinom), apop_multivariate_normal);
    for (int i=0; i< 2; i++)
        for (int j=-1; j< 2; j++)
            assert(fabs(apop_data_get(multinom->parameters, i, j)
                    - apop_data_get(d1->parameters, i, j)) < .25);
    multinom->draw = NULL; //so draw via MCMC
    apop_model *d2 = apop_estimate(apop_model_draws(multinom, 10000), apop_multivariate_normal);
    for (int i=0; i< 2; i++)
        for (int j=-1; j< 2; j++)
            assert(fabs(apop_data_get(multinom->parameters, i, j)
                    - apop_data_get(d2->parameters, i, j)) < .25);
}

int main(){
    make_draws();
    betabinom();
    gammafish();
}
