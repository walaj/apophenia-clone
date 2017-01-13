#include <apop.h>

int main(){
    size_t ct = 5e4;

    //set up the model & params
    apop_data *params = apop_data_falloc((2,2,2), 8,  1, 0.5,
                                                  2,  0.5, 1);
    apop_model *pvm = apop_model_copy(apop_multivariate_normal);
    pvm->parameters = apop_data_copy(params);
    pvm->dsize = 2;
    apop_data *d = apop_model_draws(pvm, ct);

    //set up and estimate a model with fixed covariance matrix but free means
    gsl_vector_set_all(pvm->parameters->vector, GSL_NAN);
    apop_model *mep1 = apop_model_fix_params(pvm);
    apop_model *e1 = apop_estimate(d, mep1);
    
    //compare results
    printf("original params: ");
    apop_vector_print(params->vector);
    printf("estimated params: ");
    apop_vector_print(e1->parameters->vector);
    assert(apop_vector_distance(params->vector, e1->parameters->vector)<1e-2); 
}
