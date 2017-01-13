#define _GNU_SOURCE
#include <apop.h>

#define printdata(dataset)           \
        printf("\n-----------\n\n"); \
        apop_data_print(dataset);   

int main(){
    apop_data *d = apop_text_alloc(apop_data_alloc(6), 6, 1);
    apop_data_fill(d,   1,   2,   3,   3,   1,   2);
    apop_text_fill(d,  "A", "A", "A", "A", "A", "B");

    asprintf(&d->names->title, "Original data set");
    printdata(d);

        //binned, where bin ends are equidistant but not necessarily in the data
    apop_data *binned = apop_data_to_bins(d);
    asprintf(&binned->names->title, "Post binning");
    printdata(binned);
    assert(fabs(//equal distance between bins
              (apop_data_get(binned, 1) - apop_data_get(binned, 0))
            - (apop_data_get(binned, 2) - apop_data_get(binned, 1))) < 1e-5);

        //compressed, where the data is as in the original, but weights 
        //are redone to accommodate repeated observations.
    apop_data_pmf_compress(d);
    asprintf(&d->names->title, "Post compression");
    printdata(d);
    assert(apop_sum(d->weights)==6);

    apop_model *d_as_pmf = apop_estimate(d, apop_pmf);
    apop_data *firstrow = Apop_r(d, 0); //1A
    assert(fabs(apop_p(firstrow, d_as_pmf) - 2./6 < 1e-5));
}
