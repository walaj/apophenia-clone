#include <apop.h>

//Your processes are probably a bit more complex.
double process_one(gsl_rng *r){
    return gsl_rng_uniform(r) * gsl_rng_uniform(r) ;
}

double process_two(gsl_rng *r){
    return gsl_rng_uniform(r);
}

int main(){
    gsl_rng *r = apop_rng_alloc(123);

    //create the database and the data table.
    apop_db_open("runs.db");
    apop_table_exists("samples", 'd'); //If the table already exists, delete it.
    apop_query("create table samples(iteration, process, value); begin;");

    //populate the data table with runs.
    for (int i=0; i<1000; i++){
        double p1 = process_one(r);
        double p2 = process_two(r);
        apop_query("insert into samples values(%i, %i, %g);", i, 1, p1);
        apop_query("insert into samples values(%i, %i, %g);", i, 2, p2);
    }
    apop_query("commit;"); //the begin-commit wrapper saves writes to the drive.

    //pull the data from the database, converting it into a table along the way. 
    apop_data *m  = apop_db_to_crosstab("samples", "iteration","process", "value");

    gsl_vector *v1 = Apop_cv(m, 0); //get vector views of the two table columns.
    gsl_vector *v2 = Apop_cv(m, 1);

    //Output a table of means/variances, and t-test results.
    printf("\t   mean\t\t   var\n");
    printf("process 1: %f\t%f\n", apop_mean(v1), apop_var(v1));
    printf("process 2: %f\t%f\n\n", apop_mean(v2), apop_var(v2));
    printf("t test\n");
    apop_data_show(apop_t_test(v1, v2));
    apop_data_print(m, "the_data.txt");
}
