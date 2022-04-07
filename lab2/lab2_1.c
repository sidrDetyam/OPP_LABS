#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <omp.h>


void init(double* matrix, double* b, double* x, int n){

    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            matrix[i*n + j] = 0;
        }
    }

    for (int i = 0; i < n; ++i) {
        x[i] = 0;
        b[i] = 2*i+1;
        matrix[i*n + i] = 2;
    }
}


double calculate(const double* matrix, const double* b, const double* x, double* new_x,
                 int n, double tao){

    double err = 0;

    #pragma omp parallel for reduction(+:err)
    for (int i = 0; i < n; ++i) {

        double s = -b[i];
        for (int j = 0; j < n; ++j) {
            s += matrix[i * n + j] * x[j];
        }

        err += s*s; //reduce
        new_x[i] = x[i] - tao * s; //not concurrent
    }

    return sqrt(err);
}


double norm(const double* v, int n){
    
    double res = 0;
    
    #pragma omp parallel for reduction(+:res)
    for(int i=0; i<n; ++i){
        res += v[i]*v[i];
    }

    return sqrt(res);
}


int main(){

    int n = 1024;
    double eps = 0.000001;
    double tao = 0.001;

    double* matrix = (double*) malloc(sizeof(double) * n * n);
    double* b = (double*) malloc(sizeof(double) * n);
    double* x = (double*) malloc(sizeof(double) * n);
    double* new_x = (double*) malloc(sizeof(double) * n);
    if(matrix==NULL || b==NULL || x==NULL || new_x==NULL){
        perror("Out of memory");
        exit(1);
    }
    init(matrix, b, x, n);

    printf("Count of threads: %d", omp_get_num_threads());
    int is_calculate_done = 0;
    double b_norm = norm(b, n);
    while(!is_calculate_done){
        double err = calculate(matrix, b, x, new_x, n, tao);
        double* tmp = x;
        x = new_x;
        new_x = tmp; 

        if(err/b_norm < eps){
            is_calculate_done = 1;
        }
    }


    free(matrix);
    free(x);
    free(new_x);
    free(b);

    return 0;
}