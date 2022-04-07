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

    int is_calculate_done = 0;
    double err, b_norm=0;
    int count_threads;

    double start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        count_threads = omp_get_num_threads();

        #pragma omp for schedule(static) reduction(+:b_norm)
        for(int i=0; i<n; ++i){
            b_norm += b[i]*b[i];
        }
        
        while(!is_calculate_done){
            
            #pragma omp single
            err = 0;

            #pragma omp for schedule(static) reduction(+:err)
            for (int i = 0; i < n; ++i) {
                double s = -b[i];
                for (int j = 0; j < n; ++j) {
                    s += matrix[i * n + j] * x[j];
                }
                err += s*s; //reduce
                new_x[i] = x[i] - tao * s; //not concurrent
            }

            #pragma omp single
            {
                double* tmp = x;
                x = new_x;
                new_x = tmp; 

                if(sqrt(err)/b_norm < eps){
                    is_calculate_done = 1;
                }
            }
        }
    }
    printf("threads: %d\ntime: %lf\n", count_threads, omp_get_wtime() - start_time);

    free(matrix);
    free(x);
    free(new_x);
    free(b);

    return 0;
}