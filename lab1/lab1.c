
#include <mpi.h>
#include <mpe.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double calculate(double* matrix, double* x, double* b, double* new_x,
                 int n, double tao, int cnt, int first){

    double res = 0;

    for (int i = 0; i < cnt; ++i) {

        double s = -b[i];
        for (int j = 0; j < n; ++j) {
            s += matrix[i * n + j] * x[j];
        }

        res += s*s;
        new_x[i] = x[first + i] - tao * s;
    }

    return res;
}


int count_of_lines(int n, int rank, int size){
    return n/size + (rank < n%size);
}


int first_line(int n, int rank, int size){

    int res = 0;
    for(int i=0; i<rank; ++i){
        res += count_of_lines(n, i, size);
    }

    return res;
}


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


#define MESSAGE_TAG 666


int main(int argc,char **argv) {

    MPI_Init(&argc, &argv);
    MPE_Init_log();
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n;
    double eps;
    double tao;

    if (rank == 0) {
        n = 4096;
        eps = 0.000001;
        tao = 0.001;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tao, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int* lines_count = (int*) malloc(sizeof(int) * size);
    int* first_lines = (int*) malloc(sizeof(int) * size);
    for(int i=0; i<size; ++i){
        lines_count[i] = count_of_lines(n, i, size);
        first_lines[i] = first_line(n, i, size);
    }

    double *matrix = (double*) malloc(sizeof(double) * n * (rank==0? n : lines_count[rank]));
    double *x = (double*) malloc(sizeof(double) * n);
    double *b = (double*) malloc(sizeof(double) * (rank==0? n : lines_count[rank]));

    int buff_size = sizeof(double) * (lines_count[0]*n + n + lines_count[0]);
    char* buff = (char*) malloc(buff_size);

    if(rank==0){
        init(matrix, b, x, n);

        for(int i=1; i<size; ++i) {

            int pos = 0;
            MPI_Pack(matrix + n * first_lines[i], n * lines_count[i], MPI_DOUBLE, buff, buff_size, &pos, MPI_COMM_WORLD);
            MPI_Pack(x, n, MPI_DOUBLE, buff, buff_size, &pos, MPI_COMM_WORLD);
            MPI_Pack(b + first_lines[i], lines_count[i], MPI_DOUBLE, buff, buff_size, &pos, MPI_COMM_WORLD);

            MPI_Send(buff, buff_size, MPI_BYTE, i, MESSAGE_TAG, MPI_COMM_WORLD);
        }
    }
    else{

        MPI_Recv(buff, buff_size, MPI_BYTE, 0, MESSAGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int pos = 0;
        MPI_Unpack(buff, buff_size, &pos, matrix, n * lines_count[rank], MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(buff, buff_size, &pos, x, n, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(buff, buff_size, &pos, b, lines_count[rank], MPI_DOUBLE, MPI_COMM_WORLD);
    }
    free(buff);


    double b_norm_sqr = 0;
    if(rank == 0) {
        for (int i = 0; i < n; ++i) {
            b_norm_sqr += b[i] * b[i];
        }
    }

    double* new_x = (double*) malloc(sizeof(double) * lines_count[rank]);
    double* d;
    if(rank==0){
        d = (double*) malloc(sizeof(double)*size);
    }

    int init_start = MPE_Log_get_event_number();
    int init_end = MPE_Log_get_event_number();
    MPE_Describe_state(init_start, init_end, "init", "blue");

    int flag = 1;
    while(flag){

        MPE_Log_event(init_start, 0, NULL);
        double dd = calculate(matrix, x, b, new_x, n, tao, lines_count[rank], first_lines[rank]);
        MPE_Log_event(init_end, 0, NULL);

        MPI_Allgatherv(new_x, lines_count[rank], MPI_DOUBLE, x, lines_count, first_lines, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Gather(&dd, 1, MPI_DOUBLE, d, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        if(rank==0) {
            double s = 0;
            for (int i = 0; i < size; ++i) {
                s += d[i];
            }
            flag = sqrt(s/b_norm_sqr) > eps;
        }

        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }


    // if(rank==0){
    //     for(int i=0; i<n; ++i){
    //         printf("%.3f ", x[i]);
    //     }
    //     printf("\n");
    // }

    if(rank==0){
        fprintf(stderr, "\n============================\n");
        fprintf(stderr, "Count of MPI process: %d\n", size);
    }

    free(matrix);
    free(x);
    free(new_x);
    free(b);
    if(rank==0) {
        free(d);
    }
    free(lines_count);
    free(first_lines);

	MPI_Finalize();
	return 0;
}
