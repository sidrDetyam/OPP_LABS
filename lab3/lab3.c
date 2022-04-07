#include "mpi.h"
#include "mpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>
#include <time.h>
#include <errno.h>


#define VECTORIZED_BY_HAND
//#define PRINT_RESULT
//#define ROFL


#ifdef VECTORIZED_BY_HAND


#ifdef __AVX__

inline double hsum_pd_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow  = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}


inline double scalar_prod(const double *a, const double *b, int n){

    __m256d s = _mm256_set1_pd(0);

    for(int i=0; i<n/4; ++i){
        __m256d ap = _mm256_loadu_pd(a + i*4);
        __m256d bp = _mm256_loadu_pd(b + i*4);
        s = _mm256_add_pd(s, _mm256_mul_pd(ap, bp));
    }
    double singly = 0;
    for(int i = n/4 * 4; i<n; ++i){
        singly += a[i]*b[i];
    }

    return singly + hsum_pd_avx(s);
}

#else

double hsum_pd_sse2(__m128d v) {
    double high;
    _mm_storeh_pd(&high, v);
    double low = _mm_cvtsd_f64(v);
    return low + high;
}


inline double scalar_prod(const double *a, const double *b, int n){

    __m128d s = _mm_set1_pd(0);

    for(int i=0; i<n/2; ++i){
        __m128d ap = _mm_loadu_pd(a + i*2);
        __m128d bp = _mm_loadu_pd(b + i*2);
        s = _mm_add_pd(s, _mm_mul_pd(ap, bp));
    }

    return hsum_pd_sse2(s) + (n%2==1? a[n-1]*b[n-1] : 0);
}

#endif


void gemm(const double* A, const double* B_T, double *C, int n, int k, int m){

    for(int i=0; i<n; ++i){
        for(int j=0; j<m; ++j){
            C[i*m + j] = scalar_prod(A+i*k, B_T+j*k, k);
        }
    }
}

#else

void gemm(const double* A, const double* B_T, double *C, int n, int k, int m){

    for(int i=0; i<n; ++i){
        for(int j=0; j<m; ++j){
            C[i*m + j] = 0;
            for(int k1=0; k1<k; ++k1){
                C[i*m + j] += A[i*k + k1] * B_T[j*k + k1];
            }
        }
    }
}

#endif


#define MESSAGE_TAG 666

void init(double **A, double **B, int *n, int *k, int *m){

#ifdef PRINT_RESULT
    *n = 5;
    *k = 5;
    *m = 5;
#else
    *n = 4096;
    *k = 4096;
    *m = 4096;
#endif

    *A = (double*) malloc(sizeof(double) * (*n) * (*k));
    *B = (double*) malloc(sizeof(double) * (*k) * (*m));

    if(*A == NULL || *B == NULL){
        perror("initialization fail\n");
        MPI_Abort(MPI_COMM_WORLD, errno);
    }

    for(int i=0; i<*n; ++i){
        for(int j=0; j<*k; ++j){
            (*A)[i*(*k) + j] = i*(*k) + j;
        }
    }

    for(int i=0; i<*k; ++i){
        for(int j=0; j<*m; ++j){
            (*B)[i*(*m) + j] = i*(*m) + j;
        }
    }
}


int count_of_lines(int n, int rank, int size){
    return n/size + (rank<n%size? 1:0);
}


int first_line(int n, int rank, int size){

    int res = 0;
    for(int i=0; i<rank; ++i){
        res += count_of_lines(n, i, size);
    }

    return res;
}



#ifdef ROFL

void transpose(double **matrix, int n, int k){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype column, transpose;
    MPI_Type_vector(n, 1, k, MPI_DOUBLE, &column);
    MPI_Type_hvector(k, 1, sizeof(double), column, &transpose);
    MPI_Type_commit(&transpose);

    double *tmp = (double *) malloc(sizeof(double) * n * k);
    if(tmp == NULL){
        perror("Transpose fail: \n");
        MPI_Abort(MPI_COMM_WORLD, errno);
    }

    MPI_Sendrecv(*matrix, 1, transpose, rank, MESSAGE_TAG-1, tmp, n*k, MPI_DOUBLE,
                 rank, MESSAGE_TAG-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    free(*matrix);
    *matrix = tmp;
    MPI_Type_free(&transpose);
}

#else

void transpose(double**matrix, int n, int k){

    double *tmp = (double *) malloc(sizeof(double) * n * k);
    
    for(int i=0; i<k; ++i){
        for(int j=0; j<n; ++j){
            tmp[i*n + j] = (*matrix)[j*k + i];
        }
    }

    free(*matrix);
    *matrix = tmp;
}

#endif



int main(int argc, char** argv){

    MPI_Init(&argc, &argv);
    MPE_Init_log();

    int size_comm_world, rank_comm2d, coordy, coordx;
    int lead_rank = 0, lead_coordy, lead_coordx;
    MPI_Comm_size(MPI_COMM_WORLD, &size_comm_world);

    MPI_Comm comm2d;
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
    MPI_Dims_create(size_comm_world, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm2d);

    MPI_Comm_rank(comm2d, &rank_comm2d);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);
    coordy = coords[0];
    coordx = coords[1];
    MPI_Cart_coords(comm2d, lead_rank, 2, coords);
    lead_coordy = coords[0];
    lead_coordx = coords[1];

    int n, k, m;
    double *A_in, *B_in;

    struct timespec start_time;

    int init_start = MPE_Log_get_event_number();
    int init_end = MPE_Log_get_event_number();
    MPE_Describe_state(init_start, init_end, "init", "blue");

    MPE_Log_event(init_start, 0, NULL);
    if(rank_comm2d==lead_rank){
        init(&A_in, &B_in, &n, &k, &m);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
        transpose(&B_in, k, m);
    }
    MPE_Log_event(init_end, 0, NULL);

    MPI_Bcast(&n, 1, MPI_INT, lead_rank, comm2d);
    MPI_Bcast(&k, 1, MPI_INT, lead_rank, comm2d);
    MPI_Bcast(&m, 1, MPI_INT, lead_rank, comm2d);

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(comm2d, coordy, coordx, &row_comm);
    MPI_Comm_split(comm2d, coordx, coordy, &col_comm);

    int* rows_count = (int*) malloc(sizeof(int) * dims[0]);
    int* first_row = (int*) malloc(sizeof(int) * dims[0]);
    if(rows_count == NULL || first_row == NULL){
        perror("something goes wrong\n");
        MPI_Abort(MPI_COMM_WORLD, errno);
    }
    for(int i=0; i<dims[0]; ++i){
        rows_count[i] = count_of_lines(n, i, dims[0]);
        first_row[i] = first_line(n, i, dims[0]);
    }
    int* cols_count = (int*) malloc(sizeof(int) * dims[1]);
    int* first_col = (int*) malloc(sizeof(int) * dims[1]);
    if(cols_count == NULL || first_col == NULL){
        perror("something goes wrong\n");
        MPI_Abort(MPI_COMM_WORLD, errno);
    }
    for(int i=0; i<dims[1]; ++i){
        cols_count[i] = count_of_lines(m, i, dims[1]);
        first_col[i] = first_line(m, i, dims[1]);
    }

    MPI_Datatype row_type;
    MPI_Type_contiguous(k, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    double *A = (double*) malloc(sizeof(double) * k * rows_count[coordy]);
    double *B = (double*) malloc(sizeof(double) * k * cols_count[coordx]);
    if(A == NULL || B == NULL){
        perror("something goes wrong\n");
        MPI_Abort(MPI_COMM_WORLD, errno);
    }

    if(coordx==lead_coordx) {
        MPI_Scatterv(A_in, rows_count, first_row, row_type, A, rows_count[coordy], row_type, lead_coordy, col_comm);
    }

    if(coordy==lead_coordy) {
        MPI_Scatterv(B_in, cols_count, first_col, row_type, B, cols_count[coordx], row_type, lead_coordx, row_comm);
    }

    if(rank_comm2d==lead_rank){
        free(A_in);
        free(B_in);
    }

    MPI_Bcast(A, rows_count[coordy], row_type, lead_coordx, row_comm);
    MPI_Bcast(B, cols_count[coordx], row_type, lead_coordy, col_comm);

    double *C = (double*) malloc(sizeof(double) * rows_count[coordy] * cols_count[coordx]);
    if(C == NULL){
        perror("something goes wrong\n");
        MPI_Abort(MPI_COMM_WORLD, errno);
    }

    int gemm_start = MPE_Log_get_event_number();
    int gemm_end = MPE_Log_get_event_number();
    MPE_Describe_state(gemm_start, gemm_end, "gemm", "red");
    
    MPE_Log_event(gemm_start, 0, NULL);
    gemm(A, B, C, rows_count[coordy], k, cols_count[coordx]);
    MPE_Log_event(gemm_end, 0, NULL);


    int* count_doubles = (int*) malloc(sizeof(int) * size_comm_world);
    int* displs = (int*) calloc(size_comm_world, sizeof(int));
    if(count_doubles == NULL || displs == NULL){
        perror("something goes wrong\n");
        MPI_Abort(MPI_COMM_WORLD, errno);
    }
    for(int i=0; i<size_comm_world; ++i){
        MPI_Cart_coords(comm2d, i, 2, coords);
        count_doubles[i] = rows_count[coords[0]]*cols_count[coords[1]];
        for(int j=0; j<i; ++j){
            displs[i] += count_doubles[j];
        }
    }

    double* res_buf;
    if(rank_comm2d==lead_rank){
        res_buf = (double*) malloc(sizeof(double) * n * m);
        if(res_buf == NULL){
            perror("something goes wrong\n");
            MPI_Abort(MPI_COMM_WORLD, errno);
        }
    }

    MPI_Gatherv(C, count_doubles[rank_comm2d], MPI_DOUBLE,
    res_buf, count_doubles, displs, MPI_DOUBLE,
    lead_rank, comm2d);

    if(rank_comm2d==lead_rank){
        double* res = (double*) malloc(sizeof(double) * n * m);
        if(res == NULL){
            perror("something goes wrong\n");
            MPI_Abort(MPI_COMM_WORLD, errno);
        }

        for(int i=0; i<dims[0]; ++i){
            for(int j=0; j<dims[1]; ++j){

                int node_rank;
                coords[0] = i;
                coords[1] = j;
                MPI_Cart_rank(comm2d, coords, &node_rank);

                for(int i1=0; i1<rows_count[i]; ++i1){
                    for(int j1=0; j1<cols_count[j]; ++j1){
                        res[(i1+first_row[i])*m + j1 + first_col[j]] =
                                (res_buf + displs[node_rank])[i1*cols_count[j] + j1];
                    }
                }
            }
        }

        struct timespec end_time;
        clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);

#ifdef PRINT_RESULT
        for(int i=0; i<n; ++i){
            for(int j=0; j<m; ++j){
                printf("%.0F   ", res[i*m + j]);
            }
            printf("\n");
        }
#endif

        fprintf(stderr, "====================================\n");

#ifdef VECTORIZED_BY_HAND
        fprintf(stderr, "\nHand vectorized: ");
#ifdef __AVX__
        fprintf(stderr, "AVX");
#else
        fprintf(stderr, "SSE");
#endif

#endif

//        fprintf(stderr, "Time taken: %lf sec.\nCount of MPI process: %d\n\n",
//               end_time.tv_sec-start_time.tv_sec + 10e-9*(end_time.tv_nsec-start_time.tv_nsec),
//               size_comm_world);
        fprintf(stderr,"\nCount of MPI process: %d\n", size_comm_world);

        free(res_buf);
        free(res);
    }

    free(A);
    free(B);
    free(C);
    free(first_col);
    free(first_row);
    free(rows_count);
    free(cols_count);
    free(displs);
    free(count_doubles);

    MPI_Finalize();
    return 0;
}
