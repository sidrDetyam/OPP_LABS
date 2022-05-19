
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


//#define PROFILE

#ifdef PROFILE
#include "mpe.h"
#endif


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


double x0, y0_, z0, Dx, Dy, Dz, eps, cst;
int Nx, Ny, Nz;
double hx, hy, hz;


double g(double x, double y, double z){
    return x*x + y*y + z*z;
}


double f(double x, double y, double z){
    return 6 - cst * g(x, y, z);
}


int index_(int x, int y, int z){
    return z*Nx*Ny + y*Nx + x;
}


int in_bounds(int x, int y, int z, int sx, int sy, int sz){
    return 0<=x && x<sx && 0<=y && y<sy && 0<=z && z<sz;
}


double get_value(const double* p, int x, int y, int z, int fz){
    return in_bounds(x, y, z, Nx, Ny, Nz+2)?
        p[index_(x, y, z)] : g(x0+x*hx, y0_+y*hy, z0+(z-1+fz)*hz);
}


void calculate(const double* p, double* p_tmp, int a, int b, int fz){

    for(int z=a; z<b; ++z){
        for(int y=0; y<Ny; ++y){
            for(int x=0; x<Nx; ++x){

                p_tmp[index_(x, y, z)] = (1/(2/hx/hx + 2/hx/hx + 2/hx/hx + cst)) *
                        ((get_value(p, x+1, y, z, fz)+get_value(p, x-1, y, z, fz))/hx/hx +
                        (get_value(p, x, y+1, z, fz)+get_value(p, x, y-1, z, fz))/hy/hy +
                        (get_value(p, x, y, z+1, fz)+get_value(p, x, y, z-1, fz))/hz/hz -
                        f(x0+x*hx, y0_+y*hy, z0+(z-1+fz)*hz));
            }
        }
    }
}


#define MAX(x, y) (((x) > (y)) ? (x) : (y))


double calculate_eps(double* p, double* p_tmp, int cnt_z) {

    double res = 0;
    for(int i=index_(0, 0, 1); i< index_(0, 0, cnt_z+1); ++i){
        res = MAX(res, fabs(p[i]-p_tmp[i]));
    }
    return res;
}


void init_edge(double* p, double* p_tmp, int z, int fz){

    for(int y=0; y<Ny; ++y){
        for(int x=0; x<Nx; ++x){
            p[index_(x, y, z)] = p_tmp[index_(x, y, z)] = g(x0+x*hx, y0_+y*hy, z0 + (fz+z-1)*hz);
        }
    }
}


#define MESSAGE_TAG 42


int main(int argc, char** argv){

    MPI_Init(&argc, &argv);
#ifdef PROFILE
    MPE_Init_log();
#endif
    int size_comm_world;
    int rank_comm_world;
    MPI_Comm_size(MPI_COMM_WORLD, &size_comm_world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_comm_world);

    x0 = y0_ = z0 = -1;
    Dx = Dy = Dz = 2;
    Nx = Ny = Nz = 400;
    hx = Dx / (Nx - 1);
    hy = Dy / (Ny - 1);
    hz = Dz / (Nz - 1);
    eps = 1e-4;
    cst = 1e5;

    int cnt_z = count_of_lines(Nz, rank_comm_world, size_comm_world);
    int cnt_el = Nx * Ny * (cnt_z+2);
    int first_z = first_line(Nz, rank_comm_world, size_comm_world);
    double* p = (double*) malloc(sizeof(double) * cnt_el);
    double* p_tmp = (double*) malloc(sizeof(double) * cnt_el);
    memset(p, 0, cnt_el);

    if(rank_comm_world==0){
        init_edge(p, p_tmp, 0, first_z);
    }

    if(rank_comm_world==size_comm_world-1){
        init_edge(p, p_tmp, cnt_z+1, first_z);
    }

#ifdef PROFILE
    int calc_start = MPE_Log_get_event_number();
    int calc_end = MPE_Log_get_event_number();
    MPE_Describe_state(calc_start, calc_end, "calculate", "blue");
#endif

    while(1){

        MPI_Request req[4];

#ifdef PROFILE
        MPE_Log_event(calc_start, 0, NULL);
#endif
        calculate(p, p_tmp, 1, 2, first_z);
#ifdef PROFILE
        MPE_Log_event(calc_end, 0, NULL);
#endif

        if(rank_comm_world != 0){

            MPI_Isend(p_tmp + index_(0, 0, 1), Nx*Ny, MPI_DOUBLE, rank_comm_world-1,
                      MESSAGE_TAG+1, MPI_COMM_WORLD, req);

            MPI_Irecv(p_tmp, Nx*Ny, MPI_DOUBLE, rank_comm_world-1,
                      MESSAGE_TAG+2, MPI_COMM_WORLD, req+1);
        }

#ifdef PROFILE
        MPE_Log_event(calc_start, 0, NULL);
#endif
        calculate(p, p_tmp, cnt_z, cnt_z+1, first_z);
#ifdef PROFILE
        MPE_Log_event(calc_end, 0, NULL);
#endif

        if(rank_comm_world != size_comm_world-1){

            MPI_Isend(p_tmp + index_(0, 0, cnt_z), Nx*Ny, MPI_DOUBLE, rank_comm_world+1,
                      MESSAGE_TAG+2, MPI_COMM_WORLD, req+2);

            MPI_Irecv(p_tmp + index_(0, 0, cnt_z+1), Nx*Ny, MPI_DOUBLE, rank_comm_world+1,
                      MESSAGE_TAG+1, MPI_COMM_WORLD, req+3);
        }

#ifdef PROFILE
        MPE_Log_event(calc_start, 0, NULL);
#endif
        calculate(p, p_tmp, 2, cnt_z, first_z);
#ifdef PROFILE
        MPE_Log_event(calc_end, 0, NULL);
#endif

        if(rank_comm_world != 0){
            MPI_Wait(req, MPI_STATUS_IGNORE);
            MPI_Wait(req+1, MPI_STATUS_IGNORE);
        }

        if(rank_comm_world != size_comm_world-1){
            MPI_Wait(req+2, MPI_STATUS_IGNORE);
            MPI_Wait(req+3, MPI_STATUS_IGNORE);
        }

        double* tmp = p;
        p = p_tmp;
        p_tmp = tmp;

        double mx_eps, local_eps = calculate_eps(p, p_tmp, cnt_z);
        MPI_Reduce(&local_eps, &mx_eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mx_eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#ifndef PROFILER
        if(rank_comm_world==0){
            printf(" --- %f\n", mx_eps);
        }
#endif

        if(mx_eps<eps){
            break;
        }
    }


    int* counts = (int*) malloc(sizeof(int) * size_comm_world);
    int* displs = (int*) malloc(sizeof(int) * size_comm_world);

    for(int i=0; i<size_comm_world; ++i){
        counts[i] = count_of_lines(Nz, i, size_comm_world) * Nx * Ny;
        displs[i] = first_line(Nz, i, size_comm_world) * Nx * Ny;
    }

    double* res;
    if(rank_comm_world==0){
        res = (double*) malloc(sizeof(double)*Nx*Ny*Nz);
    }

    MPI_Gatherv(p + index_(0, 0, 1), counts[rank_comm_world], MPI_DOUBLE,
                res, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank_comm_world==0) {
        free(res);
    }
    free(p);
    free(p_tmp);
    free(counts);
    free(displs);
    MPI_Finalize();

    return 0;
}
