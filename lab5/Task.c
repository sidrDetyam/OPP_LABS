
#include "Task.h"

#define ELEMENT_TYPE Task
#include "CVector_impl.h"

void task_doer(Task* task){
    sleep(task->job);
    task->res = 1;
}


void send_task(Task* task, int rank){
    //packing is better!
    MPI_Send(&task->job, 1, MPI_INT, rank, MESSAGE_TAG_TASK+1, MPI_COMM_WORLD);
    MPI_Send(&task->id, 1, MPI_INT, rank, MESSAGE_TAG_TASK+1, MPI_COMM_WORLD);
}


void recv_task(Task* task, int rank){
    //packing is better!
    MPI_Recv(&task->job, 1, MPI_INT, rank, MESSAGE_TAG_TASK+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&task->id, 1, MPI_INT, rank, MESSAGE_TAG_TASK+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
