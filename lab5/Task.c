
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

void print_task(Task* task){
    printf("%d %d %d\n", task->job, task->res, task->id);
}


void create_task(Task* task, int job, int id){
    task->id = id;
    task->job = job;
    task->res = 0;
}


static int comp(const void* a, const void* b){
    return ((Task*) a)->id > ((Task*) b)->id;
}

void sort_tasks_id(Task* tasks, size_t counts){
    qsort(tasks, counts, sizeof(Task), comp);
}
