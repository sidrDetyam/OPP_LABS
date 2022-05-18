
#ifndef OPP_LAB5
#define OPP_LAB5

#include <unistd.h>
#include <mpi.h>

struct Task{
    int job;
    int res;
    int id;
};
typedef struct Task Task;
#define ELEMENT_TYPE Task
#include "CVector_def.h"


void task_doer(Task* task);

#define MESSAGE_TAG_TASK 666

void send_task(Task* task, int rank);

void recv_task(Task* task, int rank);

#endif
