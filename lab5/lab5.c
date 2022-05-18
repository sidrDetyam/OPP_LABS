
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include "Task.h"


#define BALANCE


struct ExecutionContext{
    pthread_mutex_t mutex;
    size_t current_task;
    int is_own_task_done;
    vTask tasks;
    int rank_comm_world;
    int comm_world_size;
};


void context_init(struct ExecutionContext* context){
    if(pthread_mutex_init(&context->mutex, NULL)!=0){
        perror("mutex init fail: ");
        MPI_Finalize();
        exit(1);
    }
    vTask_init(&context->tasks);
    context->current_task = 0;
    context->is_own_task_done = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &context->comm_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &context->rank_comm_world);
}


#define MESSAGE_TAG 42


void* main_thread(void* context_){

    struct ExecutionContext* context = (struct ExecutionContext*) context_;

    while(1){
        pthread_mutex_lock(&context->mutex);

        size_t current_task = context->current_task;
        if(current_task == context->tasks.cnt){
            context->is_own_task_done = 1;
            pthread_mutex_unlock(&context->mutex);
            break;
        }

        ++context->current_task;
        pthread_mutex_unlock(&context->mutex);

        task_doer(&context->tasks.ptr[current_task]);
    }


#ifdef BALANCE

    for(int i=0; i<context->comm_world_size; ){

        int want_task = 1;
        MPI_Send(&want_task, 1, MPI_INT, i, MESSAGE_TAG, MPI_COMM_WORLD);
        MPI_Recv(&want_task, 1, MPI_INT, i, MESSAGE_TAG+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(want_task==0){
            ++i;
            continue;
        }

        printf("%d %d\n", context->rank_comm_world, i);

        Task task;
        recv_task(&task, i);
        task_doer(&task);

        //not necessary
        pthread_mutex_lock(&context->mutex);
        vTask_push_back(&context->tasks, &task);
        pthread_mutex_unlock(&context->mutex);
    }

    printf("%d\n", context->rank_comm_world);

    MPI_Barrier(MPI_COMM_WORLD);
    int want_task=0;
    MPI_Send(&want_task, 1, MPI_INT, context->rank_comm_world, MESSAGE_TAG, MPI_COMM_WORLD);

#endif
}


void* support_thread(void* context_) {

    struct ExecutionContext *context = (struct ExecutionContext *) context_;

    while(1){
        int want_task = 1;
        MPI_Status status;
        MPI_Recv(&want_task, 1, MPI_INT, MPI_ANY_SOURCE, MESSAGE_TAG, MPI_COMM_WORLD, &status);

        if(want_task==0){
            break;
        }

        Task task;
        pthread_mutex_lock(&context->mutex);
        if(context->current_task == context->tasks.cnt || context->is_own_task_done){
            want_task = 0;
        }
        else{
            want_task = 1;
            memcpy(&task, vTask_back(&context->tasks), sizeof(Task));
            vTask_pop_back(&context->tasks);
        }
        pthread_mutex_unlock(&context->mutex);

        MPI_Send(&want_task, 1, MPI_INT, status.MPI_SOURCE, MESSAGE_TAG+1, MPI_COMM_WORLD);
        if(want_task){
            send_task(&task, status.MPI_SOURCE);
        }
    }
}



int main(int argc, char** argv){

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if(provided != MPI_THREAD_MULTIPLE){
        fprintf(stderr, "MPI doesn't support required level");
        MPI_Finalize();
        return 1;
    }

    struct ExecutionContext context;
    context_init(&context);
    for(int i=0; i<3; ++i){
        Task task;
        task.res = 0;
        task.job = (context.rank_comm_world+1)*3;
        task.id = context.rank_comm_world;
        vTask_push_back(&context.tasks, &task);
    }

    pthread_attr_t attr;
    if(pthread_attr_init(&attr) != 0){
        perror("pthread_attr_init fail");
        return 1;
    };

    if(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE) != 0){
        perror("pthread_attr_setdetachstate fail");
        return 1;
    }

    pthread_t threads[2];

    if(pthread_create(threads, &attr, main_thread, &context) != 0){
        perror("pthread_create fail");
        return 1;
    }

#ifdef BALANCE
    if(pthread_create(threads+1, &attr, support_thread, &context) != 0){
        perror("pthread_create fail");
        return 1;
    }
#endif

    if(pthread_attr_destroy(&attr) !=0 ){
        perror("pthread_attr_destroy fail");
        return 1;
    }


    if (pthread_join(threads[0], NULL) != 0) {
        perror("pthread_attr_destroy fail");
        return 1;
    }


#ifdef BALANCE
    if (pthread_join(threads[1], NULL) != 0) {
        perror("pthread_attr_destroy fail");
        return 1;
    }
#endif

    for(int i=0; i<context.comm_world_size; ++i){

        if(i==context.rank_comm_world){
            printf("\nProcess %d:\n", i);
            for(int j=0; j<context.tasks.cnt; ++j){
                Task* task = context.tasks.ptr+j;
                printf("%d, %d, %d\n", task->job, task->res, task->id);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }


    vTask_free(&context.tasks);

    MPI_Finalize();
    return 0;
}
