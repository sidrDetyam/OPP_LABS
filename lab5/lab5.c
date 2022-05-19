
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include "Task.h"


#define BALANCE
#define DEBUG


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

#ifdef DEBUG
        fprintf(stderr, "%d %d\n", context->rank_comm_world, i);
#endif

        Task task;
        recv_task(&task, i);
        task_doer(&task);

        //not concurrent
        vTask_push_back(&context->tasks, &task);
    }

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


Task* gather_results(struct ExecutionContext* context, size_t* task_cnt){

    int* counts = (int*) malloc(sizeof(int) * context->comm_world_size);
    int* displs = (int*) malloc(sizeof(int) * context->comm_world_size);
    if(counts == NULL || displs == NULL){
        fprintf(stderr, "alloc fail\n");
        exit(1);
    }

    int count_byte = context->tasks.cnt * sizeof(Task);
    MPI_Allgather(&count_byte, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
    int displ = 0;
    int total_bytes = 0;
    for(int i=0; i<context->comm_world_size; ++i){
        displs[i] = displ;
        displ += counts[i];
        total_bytes += counts[i];
    }

    Task *tasks = NULL;
    if(context->rank_comm_world == 0){
        if ((tasks = (Task *) malloc(total_bytes)) == NULL) {
            perror("alloc fail");
            exit(1);
        }
    }
    //warning - does not work correctly on different architectures
    MPI_Gatherv(context->tasks.ptr, count_byte, MPI_BYTE, tasks,
                counts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);

    free(displs);
    free(counts);

    *task_cnt = total_bytes / sizeof(Task);
    if(context->rank_comm_world == 0){
        sort_tasks_id(tasks, *task_cnt);
    }

    return tasks;
}


void complete_tasks(struct ExecutionContext* context){

    pthread_attr_t attr;
    if(pthread_attr_init(&attr) != 0){
        perror("pthread_attr_init fail");
        exit(1);
    };

    if(pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE) != 0){
        perror("pthread_attr_setdetachstate fail");
        exit(1);
    }

    pthread_t threads[2];

    if(pthread_create(threads, &attr, main_thread, context) != 0){
        perror("pthread_create fail");
        exit(1);
    }

#ifdef BALANCE
    if(pthread_create(threads+1, &attr, support_thread, context) != 0){
        perror("pthread_create fail");
        exit(1);
    }
#endif

    if(pthread_attr_destroy(&attr) !=0 ){
        perror("pthread_attr_destroy fail");
        exit(1);
    }


    if (pthread_join(threads[0], NULL) != 0) {
        perror("pthread_attr_destroy fail");
        exit(1);
    }


#ifdef BALANCE
    if (pthread_join(threads[1], NULL) != 0) {
        perror("pthread_attr_destroy fail");
        exit(1);
    }
#endif


#ifdef DEBUG
    for(int i=0; i<context->comm_world_size; ++i){

        if(i==context->rank_comm_world){
            printf("\nProcess %d:\n", i);
            for(int j=0; j<context->tasks.cnt; ++j){
                print_task(context->tasks.ptr+j);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

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
        create_task(&task, (context.rank_comm_world+1)*3, context.rank_comm_world*3+i);
        vTask_push_back(&context.tasks, &task);
    }

    complete_tasks(&context);
    size_t cnt;
    Task* tasks = gather_results(&context, &cnt);
    vTask_free(&context.tasks);

    MPI_Barrier(MPI_COMM_WORLD);
    if(context.rank_comm_world == 0) {
        printf("\nResult: \n");
        for (int i = 0; i < cnt; ++i) {
            print_task(tasks + i);
        }

        free(tasks);
    }

    MPI_Finalize();
    return 0;
}
