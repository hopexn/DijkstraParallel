#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include <unistd.h>
#include <sys/time.h>

using namespace std;

#define INF 1000000

int Find_Min_Dist(int *local_dist, int local_known[], int local_n);

void Dijkstra(int *local_mat, int *local_dist, int *local_pred, int n, int local_n, int my_rank, MPI_Comm comm);


/**
 *
 * @param n : Global size of matrix
 * @param local_n : Local size of matrix
 * @return Formated MPI data type
 */
MPI_Datatype Build_Column_Type(int n, int local_n) {
    MPI_Aint lb, extent;
    MPI_Datatype type;
    MPI_Datatype block_type;
    MPI_Datatype vector_type;

    MPI_Type_contiguous(local_n, MPI_INT, &block_type);
    MPI_Type_get_extent(block_type, &lb, &extent);
    MPI_Type_vector(n, local_n, n, MPI_INT, &vector_type);
    MPI_Type_create_resized(vector_type, lb, extent, &type);

    MPI_Type_commit(&type);

    MPI_Type_free(&block_type);
    MPI_Type_free(&vector_type);

    return type;
}

int main(int argc, char *argv[]) {
    int *mat = NULL, *dist = NULL, *local_mat = NULL;
    int n, local_n, size, my_rank;
    int *local_dist = NULL, *local_pred = NULL;
    MPI_Comm comm;
    MPI_Datatype datatype;

    struct timeval tv1, tv2;
    struct timezone tz;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &my_rank);

    if (my_rank == 0) {
        dist = (int *) malloc(sizeof(int) * n);
        mat = (int *) malloc(sizeof(int) * n * n);
        scanf("%d", &n);  //Get the rows of matrix
    }

    /*
     * Broadcast the size of matrix
     */
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    if(n % size == 0)
        local_n = n / size;
    else
        local_n = n / size + 1;
    /*
     * Get MPI data type
     * row = n , column = local_n
     */
    datatype = Build_Column_Type(n, local_n);

    /*
     * Read adjacency matrix from input
     */
    //srand(time(NULL));
    
    if (my_rank == 0) {
        srand(234);
        cout << "Matrix:"<<endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mat[i*n+j] = rand() % 20;
                //scanf("%d", &mat[i * n + j]);
                //cout<<mat[i*n+j] << " ";
            }
            //cout<<endl;
        }
        gettimeofday(&tv1, &tz);
    }

    /*
     * Allocate space for matrices and initialize them
     */
    local_mat = (int *) malloc(sizeof(int) * local_n * n);
    local_dist = (int *) malloc(sizeof(int) * local_n);
    local_pred = (int *) malloc(sizeof(int) * local_n);
    memset(local_mat, INF, sizeof(int) * local_n * n);
    memset(local_dist, INF, sizeof(int) * local_n);
    memset(local_pred, -1, sizeof(int) * local_n);

    /*
     * Scatter data to other processes
     */
    MPI_Scatter(mat, 1, datatype, local_mat, n * local_n, MPI_INT, 0, comm);

    /*
     * Find shortest path by Dijkstra algorithm
     */
    Dijkstra(local_mat, local_dist, local_pred, n, local_n, my_rank, comm);

    /*
     * Gather partial result and store them in array 'dist'
     */
    MPI_Gather(local_dist, local_n, MPI_INT, dist, local_n, MPI_INT, 0, comm);

    /*
     * Print MinDist in main process
     */
    if (my_rank == 0) {
        gettimeofday(&tv2, &tz);
        cout << "MinDist:" << endl;
        for (int i = 0; i < n; i++) {
            cout << i << ": " << dist[i] << endl;
        }
        cout<<"Time cost: "<<tv2.tv_sec*1000+tv2.tv_usec - tv1.tv_sec*1000+tv1.tv_usec<<" usec"<<endl;
        free(dist);
        free(mat);
    }

    /*
     * Free the space
     */
    free(local_mat);
    free(local_dist);
    free(local_pred);
    MPI_Type_free(&datatype);
    MPI_Finalize();

    return 0;
}

/**
 * Find first local node with minimum distance
 * @param local_dist
 * @param local_known
 * @param local_n
 * @return
 */
int Find_Min_Dist(int *local_dist, int local_known[], int local_n) {
    int current_min_dist = INF;
    int local_u = -1;
    for (int i = 0; i < local_n; i++) {
        if (local_known[i] == 0 && local_dist[i] < current_min_dist) {
            local_u = i;
            current_min_dist = local_dist[i];
        }
    }
    return local_u;
};

/**
 * Parallel Dijkstra Algorithm
 *
 * @param local_mat
 * @param local_dist
 * @param local_pred
 * @param n
 * @param local_n
 * @param my_rank
 * @param comm
 */
void Dijkstra(int *local_mat, int *local_dist, int *local_pred,
              int n, int local_n, int my_rank, MPI_Comm comm) {
    int local_known[local_n];
    int local_u, local_v, u, dist_u, new_dist;
    int min_dist_loc[2];
    int min_dist[2];

    /*
     * Initial local minimum distance
     */
    for (local_v = 0; local_v < local_n; local_v++) {
        local_dist[local_v] = local_mat[local_v];
        local_pred[local_v] = local_known[local_v] = 0;
    }
    /*
     * Set Node 0 with visited mark
     */
    if (my_rank == 0) {
        local_known[0] = 1;
    }

    /*
     * find a local node which is not visited with minimum distance
     */
    for (int i = 1; i < n; i++) {
        local_u = Find_Min_Dist(local_dist, local_known, local_n);
        if (local_u >= 0) {
            min_dist_loc[0] = local_dist[local_u];
            min_dist_loc[1] = my_rank * local_n + local_u;
        } else {
            min_dist_loc[0] = INF;
            min_dist_loc[1] = -1;
        }
    }

    /*
     * Send:
     * min_dist_loc[0]: the minimum distance to a local node
     * min_dist_loc[1]: the global position of the local node
     *
     * Receive:
     * min_dist[0]: the minimum distance to a node
     * min_dist[1]: the global position of the node
     */
    MPI_Allreduce(min_dist_loc, min_dist, 1, MPI_2INT, MPI_MINLOC, comm);


    dist_u = min_dist[0];
    u = min_dist[1];
    if (u / local_n == my_rank) {
        local_known[local_u] = 1;
    }

    /*
     * calculate new minimum distance
     */
    for (local_v = 0; local_v < local_n; local_v++) {
        if (local_known[local_v] == 0) {
            new_dist = dist_u + local_mat[local_n * u + local_v];
            if (new_dist < local_dist[local_v]) {
                local_dist[local_v] = new_dist;
                local_pred[local_v] = u;
            }
        }
    }
}

