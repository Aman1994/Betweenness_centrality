#ifdef __CUDACC__
#include "sm_11_atomic_functions.h"

#define ISBITSET(x,i) ((x[i>>3] & (1<<(i&7)))!=0)
#define SETBIT(x,i) x[i>>3]|=(1<<(i&7))
#define CLEARBIT(x,i) x[i>>3]&=(1<<(i&7))^0xFF

extern "C"
{

void __global__ bfs_kernel_gpu( int* graph_node_start , int* graph_edges, int* queue1, int* queue2, int* queue_pointer1,char* visited, int* frontier_vector, int current_frontier, int offset, int ntasks, int chunksize, int current_task,int* queue_in,int q_in_values, int first_it)
{
    volatile __shared__ unsigned int comm [16][3];
    int r, rend;
    int tid,node,edge_displacement;
    int warp_id=threadIdx.x/32;
    int lane_id=threadIdx.x%32;
    int rgather,rgatherend;
    int *q_in;
    int numelems;
    int stride;
    
    volatile int neigh;
    
    
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (first_it==1){
        q_in=queue_in;
        numelems=q_in_values;
        stride=0;
    }
    else {
        q_in=queue1;
        numelems=queue_pointer1[current_task];
        stride=0;//current_task*chunksize;
    }

    edge_displacement=graph_node_start[0];
    if (tid<numelems){
        node = q_in[stride+tid]-offset;
        if(frontier_vector[node]==-1) frontier_vector[node]=current_frontier;
        r=graph_node_start[node];
        rend=graph_node_start[node+1];
        
    }
    else {
        r=0;
        rend=0;
    }
    
    while (__any(rend-r>0)){
        
        if(rend-r>0)comm[warp_id][0]=lane_id;
        
        if(comm[warp_id][0]==lane_id){
            comm[warp_id][1]=r;
            comm[warp_id][2]=rend;
            
            
            r=rend;
            
        }
        
        rgather=comm[warp_id][1]+lane_id;
        rgatherend=comm[warp_id][2];
        while(rgather<rgatherend){
            neigh=graph_edges[rgather-edge_displacement];
            rgather+=warpSize;
            
            if(!ISBITSET(visited,neigh))
            {
           
                SETBIT(visited,neigh);
                queue2[neigh]=1;
                
                
            }
        }
        
        
    }

}

void __global__ bfs_kernel_copy( int* graph_node_start , int* graph_edges, int* queue1, int* queue2, int* queue_pointer1,char* visited, int current_frontier, int offset, int ntasks, int chunksize)  
{
    
   
}
}

#endif

