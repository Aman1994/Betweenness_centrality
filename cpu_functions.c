#define ISBITSET(x,i) ((x[i>>3] & (1<<(i&7)))!=0)
#define SETBIT2(x,i) (__sync_fetch_and_or(&x[i>>3],(1<<(i&7))))
#define CLEARBIT(x,i) x[i>>3]&=(1<<(i&7))^0xFF

#pragma omp target device(smp) copy_inout([chunksize]graph_frontier,[queue_elems]q_in)
#pragma omp task input([chunksize+1]graph_node_start,[edgesize]graph_edges) concurrent( [visited_size]visited,[no_of_nodes]q_out,[ntasks]q_in_ptr,[ntasks]q_out_ptr, [1]elems_processed) 
void bfs_cpu( int* graph_node_start, int* graph_edges, int* graph_frontier,char* visited, int chunksize, int current_frontier, int total_nodes, int no_of_edges, int* q_in, int* q_out, int* q_in_ptr, int* q_out_ptr,int task,int ntasks,int edgesize, int** cache_ptr, int** cache, int visited_size, int queue_elems, int* elems_processed) 
{
   // printf("called with elems %d and\n",task);
      
    int mask,id,i,j,k,out_q, threadid,offset;
 	int local_elems=0;
    threadid=omp_get_thread_num();
    
    mask=0;
    offset=graph_node_start[0];
	
    for (i=0; i<q_in_ptr[task]; i++){

        mask = q_in[i]-chunksize*task;
//printf("now dealing with %d and offset %d. Num elements to see: %d\n",q_in_ptr[task],offset,graph_node_start[mask+1]-graph_node_start[mask]);
        if (graph_frontier[mask]==-1)graph_frontier[mask]=current_frontier;
        for( j=graph_node_start[mask]; j<(graph_node_start[mask+1]); j++)
        {
            id = graph_edges[j-offset];
            if(!ISBITSET(visited,id))
            {
                       local_elems++;
                SETBIT2(visited,id);
                out_q=id/chunksize;
                cache[threadid][cache_ptr[threadid][out_q]+out_q*chunksize]=id;
                cache_ptr[threadid][out_q]++;
             
            }
    
        }
    }

    q_in_ptr[task]=0;
    for(i=0; i<ntasks;i++){
        if (cache_ptr[threadid][i]>0){
            k=__sync_fetch_and_add(&q_out_ptr[i],cache_ptr[threadid][i]);
       
            memcpy(&q_out[k+i*chunksize],&cache[threadid][chunksize*i],sizeof(int)*cache_ptr[threadid][i]);
            cache_ptr[threadid][i]=0;
        }
    }
        __sync_fetch_and_add(elems_processed,local_elems);
}
