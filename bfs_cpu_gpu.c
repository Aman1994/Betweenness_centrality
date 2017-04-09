#include <malloc.h>
#include <string.h>
#include <cuda.h>
#include <omp.h>
#include "cpu_functions.c"
//#include "kernel.cu"
//#include "kernel_scan.cu"
#include <unistd.h>
#include <stdio.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))	//macro returning minimum of two numbers
#define ISBITSET(x,i) ((x[i>>3] & (1<<(i&7)))!=0)	//macro checking whether the i-th position of the bitmap is a one
#define SETBIT(x,i) x[i>>3]|=(1<<(i&7))	//set the i-th position of the bitmap to one
#define CLEARBIT(x,i) x[i>>3]&=(1<<(i&7))^0xFF	//set the i-th position of the bitmap to 0
#define THREADBLOCK_SIZE 512	//Size of the GPU thread-block
void __global__ bfs_kernel_gpu( int* graph_node_start , int* graph_edges, int* queue1, int* queue2, int* queue_pointer1,char* visited, int* frontier_vector, int current_frontier, int offset, int ntasks, int chunksize, int current_task,int* queue_in,int q_in_values, int first_it);

//Graph information
int no_of_nodes;		//Number of nodes
int no_of_edges;		//Number of edges
int orig_node;			//Source node
int visited_size;		//Number of already visited vertices
int *max_edge_chunk_size;	//Edge size for each of the tasks
int *graph_node_start;		//Array storing the pointers to the edge array
int *graph_node_num;		//Array storing the number of neighbours for each vertex
int *graph_frontier;		//Array storing the frontier of each vertex
int *graph_edges;		//Array storing the edges
char *visited;			//Visited Vertex bitmap array

//GPU data structures
int *queue1;			//GPU Vertex origin array
int *queue2;			//GPU Vertex destination array
int *queue_pointer1;		//GPU Vertex queue pointer (WE ONLY NEED ONE)
int *aux_queue_pointer;		//Auxiliar data structure used for vertex compaction
int *aux;			//Auxiliar data structure used for vertex compaction
int *aux_queue;			//Auxiliar data structure used for vertex compaction
int *buf;			//Auxiliar data structure used for vertex compaction

//CPU data structures
int *queue1_cpu;		//CPU Vertex origin array
int *queue2_cpu;		//CPU Vertex destination array
int *queue_pointer1_cpu;	//queue1_cpu pointer 
int *queue_pointer2_cpu;	//queue2_cpu pointer
int **cache;			//Auxiliary thread-safe queue for vertex store
int **cache_ptr;		//Cache queue pointer

//other structures
FILE *fp;
uint N = 256 * 256 * 16;
uint MAX_BATCH_ELEMENTS = 64 * 1048576;
uint BUFF_SIZE = (4 * THREADBLOCK_SIZE);
uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;	//4 * THREADBLOCK_SIZE*THREADBLOCK_SIZE;
int cpu_only = 0;
char *source = NULL;
char *dest = NULL;
int limit;
int ntasks;

//Load graph function
void load_graph (int argc, char **argv);

//Auxiliar sorting function
int
sort (const void *x, const void *y)
{
  return (*(int *) x - *(int *) y);
}


#pragma omp target device(cuda) copy_deps
#pragma omp task input([chunksize+1]graph_node_start,[minedgesize]graph_edges,[q_in_values] queue_in)\
inout([no_of_nodes]queue2,[chunksize]queue1,[chunksize]graph_frontier,[ntasks]queue_pointer1,[visited_size]visited)
void
bfs_gpu (int *graph_node_start, int *graph_edges, int *queue1, int *queue2,
	 int *queue_pointer1, int *graph_frontier, char *visited,
	 int chunksize, int no_of_nodes, int minedgesize, int offset,
	 int ntasks, int frontier, int current_task,
	 int blocksize, int visited_size, int *queue_in, int q_in_values,
	 int first_it_in_gpu);

//Main function dedicated to loop through the iterations of the BFS
void
launch_bfs_gpu (int ntasks, int limit)
{

  int frontier = 0;		//BFS iteration value
  int i;
  int *swap_queue;		//Swap auxiliar pointer
  int *swap_ptr;

  int first_it_in_gpu = 1;	//First_it_in_gpu=1 iff execution not moved to GPU yet
  int elems_processed = 0;	//Number of processed nodes at a given iteration
  int chunksize = no_of_nodes / ntasks;	//Number of nodes per task

  do
    {
     
	  printf ("Iteration %d done in GPU \n", frontier);

	  //If it's the first time accessing the GPU we need to copy the queue of unexplored vertices
	  if (first_it_in_gpu == 1)
	    {
	      for (i = 0; i < no_of_nodes; i += chunksize)
		{

		  bfs_gpu (&graph_node_start[i],	//Start of the vertex array for task i/chunksize
			   &graph_edges[graph_node_start[i]],	//Start of the edge array for task i/chunksize
			   &queue1[i],	//Address to the origin vertex queue for task i/chunksize
			   queue2,	//Address to the destination vertex queue
			   queue_pointer1,	//Origin vertex queue pointer array
			   &graph_frontier[i],	//Start of the frontier array for task i/chunksize
			   visited,	//Visited bitmap array
			   chunksize,	//Number of nodes per task
			   no_of_nodes,	//Number of total nodes
			   max_edge_chunk_size[i / chunksize],	//Total number of edges for task i/chunksize
			   i,	//Starting node number for task i/chunksize
			   ntasks,	//Total number of tasks
			   frontier,	//BFS iteration value
			   i / chunksize,	//Task 
			   (queue_pointer1_cpu[i / chunksize] + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE,	//number of CUDA blocks needed
			   visited_size,	//Visited vertex bitmap size
			   &queue1_cpu[i],	//Origin vertex queue pointer array from CPU
			   queue_pointer1_cpu[i / chunksize],	//Number of elements in the Origin vertex queue
			   first_it_in_gpu);	//First_it_in_gpu=1 iff execution not moved to GPU yet

		}
	    }
	  //Else all the relevant information is already resident on GPU
	  else
	    {

	      for (i = 0; i < no_of_nodes; i += chunksize)
		{
		  bfs_gpu (&graph_node_start[i],	//Start of the vertex array for task i/chunksize
			   &graph_edges[graph_node_start[i]],	//Start of the edge array for task i/chunksize
			   &queue1[i],	//Address to the origin vertex queue for task i/chunksize
			   queue2,	//Address to the destination vertex queue
			   queue_pointer1,	//Origin vertex queue pointer array
			   &graph_frontier[i],	//Start of the frontier array for task i/chunksize
			   visited,	//Visited bitmap array
			   chunksize,	//Number of nodes per task
			   no_of_nodes,	//Number of total nodes
			   max_edge_chunk_size[i / chunksize],	//Total number of edges for task i/chunksize
			   i,	//Starting node number for task i/chunksize
			   ntasks,	//Total number of tasks
			   frontier,	//BFS iteration value
			   i / chunksize,	//Task
			   (queue_pointer1[i / chunksize] + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE,	//number of CUDA blocks needed
			   visited_size,	//Visited vertex bitmap size
			   &queue1_cpu[i],	//Origin vertex queue pointer array from CPU
			   queue_pointer1_cpu[i / chunksize],	//Number of elements in the Origin vertex queue
			   first_it_in_gpu);	//First_it_in_gpu=1 iff execution not moved to GPU yet

		}
	    }
	
#pragma omp taskwait noflush
	  first_it_in_gpu = 0;	//We have already accessed the GPU at least once

#pragma omp target device (smp) copy_deps
#pragma omp task input ([ntasks]queue_pointer1,[ntasks]aux_queue_pointer)
	  {
	    //Empty task to copy the number of elements in the queue

	  }
#pragma omp taskwait noflush
	  elems_processed = 0;
	  for (i = 0; i < ntasks; i++)
	    elems_processed += queue_pointer1[i];


	
      printf ("Elements processed %d.\n", elems_processed);
      frontier++;
      printf ("------------------------\n\n");
    }
  while (elems_processed > 0 && frontier < 100);	//the value frontier < 100 is a prevention measure to avoid infinite loops.

  printf ("BFS traversal executed in %d exploration levels.\n", frontier);

}


int
main (int argc, char *argv[])
{
  double time1, time2;

  int minedgesize;
  int frontier;
  int i;
  int chunksize;
  FILE *fpo;

  int index = 0;
  int c;
  ntasks = 10;
  cpu_only = 1;
  dest = "output.txt";
  limit = 1000000000;
  orig_node = 0; 

  load_graph (argc, argv);

  N = no_of_nodes;
  chunksize = no_of_nodes / ntasks;
  MAX_LARGE_ARRAY_SIZE = chunksize;



//  N, MAX_LARGE_ARRAY_SIZE, N / MAX_LARGE_ARRAY_SIZE, no_of_nodes,
    frontier = 0;
  
#pragma omp taskwait noflush

      time1 = omp_get_wtime ();

      launch_bfs_gpu (ntasks, limit);
#pragma omp taskwait noflush
      time2 = omp_get_wtime ();
#pragma omp taskwait


  printf ("computation time (in seconds) with %d divisions: %f\n", ntasks,
	  time2 - time1);


  //Store the result into a file
  fpo = fopen (dest, "w");
  for (i = 0; i < no_of_nodes; i++)
    fprintf (fpo, "%d) cost:%d\n", i, graph_frontier[i]);
  fclose (fpo);
  if (1)
    printf ("Result stored in %s.txt\n", dest);
   /**/ return 0;
}


//Function used to read from disk and fill the data structures
void
load_graph (int argc, char **argv)
{
  int i, siz, k, j;
  char c;
  int o, d, w, prev, ptr;
  FILE *fpo;
  fp = fopen ("/scratch/aman/Project/GTgraph/R-MAT/sample2.gr", "r");    //need to change the file name 

  fscanf (fp, "%d %d", &no_of_nodes, &no_of_edges);

   
	     // if (no_of_nodes >= THREADBLOCK_SIZE * THREADBLOCK_SIZE * 16)ntasks=(no_of_nodes/(THREADBLOCK_SIZE * THREADBLOCK_SIZE * 16));

	
//Structure allocation
  visited_size = (no_of_nodes / 8) + 1;
  graph_node_start = (int *) malloc (sizeof (int) * no_of_nodes);
  graph_node_num = (int *) malloc (sizeof (int) * no_of_nodes);
  visited = (char *) malloc (sizeof (char) * visited_size);
  max_edge_chunk_size = (int *) malloc (sizeof (int) * ntasks);
  graph_edges = (int *) malloc (sizeof (int) * no_of_edges);
  queue1 = (int *) malloc (sizeof (int) * no_of_nodes);
  queue1_cpu = (int *) malloc (sizeof (int) * no_of_nodes);
  queue2 = (int *) malloc (sizeof (int) * no_of_nodes);
  queue2_cpu = (int *) malloc (sizeof (int) * no_of_nodes);
  queue_pointer1 = (int *) malloc (sizeof (int) * ntasks);
  queue_pointer1_cpu = (int *) malloc (sizeof (int) * ntasks);
  queue_pointer2_cpu = (int *) malloc (sizeof (int) * ntasks);
  aux = (int *) malloc (sizeof (int) * N);
  buf = (int *) malloc (sizeof (int) * BUFF_SIZE);
  graph_frontier = (int *) malloc (sizeof (int) * no_of_nodes);
  cache_ptr = (int **) malloc (sizeof (int *) * nthreads);
  cache = (int **) malloc (sizeof (int *) * nthreads);
  aux_queue_pointer = (int *) malloc (sizeof (int) * (no_of_nodes / (THREADBLOCK_SIZE * THREADBLOCK_SIZE * 4)));
  aux_queue = (int *) malloc (sizeof (int) * (no_of_nodes * 2));


  for (i = 0; i < visited_size; i++)
    visited[i] = 0;


  for (i = 0; i < no_of_nodes; i++)
    {

      graph_frontier[i] = -1;
      graph_node_num[i] = 0;
    }


  graph_node_start[0] = 0;
  k = 0;


  for (i = 0; i < no_of_edges; i++)
    {
      fscanf (fp, "%d %d", &o, &d);
      o--;
      d--;
      graph_node_num[o]++;


      k++;
      if (k == no_of_edges)
	break;

    }

  for (i = 1; i < no_of_nodes; i++)
    {

      queue2[i] = 0;
      graph_node_start[i] = graph_node_start[i - 1] + graph_node_num[i - 1];
      graph_node_num[i - 1] = 0;
    }

  graph_node_num[no_of_nodes - 1] = 0;

  graph_node_start[no_of_nodes] = no_of_edges;

  if (fp)
    fclose (fp);

  fp = fopen ("/scratch/aman/Project/GTgraph/R-MAT/sample2.gr", "r");
  if (!fp)
    {
      if (1)
	printf ("Error Reading graph file\n");
      return;
    }


  k = 0;
  fscanf (fp, "%d %d", &no_of_nodes, &no_of_edges);

  for (i = 0; i < no_of_edges; i++)
    {
      fscanf (fp, "%d %d", &o, &d);
      o--;
      d--;

      graph_edges[graph_node_start[o] + graph_node_num[o]] = d;
      graph_node_num[o]++;


      k++;
      if (k == no_of_edges)
	break;

    }
  if (fp)
    fclose (fp); 

  for (i = 0; i < no_of_nodes; i++)
    {
      qsort (&graph_edges[graph_node_start[i]],
	     graph_node_start[i + 1] - graph_node_start[i], sizeof (int),
	     sort);
    }
  i = orig_node / (no_of_nodes / ntasks);
  graph_frontier[orig_node] = 0;
  queue1_cpu[i] = orig_node;
  queue_pointer1_cpu[i] = 1;

  SETBIT (visited, (uint) orig_node);

}

#pragma omp target device(cuda) copy_deps
#pragma omp task input([chunksize+1]graph_node_start,[minedgesize]graph_edges,[q_in_values] queue_in)\
inout([no_of_nodes]queue2,[chunksize]queue1,[chunksize]graph_frontier,[ntasks]queue_pointer1,[visited_size]visited)
void
bfs_gpu (int *graph_node_start, int *graph_edges, int *queue1, int *queue2,
	 int *queue_pointer1, int *graph_frontier, char *visited,
	 int chunksize, int no_of_nodes, int minedgesize, int offset,
	 int ntasks, int frontier, int current_task,
	 int blocksize, int visited_size, int *queue_in, int q_in_values,
	 int first_it_in_gpu)
{




  bfs_kernel_gpu <<< blocksize, 512 >>> (graph_node_start, graph_edges,
					 queue1, queue2, queue_pointer1,
					 visited, graph_frontier, frontier,
					 offset, ntasks, chunksize,
					 current_task, queue_in, q_in_values,
					 first_it_in_gpu);

}

