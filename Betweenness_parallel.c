#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int front=-1,front1=-1,rear=-1,rear1=-1,max,*queue_array,*queue_array1;
int main()
{
    
    int i,j,k,l=0,p=0,q=0,v=0,element,index=0,w=0,z=0,t,sum=0;
    int *x,*y,*A,*dist,*path,*predcessor,*neighbour;
    double *centrality,*dependency,Average;
    int partition=0;
    char a,b,c,d,sp[4];
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    struct timeval tv;
    double start_t, end_t;
    fp=fopen("/home/aman/GTgraph/R-MAT/output","r");
    for(i=0; i<7; i++)
      {
        read=getline(&line, &len, fp);
      }

     max=0;

    fscanf(fp, "%c %s %d %d\n", &a,sp,&max,&p);
    partition=4*log10(max);
    int size=max*partition;
    x=(int*)malloc(p*sizeof(int));
    y=(int*)malloc(p*sizeof(int));
    dist=(int*)malloc((max+1)*sizeof(int));
    path=(int*)calloc((max+1),sizeof(int));
    queue_array=(int*)calloc(max,sizeof(int));
    queue_array1=(int*)calloc(max,sizeof(int));
    neighbour=(int*)calloc(max,sizeof(int));
    centrality=(double*)calloc((max+1),sizeof(double));
    predcessor=(int*)calloc(size,sizeof(int));
    dependency=(double*)malloc((max+1)*sizeof(double));
      while(fscanf(fp, "%c %d %d %d\n", &a,&i,&j,&k)!= -1)
      {
        x[l]=i;
        y[l]=j;
        l++;
       }
     printf("max=%d\n",max);
    A=(int*)calloc((max+1),sizeof(int));
      for(i=0; i<=max; i++)
       {
        dist[i]=-1;
        dependency[i]=0.0;
       }

      for (i=0; i<p; i++)
      {
        if(A[x[i]]==0)
         {
            A[x[i]]=i+1;
         }
      } 
gettimeofday(&tv,0);
start_t=(double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;   

//BFS TRAVERSAL
#pragma omp parallel for shared(k)
  for(k=1;k<=max;k++)
    {  
        #pragma omp parallel for private(i) shared(max,path,dist)
	for(i=0; i<=max; i++)
         {
        path[i]=0;
        dist[i]=-1;
         }
        path[k]=1;
        dist[k]=0;
        enqueue(k);
        
        
       
      //while(front<=rear)
//      #pragma omp parallel for shared(v) private(i,neighbour,t,j)
      for(front=0;front<=rear;)
      {  
           v=dequeue();
           push(v);
           i=A[v]-1;
           t=0;
            if(i>=0)
              {
		j=x[i];
                while(j==x[i])
                  {
                     neighbour[t]=y[i];
 		     t++;
                     i++;
                  }
               }
              
       #pragma omp parallel for private(j) shared(v,t,dist,partition,predcessor)
              for(j=0;j<t;j++)
               {
                if(dist[neighbour[j]]<0)
                   {
                      enqueue(neighbour[j]);
                      dist[neighbour[j]]=dist[v]+1;
                    }
                 if(dist[neighbour[j]]==dist[v]+1)
                    {
			 //printf("path between %d and %d is %d\n",k ,y[i],v);
                     path[neighbour[j]]=path[neighbour[j]]+path[v];
                     index=(neighbour[j]-1)*partition;
                        if(predcessor[index]==0)
                        {
                        predcessor[index]=v;
                        }
                        else{
                              while(predcessor[index]!=0)
                              {
                                 index++;
                              }
                               predcessor[index]=v;
                            }
                     }
                    
                     // i++;
                }  
      
            }
           
       
         #pragma omp parallel for private(i) shared(max,dependency)  
         for(i=0;i<=max;i++)
           { 
               dependency[i]=0.0;
           }
           

          //FINDING THE BETWEENNESS CENTRALITY.
           
          
           for(front1=0;front1<=rear1;)
           {
                 w= pop();
                 z=(w-1)*partition;
                 int limit=w*partition;
               #pragma omp parallel for private(i) shared(z,limit,w)
                 for(i=z; i<limit;i++)
                   { 
                
                     if(predcessor[i]!=0)
                       {
                         dependency[predcessor[i]]=dependency[predcessor[i]]+ ((double)(path[predcessor[i]]/path[w]))*(1+dependency[w]);
                   
                       }
                   }

              if(w!=k)
               {
                 centrality[w]=centrality[w]+dependency[w];
               }

            }
            #pragma omp parallel for private(i) shared(size,predcessor)
           for(i=0;i<size;i++)
             { 
               predcessor[i]=0;
             }   
             
      front=-1;
     rear=-1;
      front1=-1;
      rear1=-1;
}

#pragma omp parallel for private(i) shared(max,centrality) reduction(+:sum)
for(i=1;i<=max;i++)
   {
      sum=sum+centrality[i];
   }
  printf("The Betweenness centrality for each node is =\n");

#pragma omp parallel for private(i,Average) shared(max,sum,centrality)
 for(i=1;i<=max;i++)
   { 
      Average=centrality[i]/sum;
     printf(" centrality for %d is = %f \n",i,Average);
      // printf("%f ",Average);
   }                           

gettimeofday(&tv,0);
end_t=(double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
printf("time taken= %lf",end_t-start_t);
            fclose(fp);
            return 0;

}



enqueue(int element)
{

    if (rear == max-1)
        printf("Queue Overflow 1\n");
    else
    {
        if (front == - 1)
        front = 0;
        rear = rear + 1;
        queue_array[rear] = element;
    }
}

push(int element)
{

    if (rear1 == max-1)
        printf("Queue Overflow 2\n");
    else
    {
        if (front1 == - 1)
        front1 = 0;
        rear1 = rear1 + 1;
        queue_array1[rear1] = element;
    }
}

int pop()
{
        int i;
    if (front1 == - 1 || front1 > rear1)
    {
        printf("Queue Underflow \n");
        return ;
    }
    else
    {
        rear1 = rear1-1;
        return (queue_array1[rear1+1]);
    }
}

int dequeue()
{
        int i;
    if (front == - 1 || front > rear)
    {
        printf("Queue Underflow \n");
        return ;
    }
    else
    {
        front = front + 1;
        return (queue_array[front-1]);
    }
}
























