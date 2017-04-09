#ifdef __CUDACC__
#include "sm_11_atomic_functions.h"

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
#define THREADBLOCK_SIZE 512


#pragma omp target device(cuda)

#if(0)
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size){
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    
    for(uint offset = 1; offset < size; offset <<= 1){
        __syncthreads();
        uint t = s_Data[pos] + s_Data[pos - offset];
        __syncthreads();
        s_Data[pos] = t;
    }
    
    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size){
    return scan1Inclusive(idata, s_Data, size) - idata;
}

#else
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

//Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
//assuming size <= WARP_SIZE
inline __device__ uint warpScanInclusive(uint idata, volatile uint *s_Data, uint size){
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    
    for(uint offset = 1; offset < size; offset <<= 1)
        s_Data[pos] += s_Data[pos - offset];
    
    return s_Data[pos];
}

inline __device__ uint warpScanExclusive(uint idata, volatile uint *s_Data, uint size){
    return warpScanInclusive(idata, s_Data, size) - idata;
}

inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size){
    if(size > WARP_SIZE){
        //Bottom-level inclusive warp scan
        uint warpResult = warpScanInclusive(idata, s_Data, WARP_SIZE);
        
        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        if( (threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
            s_Data[threadIdx.x >> LOG2_WARP_SIZE] = warpResult;
        
        //wait for warp scans to complete
        __syncthreads();
        if( threadIdx.x < (THREADBLOCK_SIZE / WARP_SIZE) ){
            //grab top warp elements
            uint val = s_Data[threadIdx.x];
            //calculate exclsive scan and write back to shared memory
            s_Data[threadIdx.x] = warpScanExclusive(val, s_Data, size >> LOG2_WARP_SIZE);
        }
        
        //return updated warp scans with exclusive scan results
        __syncthreads();
        return warpResult + s_Data[threadIdx.x >> LOG2_WARP_SIZE];
    }else{
        return warpScanInclusive(idata, s_Data, size);
    }
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size){
    return scan1Inclusive(idata, s_Data, size) - idata;
}

#endif

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size){
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;
    
    //Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4);
    
    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;
    
    return idata4;
}

inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size){
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

__global__ void scan_kernel(uint4 *d_Dst, uint4 *d_Src,uint size ){
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];
    
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    //Load data
    uint4 idata4 = d_Src[pos];
    
    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size);
    
    //Write back
    d_Dst[pos] = odata4;
}

__global__ void scan_kernel2(
                                     uint *d_Buf,
                                     uint *d_Dst,
                                     uint *d_Src,
                                     uint N,
                                     uint arrayLength
                                     ){
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];
    
    //Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    uint idata = 0;
    if(pos < N)
        idata = 
        d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] + 
        d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];
    
    //Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength);
    
    //Avoid out-of-bound access
    if(pos < N)
        d_Buf[pos] = odata;
}
__global__ void uniformUpdate(uint4 *d_Data,uint *d_Buffer,uint4* mask, uint* output,uint* sizes, uint max_array_size, uint offset){
    __shared__ uint buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint factor = blockIdx.x/512;
    uint val = (pos*4);
    if(threadIdx.x == 0){
        buf = d_Buffer[blockIdx.x];
        
    }
    __syncthreads();
   
   
    uint4 data4 = d_Data[pos];
  

    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
    uint4 mask4 =mask[pos];
    if (mask4.x==1)output[data4.x+factor*1048576]=val+offset;
    if(mask4.y==1)output[data4.y+factor*1048576]=val+1+offset;
    if(mask4.z==1)output[data4.z+factor*1048576]=val+2+offset;
    if(mask4.w==1)output[data4.w+factor*1048576]=val+3+offset;
    
    __syncthreads();

    if(pos+1==((factor+1)*(max_array_size/4)))sizes[factor]=data4.w+(mask4.w==1);
        mask4.x=0;
    mask4.y=0;
    mask4.z=0;
    mask4.w=0;
    mask[pos]=mask4;
    
}

__global__ void unify(uint *output, uint* sizes, uint4 *aux_queue, uint *aux_queue_pointer){
    __shared__ uint offset;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
	uint individual_pos = ((blockIdx.x%512) * blockDim.x + threadIdx.x)*4;
	uint chunksize_pos=blockIdx.x/512;
    uint limit = aux_queue_pointer[chunksize_pos];

	
    if(threadIdx.x == 0){
	   offset=0;
      
	   uint i;
		for (i=0;i<chunksize_pos;i++){
			offset+=aux_queue_pointer[i];
		}
        if (blockIdx.x+1==gridDim.x) sizes[0]=offset+aux_queue_pointer[chunksize_pos];
		
    }
    __syncthreads();
	uint4 data4 = aux_queue[pos];
	
    if(individual_pos<limit) output[offset+individual_pos]=data4.x;
    if(individual_pos+1<limit)output[offset+individual_pos+1]=data4.y;
    if(individual_pos+2<limit)output[offset+individual_pos+2]=data4.z;
    if(individual_pos+3<limit)output[offset+individual_pos+3]=data4.w;
	
	
}

/*
    if(individual_pos<limit) output[offset+individual_pos]=data4.x;
    if(individual_pos+1<limit)output[offset+individual_pos+1]=data4.y;
    if(individual_pos+2<limit)output[offset+individual_pos+2]=data4.z;
    if(individual_pos+3<limit)output[offset+individual_pos+3]=data4.w;


*/

__global__ void scan_kernel_empty(uint4 *d_Dst, uint4 *d_Src,uint size ){
   
}

__global__ void scan_kernel2_empty(
                             uint *d_Buf,
                             uint *d_Dst,
                             uint *d_Src,
                             uint N,
                             uint arrayLength
                             ){
    
}
__global__ void uniformUpdate_empty(uint4 *d_Data,uint *d_Buffer,uint4* mask, uint* output,uint* sizes, uint max_array_size){
   
    
}

#endif
