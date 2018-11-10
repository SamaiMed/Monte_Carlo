/**************************************
***************************************
* Code Can be compiled using --> nvcc kernel5.cu -lcurand if the cuRand lib is the envirement PATH
* else use nvcc kernel5.cu -L</path/to/the/lib> -lcurand 
***************************************
**************************************/


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void MC_test(unsigned int seed,curandState *states,unsigned int numsim,unsigned int *results)
{
    extern __shared__ int sdata[];
    int i;
    int nthreads = gridDim.x * blockDim.x;
    unsigned int innerpoint=0;
    int tx=threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tx;
    curandState *state =states + idx;
    float x,y,l2norm2;
    sdata[tx]=0;
    __syncthreads();
    curand_init(seed, tx, 0, state);
    __syncthreads();
    for(i=tx;i<numsim;i+=nthreads){
         x = curand_uniform(state);
         y = curand_uniform(state);
         l2norm2 = x * x + y * y;
        if (l2norm2 < static_cast<float>(1))
            {
                innerpoint++;;
            }
    }   
    __syncthreads();
    sdata[tx]=innerpoint;
    __syncthreads();
    //-------reduction
    for (unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tx < s){
            sdata[tx]=sdata[tx]+sdata[tx+s];
        }
    }
    //-----------------
    __syncthreads();   
    if(tx==0){
    results[blockIdx.x]=sdata[0];
    }
   
}
void caller(unsigned int numsim){
    static curandState *states=NULL;
    unsigned int *results;
    unsigned int seed=rand();
    float pi=0;
    float r_pi= 3.14159265358979323846;
    dim3 block;
    dim3 grid;
    block.x=1<<10;
    grid.x=2;//=(numsim +block.x -1)/block.x; //ceil((float)numsim/(float)(block.x));
    printf(" \n grid %d block %d  ",grid.x,block.x);
    cudaMallocManaged(&states,sizeof(curandState)*block.x * grid.x);
    cudaMallocManaged(&results,2*sizeof(unsigned int));
    results[0]=0;
    results[1]=0;
    MC_test<<<grid , block, block.x*sizeof(unsigned int)>>>(seed,states,numsim,results);
    cudaDeviceSynchronize();
    pi=4*(float)(results[0]+results[1])/(float)(numsim);
    printf(":: sims= %d, MC_pi= %f , error= %f  \t",numsim,pi,abs(pi-r_pi));
    cudaFree(states);
}

int main(){
 unsigned int N=50;
 for (int i=1; i < N ;i++){
    caller(1<<i);
 }
 printf("\n");
    return 0;
}






