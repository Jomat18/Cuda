#include <stdio.h>
#include <stdlib.h>
#include <math.h>

 
//CUDA kernel
__global__ void vecAddKernel(float *a, float *b, float *c, int n)
{
    //ID del thread
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 

    //No salir del tamaño del vector
    if (id < n)
        c[id] = a[id] + b[id];
}

void vecAdd(float *a, float *b, float *c, int n);
 
int main( int argc, char* argv[] )
{
	
    int vector_size = 256*1024;	
 
    //host entradas
    float *h_a=0;
    float *h_b=0;
    //host salida
    float *h_c=0;

    //tamaño de cada vector	
    size_t bytes = vector_size*sizeof(float);
 
    //Asignacion de memoria en cpu
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    if(h_a==0 || h_b==0 || h_c==0)
    {
	printf("Error asignando memoria cpu\n");
	return 1;
    }
	
    //inicializar en host 	
    int i;
    for( i = 0; i < vector_size; i++) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }	

    vecAdd(h_a, h_b, h_c, vector_size);		

    bool sucess = true;

    for(i=0; i<vector_size; i++)
    {   
	if(h_a[i]+h_b[i]!=h_c[i])
	{
		sucess = false;
		break;	
	}
    }

    if(sucess)
	printf("Exitoooo en GPU!!\n");
 
    //Liberando memoria en el host
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}

void vecAdd(float *a, float *b, float *c, int n)	
{	
	size_t bytes = n*sizeof(float);	

    	//Device entradas
    	float *d_a=0;
    	float *d_b=0;
    	//Device salida
    	float *d_c=0;
  
    	//Asignacion de memoria en gpu
    	cudaMalloc((void **) &d_a, bytes);
    	cudaMalloc((void **) &d_b, bytes);
    	cudaMalloc((void **) &d_c, bytes);
 
	if(d_a==0 || d_b==0 || d_c==0)
	{
		printf("Error asignando memoria gpu\n");
	}


    	//Copia de host a device
    	cudaMemcpy( d_a, a, bytes, cudaMemcpyHostToDevice);
    	cudaMemcpy( d_b, b, bytes, cudaMemcpyHostToDevice);
 
    	int blockSize, gridSize;
 
    	//Numero de threads por bloque
    	blockSize = 256;
 
    	//Numero de bloques de threads 
    	gridSize = (int)ceil((float)n/blockSize);
 
    	//Ejecucion
    	vecAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
        //Copiando al host los resultados
    	cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost );
 
	//Liberando memoria en el device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);	
}

