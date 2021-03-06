#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 16

__global__ void Matrix_Mul_Kernel(float* d_M, float* d_N, float* d_P, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	for(int ph = 0; ph < Width/TILE_WIDTH; ++ph)
	{

		Mds[ty][tx] = d_M[Row * Width + ph * TILE_WIDTH + tx];
		Nds[ty][tx] = d_N[(ph * TILE_WIDTH + ty) * Width + Col];
		__syncthreads(); //Sincroniza todos los hilos en un bloque
		 //Asegúrarse de que todos los datos estén cargados.
			
		for (int k = 0; k < TILE_WIDTH; ++k){		

			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();//Evita los peligros de la memoria.
		//Asegurarse de que los calculos se realizen antes de la
		//siguiente fase
	}	

	d_P[Row * Width + Col] = Pvalue;
}


void cpu_matrix_mult(float *M, float *N, float *P, int Width) {
    for (int i = 0; i < Width; ++i) 
    {
        for (int j = 0; j < Width; ++j) 
        {
            int tmp = 0.0;
            for(int k = 0; k < Width; ++k) 
            {
                tmp += M[i * Width + k] * N[k * Width + j];
            }
            P[i * Width + j] = tmp;
        }
    }
}


int main()
{
    int Width = 1024;
    srand(3333);

    float *h_a=0, *h_b=0, *h_c=0, *h_cc=0;
    cudaMallocHost((void **) &h_a, sizeof(float)*Width*Width);
    cudaMallocHost((void **) &h_b, sizeof(float)*Width*Width);
    cudaMallocHost((void **) &h_c, sizeof(float)*Width*Width);
    cudaMallocHost((void **) &h_cc, sizeof(float)*Width*Width);	    
    if(h_a==0 || h_b==0 || h_c==0 || h_cc==0)
    {
	printf("No asignacion de memoria\n");
	return 1;
    }
    
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            h_a[i * Width + j] = rand()%1024;
        }
    }

    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            h_b[i * Width + j] = rand()%1024;
        }
    }

    float gpu_time_ms, cpu_time_ms;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    float *d_a=0, *d_b=0, *d_c=0;
    cudaMalloc((void **) &d_a, sizeof(float)*Width*Width);
    cudaMalloc((void **) &d_b, sizeof(float)*Width*Width);
    cudaMalloc((void **) &d_c, sizeof(float)*Width*Width);

    if(d_a==0 || d_b==0 || d_c==0)
    {
  	printf("No asignacion Gpu\n");
	return 1;
    }

    cudaMemcpy(d_a, h_a, sizeof(float)*Width*Width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*Width*Width, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);
    dim3 dimGrid((int)ceil(float(Width)/dimBlock.x), (int)ceil(float(Width)/dimBlock.y));

    Matrix_Mul_Kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, Width);

    cudaMemcpy(h_c, d_c, sizeof(float)*Width*Width, cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("Tiempo transcurrido en GPU: %f ms.\n\n", gpu_time_ms);


    //CPU version
    cudaEventRecord(start, 0);

    cpu_matrix_mult(h_a, h_b, h_cc, Width);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time_ms, start, stop);
	printf("Tiempo transcurrido en CPU: %f ms.\n\n", cpu_time_ms);
	

    //Validando resultados
    int all_ok = 1;
    for (int i = 0; i < Width; ++i)
    {
        for (int j = 0; j < Width; ++j)
        {
            if(h_c[i*Width + j] != h_cc[i*Width + j])
            {
                all_ok = 0;
            }
        }
    }

    if(all_ok)
    {
        printf("Todo bien!!, speedup = %f\n", cpu_time_ms / gpu_time_ms);
    }
    else
    {
        printf("Error\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}












