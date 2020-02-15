#include <stdio.h>

int main() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	printf("N dispositivos: %d\n",nDevices);
	cudaDeviceProp prop;

	for (int i = 0; i < nDevices; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Size warp: %d\n", prop.warpSize);
		printf("  Memoria compartida por bloque: %d KB \n", prop.sharedMemPerBlock/1024);		
		printf("  registros por bloque: %d\n", prop.regsPerBlock);
		printf("  Max hilos por bloque: %d\n", prop.maxThreadsPerBlock);
		printf("  Max dimension de hilos en x: %d\n", prop.maxThreadsDim[0]);
		printf("  Max dimension de hilos en y: %d\n", prop.maxThreadsDim[1]);
		printf("  Max dimension de hilos en z: %d\n", prop.maxThreadsDim[2]);	
		printf("  Max size grid en x: %d\n", prop.maxGridSize[0]);
		printf("  Max size grid en y: %d\n", prop.maxGridSize[1]);
		printf("  Max size grid en z: %d\n", prop.maxGridSize[2]);
		printf("  Frecuencia del reloj del device (KHz): %d\n", prop.clockRate);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
 		printf("  Numero de SMs %d\n", prop.multiProcessorCount);
		printf("  Max numero de threads por SM:  %d\n", prop.maxThreadsPerMultiProcessor);
	}
}
