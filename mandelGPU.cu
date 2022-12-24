#include "PrototiposGPU.h"

// Macro para acceder al índice en row major.
#define IDX(i,j,lda) i*lda+j

// Para probar las distintas implementaciones, descomentar por bloques
// de "Opción X". Comentar las otras opciones y descomentar 
// la que se quiere probar. Cada kernel tiene una función de llamada asociada
// Las llamadas con memoria pinned y memoria convencional (opciones 6 y 7) utilizan el kernel 2D. 

/*////////////////////////////////////////////////
// MEMORIA UNIFICADA y 1D - OPCIÓN 1
__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

	double dx, dy, a, b, u, u_next, v, v_next, z2;
	int i, j, k;

	dx = (xmax-xmin)/xres;
	dy = (ymax-ymin)/yres;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	i = index / xres;
	j = index % xres;

	if (index < xres*yres) {
		b = j*dx + xmin;
		a = i*dy + ymin;
		u = 0;
		v = 0;
		k = 0;
		while(k < maxiter) {
			k += 1;
			z2 = u*u + v*v;
			if(z2 < 4) {

				u_next = u*u - v*v + b;
				v_next = 2*u*v + a;

				u = u_next;
				v = v_next;
				
			} else {
				break;
			}
		}
		if (k >= maxiter) {
			A[IDX(i,j,xres)] = 0;
		} else {
			A[IDX(i,j,xres)] = k;
		}
	}
}
*///////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA UNIFICADA y 2D - OPCIÓN 2
__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

	double dx, dy, a, b, u, u_next, v, v_next, z2;
	int i, j, k;

	dx = (xmax-xmin)/xres;
	dy = (ymax-ymin)/yres;

	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < yres && j < xres) {
		b = j*dx + xmin;
		a = i*dy + ymin;
		u = 0;
		v = 0;
		k = 0;
		while(k < maxiter) {
			k += 1;
			z2 = u*u + v*v;
			if(z2 < 4) {

				u_next = u*u - v*v + b;
				v_next = 2*u*v + a;

				u = u_next;
				v = v_next;
				
			} else {
				break;
			}
		}
		if (k >= maxiter) {
			A[IDX(i,j,xres)] = 0;
		} else {
			A[IDX(i,j,xres)] = k;
		}
	}
}
*///////////////////////////////////////////////////

/////////////////////////////////////////////////
// MEMORIA COMPARTIDA Y 1D - OPCIÓN 3
__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

	double a, b, u, u_next, v, v_next, z2;
	int i, j, k;

	extern __shared__ double shared[];

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	i = index / xres;
	j = index % xres;

	// Guardar dx y dy en memoria compartida.
    double *dx_s = shared, *dy_s = shared + 1;

	if(threadIdx.x == 0) {
		*dx_s = (xmax-xmin)/xres;
		*dy_s = (ymax-ymin)/yres;
	}
	__syncthreads();

	if (index < xres*yres) {
		b = j*(*dx_s) + xmin;
		a = i*(*dy_s) + ymin;
		u = 0;
		v = 0;
		k = 0;
		while(k < maxiter) {
			k += 1;
			z2 = u*u + v*v;
			if(z2 < 4) {

				u_next = u*u - v*v + b;
				v_next = 2*u*v + a;

				u = u_next;
				v = v_next;
				
			} else {
				break;
			}
		}
		if (k >= maxiter) {
			A[IDX(i,j,xres)] = 0;
		} else {
			A[IDX(i,j,xres)] = k;
		}
	}
}
///////////////////////////////////////////////////

/*///////////////////////////////////////////////
// MEMORIA COMPARTIDA (SHARED) y 2D - OPCIÓN 4
__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

	// Tamaño de la memoria compartida.
    extern __shared__ double shared[];

	double a, b, u, u_next, v, v_next, z2;
	int i, j, k;

	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;

	// Guardar dx y dy en memoria compartida.
    double *dx_s = shared, *dy_s = shared + 1;

	if(threadIdx.x == 0) {
		*dx_s = (xmax-xmin)/xres;
		*dy_s = (ymax-ymin)/yres;
	}

	__syncthreads();

	if (i < yres && j < xres) {
		b = j*(*dx_s) + xmin;
		a = i*(*dy_s) + ymin;
		u = 0;
		v = 0;
		k = 0;
		while(k < maxiter) {
			k += 1;
			z2 = u*u + v*v;
			if(z2 < 4) {

				u_next = u*u - v*v + b;
				v_next = 2*u*v + a;

				u = u_next;
				v = v_next;
				
			} else {
				break;
			}
		}

		if (k >= maxiter) {
			A[IDX(i, j, xres)] = 0;
		} else {
			A[IDX(i, j, xres)] = k;
		}
		
	}
}
*///////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA COMPARTIDA 2D Y BLOQUES - OPCIÓN 5 (NO FUNCIONA)
__global__ void kernelMandel(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A) {

	const int BLOCK_SIZE = blockDim.x;
	double a, b, u, u_next, v, v_next, z2, dx, dy;
	int i, j, k;

	dx = (xmax-xmin)/xres;
	dy = (ymax-ymin)/yres;

	__shared__ double shared_A[BLOCK_SIZE][BLOCK_SIZE];

	for (i = 0; i < yres; i += BLOCK_SIZE) {
		for (j = 0; j < xres; j += BLOCK_SIZE) {
			// Cargar un bloque de píxeles en la memoria compartida.
			int ii, jj;
			for (ii = 0; ii < BLOCK_SIZE; ++ii) {
				for (jj = 0; jj < BLOCK_SIZE; ++jj) {
					shared_A[ii][jj] = A[IDX(i + ii, j + jj, xres)];
				}
			}

			// Calcular el valor de cada pixel del bloque.
			int tid_y = threadIdx.y;
			int tid_x = threadIdx.x;
			if (i + tid_y < yres && j + tid_x < xres) {
				b = j*dx + xmin;
				a = i*dy + ymin;
				u = 0;
				v = 0;
				k = 0;
				while(k < maxiter) {
					k += 1;
					z2 = u*u + v*v;
					if(z2 < 4) {
					u_next = u*u - v*v + b;
					v_next = 2*u*v + a;
					u = u_next;
					v = v_next;
					} else {
						break;
					}
				}
				if (k >= maxiter) {
					shared_A[tid_y][tid_x] = 0;
				} else {
					shared_A[tid_y][tid_x] = k;
				}
			}

			// Escribir los resultados de la memoria compartida a la global.
			for (ii = 0; ii < BLOCK_SIZE; ++ii) {
				for (jj = 0; jj < BLOCK_SIZE; ++jj) {
					A[IDX(i + ii, j + jj, xres)] = shared_A[ii][jj];
				}
			}
		}
	}
}
*///////////////////////////////////////////////////

// Kernel externo (Manual)
__global__ void kernelPromedio(double *A, int xres, int yres, double *suma) {

	extern __shared__ double cache[];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	double tmp = 0.0;
	while (tid < xres*yres) {
		tmp += A[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = tmp;
	__syncthreads();

	int i = blockDim.x / 2;
	while(i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) {
		suma[blockIdx.x] = cache[0];
	}
}


// Kernel externo (Manual)
__global__ void kernelPromedioFinal(double *v, int N, double *res) {

	extern __shared__ double cache[];
	int tid = threadIdx.x;
	int cacheIndex = threadIdx.x;

	double tmp = 0.0;
	while (tid < N) {
		tmp += v[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = tmp;
	__syncthreads();

	int i = blockDim.x / 2;
	while(i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		res[0] = cache[0];
	}	
}


/////////////////////////////////////////////////
// MEMORIA UNIFICADA y 1D - OPCIÓN 1
__global__ void kernelBinariza(int xres, int yres, double* A, double med) {
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < xres * yres) {
		double elemento = A[idx];
		A[idx] = (elemento > med) ? 255 : 0;
	}
}
//////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA UNIFICADA y 2D - OPCIÓN 2
__global__ void kernelBinariza(int xres, int yres, double* A, double med) {
	
	int i, j;

	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < yres && j < xres) {
		A[IDX(i,j,xres)] = (A[IDX(i,j,xres)] > med) ? 255 : 0;
	}
}
*//////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA COMPARTIDA y 1D - OPCIÓN 3
__global__ void kernelBinariza(int xres, int yres, double* A, double med) {
	
	// Tamaño de la memoria compartida.
	extern __shared__ double sh_A[];

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Fila y columna de este hilo.
	int i, j;
	i = index / xres;
	j = index % xres;

	// No tiene sentido hacer esta copia a memoria compartida ya que sólo empeora el propio algoritmo, está hecha con fines académicos, para compararla con las demás.
	if (i < yres && j < xres) {
		sh_A[threadIdx.x] = A[IDX(i,j,xres)];
	}
	else {
		sh_A[threadIdx.x] = 0;
	}

	if (i < yres && j < xres) {
		A[IDX(i,j,xres)] = (sh_A[threadIdx.x] > med) ? 255 : 0;
	}
}
*//////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA COMPARTIDA y 2D - OPCIÓN 4 
__global__ void kernelBinariza(int xres, int yres, double* A, double med) {
	
	// Tamaño de la memoria compartida.
	extern __shared__ double sh_A[];

	int i, j;

	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < yres && j < xres) {
		sh_A[threadIdx.y * blockDim.x + threadIdx.x] = A[IDX(i,j,xres)];
	}
	else {
		sh_A[threadIdx.y * blockDim.x + threadIdx.x] = 0;
	}

	if (i < yres && j < xres) {
		A[IDX(i,j,xres)] = (sh_A[threadIdx.y * blockDim.x + threadIdx.x] > med) ? 255 : 0;
	}
}
*//////////////////////////////////////////////////

//
//########################################################################
//			LLAMADAS A FUNCIONES.
//########################################################################

/*////////////////////////////////////////////////
// MEMORIA UNIFICADA y 1D - OPCIÓN 1
extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk){
   
	// Memoria unificada.  
	double *Host_A = NULL;
	// Alojar memoria.
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));

	// Definir hilos por bloque y número de bloques.
	dim3 TPerBlk(ThpBlk);
	dim3 nBlocks((int)ceil((float) (xres*yres + ThpBlk - 1) / TPerBlk.x));
	// Llamada al kernel 2D.
	kernelMandel<<<nBlocks, TPerBlk>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, Host_A);

	// Copiar de vuelta la matriz al host desde la GPU. Liberar memoria.
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	CUDAERR(cudaFree(Host_A));
}
*///////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA UNIFICADA y 2D - OPCIÓN 2
extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk){
   
	// Memoria unificada.  
	double *Host_A = NULL;
	// Alojar memoria.
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));

	// Definir hilos por bloque y número de bloques.
	dim3 TPerBlk(ThpBlk, ThpBlk);
	dim3 nBlocks((int)ceil((float) xres / ThpBlk), (int)ceil((float) yres / ThpBlk));

	// Llamada al kernel 2D.
	kernelMandel<<<nBlocks, TPerBlk>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, Host_A);

	// Copiar de vuelta la matriz al host desde la GPU. Liberar memoria.
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	CUDAERR(cudaFree(Host_A));
}
*///////////////////////////////////////////////////

/////////////////////////////////////////////////
// MEMORIA COMPARTIDA Y 1D - OPCIÓN 3
extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk){
   
	// Memoria unificada.  
	double *Host_A = NULL;
	// Alojar memoria.
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));

	// Definir hilos por bloque y número de bloques.
	int numThreads = xres * yres;
   	int numBlocks = (numThreads + ThpBlk - 1) / ThpBlk;
	dim3 nBlocks(numBlocks);
	dim3 TPerBlk(ThpBlk);

	// Calcula el tamaño de la memoria compartida.
	int sharedMemorySize = 2 * sizeof(double);

	// Llamada al kernel 2D.
	kernelMandel<<<nBlocks, TPerBlk, sharedMemorySize>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, Host_A);

	// Copiar de vuelta la matriz al host desde la GPU. Liberar memoria.
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	CUDAERR(cudaFree(Host_A));
}
///////////////////////////////////////////////////

/*///////////////////////////////////////////////
// MEMORIA COMPARTIDA (SHARED) y 2D - OPCIÓN 4
extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk){
   
	double *Host_A = NULL;
	// Alojar y copiar memoria.
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));

	// Definir el tamaño de los bloques y la grid.
   	dim3 nBlocks((int)ceil((float) xres / ThpBlk), (int)ceil((float) yres / ThpBlk));
	dim3 TPerBlk(ThpBlk, ThpBlk);

	// Calcula el tamaño de la memoria compartida.
	//int sharedMemorySize = xres * yres * sizeof(double) + 2 * sizeof(double);

	int sharedMemorySize = 2 * sizeof(double);

	// Llamada al kernel 2D.
	kernelMandel<<<nBlocks, TPerBlk, sharedMemorySize>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, Host_A);
	// Copiar de vuelta la matriz al host desde la GPU. Liberar memoria.
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	CUDAERR(cudaFree(Host_A));
}
*///////////////////////////////////////////////////

/*///////////////////////////////////////////////
// MEMORIA COMPARTIDA 2D y BLOQUES - OPCIÓN 5
extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk){
   

   double *Host_A = new double[xres*yres];

   dim3 TPerBlk(ThpBlk, ThpBlk);
   dim3 nBlocks((int)ceil((float) xres / ThpBlk), (int)ceil((float) yres / ThpBlk));

   kernelMandel<<<nBlocks, TPerBlk>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, A);

   delete[] Host_A;
}
*///////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA PINNED y 2D - OPCIÓN 6
extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk){
   
	double *Host_A = NULL;
	double *Device_Mat = NULL;

	CUDAERR(cudaHostAlloc((void**)&Host_A, xres*yres*sizeof(double), cudaHostAllocMapped));
	CUDAERR(cudaHostGetDevicePointer((void**)&Device_Mat, (void *)Host_A, 0));

	// Definir hilos por bloque y número de bloques.
	dim3 TPerBlk(ThpBlk, ThpBlk);
	dim3 nBlocks((int)ceil((float) xres / ThpBlk), (int)ceil((float) yres / ThpBlk));

	// Llamada al kernel 2D.
	kernelMandel<<<nBlocks, TPerBlk>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, Device_Mat);

	// Copiar de vuelta la matriz al host desde la GPU. Liberar memoria.
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	CUDAERR(cudaFreeHost(Host_A));
}
*///////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA CONVENCIONAL y 2D - OPCIÓN 7
extern "C" void mandelGPU(double xmin, double ymin, double xmax, double ymax, int maxiter, int xres, int yres, double* A, int ThpBlk){
   
	double *Device_Mat;
	CUDAERR(cudaMalloc((void **) &Device_Mat, xres*yres*sizeof(double)));

	// Definir hilos por bloque y número de bloques.
	dim3 TPerBlk(ThpBlk, ThpBlk);
	dim3 nBlocks((int)ceil((float) xres / ThpBlk), (int)ceil((float) yres / ThpBlk));

	// Llamada al kernel 2D.
	kernelMandel<<<nBlocks, TPerBlk>>>(xmin, ymin, xmax, ymax, maxiter, xres, yres, Device_Mat);

	// Copiar de vuelta la matriz al host desde la GPU. Liberar memoria.
	CUDAERR(cudaMemcpy(A, Device_Mat, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	CUDAERR(cudaFree(Device_Mat));
}
*///////////////////////////////////////////////////

////////////////////////////////////////////
// CÁLCULO DEL PROMEDIO
// Secuencial
/*
extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk) {
	int i, j;
	double suma;
	suma = 0;
	//#pragma omp parallel for collapse(2) reduction(+ : suma)
	for(i = 0; i < yres; i++) {
	   for (j = 0; j < xres; j++) {
		  suma += A[IDX(i,j,xres)];
	   }
	}
	return suma/(xres*yres);
}
*/

// OpenMP
/*
extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk) {
	int i, j;
	double suma;
	suma = 0;
	#pragma omp parallel for collapse(2) reduction(+ : suma)
	for(i = 0; i < yres; i++) {
	   for (j = 0; j < xres; j++) {
		  suma += A[IDX(i,j,xres)];
	   }
	}
	return suma/(xres*yres);
}
*/

// CUDA cublasDasum
/*
extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk) {

	double *Host_A = NULL;
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Host_A, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	cublasCreate(&handle);
	double suma;
	cublasDasum(handle, xres*yres, Host_A, 1, &suma);

	cublasDestroy(handle);
	CUDAERR(cudaFree(Host_A));
	return suma / (xres*yres);
}
*/


// Kernel externo (Manual)
// LLAMADA 1 SÓLO BLOQUE.
/*
extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk) {

	double *Host_A = NULL;
	// Alojar y copiar memoria.
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Host_A, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));
	
	int numThreads = xres * yres;
   	int numBlocks = 1;
	dim3 nBlocks(numBlocks);
	dim3 TPerBlk(ThpBlk);

	// Calcula el tamaño de la memoria compartida.
	int memoriaCompartida = sizeof(double) * ThpBlk;
	
	double *suma = NULL;
	CUDAERR(cudaMallocManaged((void**)&suma, numBlocks*sizeof(double)));

	// Llamada al kernel.
	kernelPromedio<<<nBlocks, TPerBlk, memoriaCompartida>>>(Host_A, xres, yres, suma);
 
	// Liberar memoria.
	CUDAERR(cudaFree(Host_A));
	return suma[0] / (xres*yres);
}
*/


// Kernel externo (Manual) 
// LLAMADA MÁS DE UN BLOQUE
extern "C" double promedioGPU(int xres, int yres, double* A, int ThpBlk) {

	double *Host_A = NULL;
	// Alojar y copiar memoria.
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Host_A, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));

   	int numThreads = xres * yres;
   	int numBlocks = (numThreads + ThpBlk - 1) / ThpBlk;
	dim3 nBlocks(numBlocks);
	dim3 TPerBlk(ThpBlk);

	// Calcula el tamaño de la memoria compartida.
	int memoriaCompartida = sizeof(double) * ThpBlk;
	
	double *suma = NULL;
	CUDAERR(cudaMallocManaged((void**)&suma, numBlocks*sizeof(double)));
	double *res = NULL;
	CUDAERR(cudaMallocManaged((void**)&res, 1*sizeof(double)));

	// Llamada al kernel.
	kernelPromedio<<<nBlocks, TPerBlk, memoriaCompartida>>>(Host_A, xres, yres, suma);
	kernelPromedioFinal<<<1, TPerBlk, memoriaCompartida>>>(suma, numBlocks, res);
	cudaDeviceSynchronize();
	double result = res[0] / (xres*yres);

	// Liberar memoria.
	CUDAERR(cudaFree(Host_A));
	CUDAERR(cudaFree(res));
	CUDAERR(cudaFree(suma));
	return result;
}
////////////////////////////////////////////
// BINARIZADO

/////////////////////////////////////////////////
// MEMORIA UNIFICADA Y 1D - OPCIÓN 1
extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk) {

	double *Host_A = NULL;

	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Host_A, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));

	int numThreads = xres * yres;
   	int numBlocks = (numThreads + ThpBlk - 1) / ThpBlk;
	dim3 nBlocks(numBlocks);
	dim3 TPerBlk(ThpBlk);
 
	kernelBinariza<<<nBlocks, TPerBlk>>>(xres, yres, Host_A, med);
 
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
 
	CUDAERR(cudaFree(Host_A));
}
/////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA UNIFICADA Y 2D - OPCIÓN 2
extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk) {

	double *Host_A = NULL;

	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Host_A, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));

	dim3 TPerBlk(ThpBlk, ThpBlk);
   	dim3 nBlocks((int)ceil((float) xres / ThpBlk), (int)ceil((float) yres / ThpBlk));
 
	kernelBinariza<<<nBlocks, TPerBlk>>>(xres, yres, Host_A, med);
 
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
 
	CUDAERR(cudaFree(Host_A));
}
*/////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA COMPARTIDA Y 1D - OPCIÓN 3
extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk) {

	double *Host_A = NULL;

	// Alojar y copiar memoria.
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Host_A, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));

	int numThreads = xres * yres;
   	int numBlocks = (numThreads + ThpBlk - 1) / ThpBlk;
	dim3 nBlocks(numBlocks);
	dim3 TPerBlk(ThpBlk);

	// Calcula el tamaño de la memoria compartida.
	int sharedMemorySize = 1 * sizeof(double);

	// Llamada al kernel.
	kernelBinariza<<<nBlocks, TPerBlk, sharedMemorySize>>>(xres, yres, Host_A, med);
 
	// Copiar memoria de vuelta.
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	// Liberar memoria.
	CUDAERR(cudaFree(Host_A));
}
*/////////////////////////////////////////////////

/*////////////////////////////////////////////////
// MEMORIA COMPARTIDA y 2D - OPCIÓN 4
extern "C" void binarizaGPU(int xres, int yres, double* A, double med, int ThpBlk) {

	double *Host_A = NULL;

	// Alojar y copiar memoria.
	CUDAERR(cudaMallocManaged((void**)&Host_A, xres*yres*sizeof(double)));
	CUDAERR(cudaMemcpy(Host_A, A, xres*yres*sizeof(double), cudaMemcpyHostToDevice));

	// Definir el tamaño del bloque y de la grid.
   	dim3 nBlocks((int)ceil((float) xres / ThpBlk), (int)ceil((float) yres / ThpBlk));
	dim3 TPerBlk(ThpBlk, ThpBlk);

	// Calcula el tamaño de la memoria compartida.
	int sharedMemorySize = ThpBlk * ThpBlk * sizeof(double);

	// Llamada al kernel.
	kernelBinariza<<<nBlocks, TPerBlk, sharedMemorySize>>>(xres, yres, Host_A, med);
 
	// Copiar memoria de vuelta.
	CUDAERR(cudaMemcpy(A, Host_A, xres*yres*sizeof(double), cudaMemcpyDeviceToHost));
	// Liberar memoria.
	CUDAERR(cudaFree(Host_A));
}
*/////////////////////////////////////////////////