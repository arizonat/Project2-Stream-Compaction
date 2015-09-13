#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__
__global__ void kernScan(int n, int powd, int* odata, int* idata){
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	//int powd = (int)pow(2,d);

	if (i < n){
		if (i >= powd){
			odata[i] = idata[i - powd] + idata[i];
		} else {
			odata[i] = idata[i];
		}
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	// Compute log-rounded n
	
	int td = ilog2ceil(n);
	int n2 = (int)pow(2,td);

	int n_size = n * sizeof(int);
	int n2_size = n2 * sizeof(int);

	// Scan
	int* dev_idata;
	int* dev_odata;

	cudaMalloc((void**)&dev_idata, n2_size);
	cudaMalloc((void**)&dev_odata, n2_size);
	cudaMemcpy(dev_idata, idata, n_size, cudaMemcpyHostToDevice);
	cudaMemset(dev_idata+n, 0, n2_size-n_size);
	
	int numBlocks = (n2-1) / MAXTHREADS + 1;

	for(int d=1; d<=td; d++){
		int powd = 1 << (d-1);
		//cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		kernScan<<<numBlocks,MAXTHREADS>>>(n2, powd, dev_odata, dev_idata);
		//cudaDeviceSynchronize();
		cudaThreadSynchronize();

		//cudaMemcpy(odata,dev_odata,2,cudaMemcpyDeviceToHost);
		//odata[0] = 0;
		//cudaMemcpy(odata + 1, dev_odata, n_size - sizeof(int), cudaMemcpyDeviceToHost);
		//printf("Yeah...\n");
		//for (int i = 1024; i < n; i++){
		//	printf("%d ", odata[i]);
		//}
		//printf("\n\n");
		dev_idata = dev_odata;
	}

	// Remove leftover (from the log-rounded portion)
	// Do a shift right to make it an exclusive sum
	odata[0] = 0;
	cudaMemcpy(odata+1, dev_odata, n_size-sizeof(int), cudaMemcpyDeviceToHost);
}

}
}
