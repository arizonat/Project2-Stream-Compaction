#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__
__global__ void kernScan(int n, int powd, int* odata, int* idata){
	int i = threadIdx.x;
	//int powd = (int)pow(2,d);

	if (i < n){
		if(i >= powd){
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

	int* hst_idata2 = new int[n2]();
	memcpy(hst_idata2, idata, n_size);

	// Scan
	int* dev_idata;
	int* dev_odata;
	cudaMalloc((void**)&dev_idata, n2_size);
	cudaMemcpy(dev_idata, hst_idata2, n2_size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_odata, n2_size);

	for(int d=0; d<td; d++){
		int powd = (int)pow(2,d);
		kernScan<<<1,n2>>>(n2, powd, dev_odata, dev_idata);
		dev_idata = dev_odata;
	}

	// Remove leftover (from the log-rounded portion)
	// Do a shift right to make it an exclusive sum
	odata[0] = 0;
	cudaMemcpy(odata+1, dev_odata, n_size-sizeof(int), cudaMemcpyDeviceToHost);
}

}
}
