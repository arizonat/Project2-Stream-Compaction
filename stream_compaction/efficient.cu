#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void upsweep(int n, int powd, int powd1, int* idata){
	int k = threadIdx.x;

	if (k < n){
		if (k % (powd1) == 0){
			idata[k + powd1 - 1] += idata[k + powd - 1];
		}
	}
}

__global__ void downsweep(int n, int powd, int powd1, int* idata){
	int k = threadIdx.x;

	if (k < n){
		if (k % (powd1) == 0){
			int t = idata[k + powd - 1];
			idata[k + powd - 1] = idata[k + powd1 - 1];
			idata[k + powd1 - 1] += t;
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
	cudaMalloc((void**)&dev_idata, n2_size);
	cudaMemcpy(dev_idata, hst_idata2, n2_size, cudaMemcpyHostToDevice);

	int powd, powd1;
	for(int d=0; d<td; d++){
		powd = (int)pow(2,d);
		powd1 = (int)pow(2,d+1);
		upsweep<<<1,n2>>>(n2, powd, powd1, dev_idata);
	}

	cudaMemset((void*)&dev_idata[n2-1],0,sizeof(int));
	for(int d=td-1; d>=0; d--){
		powd = (int)pow(2,d);
		powd1 = (int)pow(2,d+1);
		downsweep<<<1,n2>>>(n2, powd, powd1, dev_idata);
	}

	// Remove leftover (from the log-rounded portion)
	// No need to shift in this one I guess?
	cudaMemcpy(odata, dev_idata, n_size, cudaMemcpyDeviceToHost);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    // TODO
    return -1;
}

}
}
