#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	
	thrust::host_vector<int> hst_v_in(idata,idata+n);
	thrust::device_vector<int> v_in = hst_v_in;
	thrust::device_vector<int> v_out(n);
	thrust::exclusive_scan(v_in.begin(), v_in.end(), v_in.begin());

	thrust::host_vector<int> hst_v_out = v_in;

	for (int i=0; i<n; i++){
		printf("%d ", hst_v_out[i]);
		odata[i] = hst_v_out[i];
	}
	printf("\n");

	//odata = &hst_v_out[0];
	//odata = thrust::raw_pointer_cast(&hst_v_out[0]);
}

}
}
