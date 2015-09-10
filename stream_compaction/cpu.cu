#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	odata[0] = 0;
	for (int i=1; i<n; i++){
		odata[i] = odata[i-1] + idata[i-1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int c = 0;
	for (int i=0; i<n; i++){
		if(idata[i] != 0){
			odata[c] = idata[i];
			c++;
		}
	}

    return c;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {

	// Temp
	int* temp = new int[n];
	for (int i=0; i<n; i++){
		if(idata[i] != 0){
			temp[i] = 1;
		} else {
			temp[i] = 0;
		}
	}

	// Scan
	int* scan_arr = new int[n];
	scan(n, scan_arr, temp);

	// Number of elements in the final array
	int c = scan_arr[n-1] + temp[n-1];

	// Scatter
	for(int i=0; i<n; i++){
		if (temp[i] == 1){
			int oind = scan_arr[i];
			odata[oind] = idata[i];
		}
	}
    return c;
}

}
}
