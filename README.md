CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Levi Cai
* Tested on: Windows 7, i7 @ 3.4GHz 16GB, Nvidia NVS 310 (Moore 100C Lab)

### Questions

![](images/comparison.PNG)

Above you can see the comparison between the run-times of all the different algorithms. Any initial and final cudamemcpy/malloc's are NOT included, however, any intermediate ones ARE, which is I believe a fairly large contributor to the fact that the algorithms are in general slower than the CPU implementation. I also believe that the array sizes are relatively small, so perhaps the overhead of launching CUDA programs out-weighs the parallelization (though I'm not certain about this to be honest).

In terms of the GPU implementations, the work-efficient algorithm is in general an order of magnitude faster than the naive version. This is most likely because of the necessary overhead of the sequential nature and increased number of operations of the algorithm in addition to the extra cudaMemcpy call in each iteration at each depth in my implementation. 

One thing to note is the Thrust (1) and Thrust (2). I noticed that the first thrust run in a single session took much longer than the second run (1st for a power-of-two and 2nd for a non-power-of-two), I am wondering if it is perhaps caching somewhere. It is clear that it is doing some kind of memory allocation from the timeline (session only contained a run from thrust):

![](images/timeline.PNG)

Overall sample output:

![](images/2_15_results.PNG)

## Radix Sort

Sample radix output with tests:

![](images/radix.PNG)

In terms of performance of the radix sort, it seems to be consistently ~0.05s regardless of the size of the array. I'm not sure why this is the case, as I do have several memcpy's in the implementation which I would expect to slow it down dramatically as the size of the array increases. Though perhaps this is an issue with cudaTiming on CPU malloc's vs. cudaMallocs, as there are few cudaMalloc's in my implementation.
