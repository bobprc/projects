Usage:

To correctness-test the C++ and CUDA code, you can do 

```bash
python test_correctness.py
```
This will generate some test data, run a known-good python implementation on it,
and compare the results to those generated by the new code. The test data consists of
N boxes centered around M points - 1000 boxes per point, around 6 points by default -
but you can specify N and M by via the --centroids and --per-centroid args. This
version can deal with multiple batches at once, and you can control the test
data's batch size via --batch-size. Also you can change the suppression threshold
with --thresh.

To benchmark, do a 
```bash
python benchmark.py
```
Again, this generates some data and runs various implementations on it. You can
adjust the test data in the same way as detailed above. The "Python version" is
in plain numpy, "C++" is the cpu implementation I'm proposing to add, and "C++ CUDA"
the gpu version. You can also include a version written in PyTorch acting on
CUDA tensors in the tests by passing --test-pytorch-impl (this will set the batch
size to 1). On my machine (intel i5 CPU, 1050ti) I find the C++ version to be
about twice as fast as the python version, and the CUDA version to be at least one,
and sometimes more than two, orders of magnitude faster - although you have to run
the script a couple of times to see this; the gpu needs to warm up (or something
like that). You can set the number of timing loops the script runs with --loops.

