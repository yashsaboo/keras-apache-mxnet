# Linear Regression Benchmark Results 

## Summary
```
   Results below show the performance comparison of linear regression with MXNet vs Keras-Tensorflow using sparse tensors
```

### Configuration
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.2                                                      |
| TensorFlow       | v1.10.0                                                      |
| MXNet            | v1.3.0                                                      |


### Results 
##### Sparse data
###### Using 25 epochs
###### Benchmark results calculated using an average of 5 runs
| Instance Type | GPUs  | Batch Size  | MXNet (Time/Epoch) | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|-----|
| C5.18XLarge |   0  | 16  | 4.97 sec | 31.02 sec | 21.05 sec |
| C5.18XLarge |   0  | 32 | 2.48 sec | 15.36 sec | 10.68 sec |
| C5.18XLarge |   0  | 64  | 1.36 sec | 8.12 sec | 5.41 sec |
| C5.18XLarge |   0  | 128  | 0.69 sec | 3.89 sec |  2.86 sec |


### Note
For reproducing above results set seed to `7` by adding this line in the `run_sparse_benchmark` script - `np.random.seed(7)`
Run the file as `python run_sparse_benchmark.py`

### References
MXNet supports sparse data in 2 NDArray formats - CSRNDArray and RowSparseNDArray which are defined in `mxnet.ndarray.sparse` package
For further details on MXNet Sparse NDArray API check [documentation related to MXNet Sparse](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html)

Keras Input layer supports sparse data by setting a boolean placeholder value - check document for [Keras Input layer](https://keras.io/layers/core/#input)
