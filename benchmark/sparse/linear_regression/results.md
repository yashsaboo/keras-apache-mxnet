# Linear Regression Benchmark Results 

## Summary
```
   Results below show the performance comparison of linear regression with MXNet vs Keras-Tensorflow using sparse tensors
```

### Configuration
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.2                                                      |
| TensorFlow       | v1.9.0                                                      |
| MXNet            | v1.2.1                                                       |


### Results 
| Instance Type | GPUs  | Batch Size  | MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|----- |-----|
|  C5.18X Large |   0  | 16  | 51.15 sec | 71.85 sec |
|  C5.18X Large |   0  | 32 | 20.82 sec | 54.31 sec |
|  C5.18X Large |   0  | 64  | 13.13 sec | 19.20 sec |
|  C5.18X Large |   0  | 128  | 5.72 sec | 9.86 sec |

### Note
For reproducing above results set seed to `7` by adding this line in the `run_sparse_benchmark` script - `np.random.seed(7)`

Run the file as `python run_sparse_benchmark.py --batch=128 --epochs=1000`

### References
MXNet supports sparse data in 2 NDArray formats - CSRNDArray and RowSparseNDArray which are defined in `mxnet.ndarray.sparse` package
For further details on MXNet Sparse NDArray API check [documentation related to MXNet Sparse](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html)

Keras Input layer supports sparse data by setting a boolean placeholder value - check document for [Keras Input layer](https://keras.io/layers/core/#input)
