# Linear Regression Benchmark Results 

## Summary
```
   Results below show the performance comparison of linear regression with MXNet vs Keras-Tensorflow using sparse tensors
```                                                   

### Results
### Inference Benchmark
#### Input samples are sparse feature vectors with 10,000 dimensions
### Configuration
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.4                                                      |
| TensorFlow       | v1.11.0                                                     |
| MXNet-mkl         | v1.3.0   

#### CPU
##### Speed
| Instance Type | GPUs  | Batch Size  | Keras-MXNet Time/Batch) | Keras-TensorFlow Time/Batch)  |
|-----|-----|-----|-----|-----|
| C5.8XLarge |   0  | 64  | 1.4 sec | 1.3 sec
| C5.8XLarge |   0  | 128 | 0.9 sec | 0.7 sec 
| C5.8XLarge |   0  | 256 | 0.6 sec | 0.5 sec
| C5.8XLarge |   0  | 512 | 0.4 sec | 0.4 sec 
| C5.8XLarge |   0  | 1024 | 0.3 sec | 0.3 sec

#### Memory Consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (MB) | Keras-TensorFlow (MB)  |
|-----|-----|-----|-----|-----|
| C5.8XLarge |   0  | 64  | 1630.8 | 1573.8 |
| C5.8XLarge |   0  | 128 | 1574.7 | 1561.2 | 
| C5.8XLarge |   0  | 256 | 1477.8 | 1501.4  |
| C5.8XLarge |   0  | 512 | 1407.0| 1472.5 |
| C5.8XLarge |   0  | 1024 | 1336.3 | 1466.8 |

#### GPU
### Configuration
##### Input samples are sparse feature vectors with 10,000 dimensions
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.4                                                      |
| TensorFlow-GPU   | v1.11.0                                                     |
| MXNet-cu90mkl    | v1.3.0                                                      |

##### Single GPU
##### Speed
| Instance Type | GPUs  | Batch Size  | Keras-MXNet (Time/Batch) | Keras-TensorFlow Time/Batch)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 3.0 sec | 2.0 sec
| P3.8XLarge |   1  | 128 | 2.3 sec | 1.2 sec 
| P3.8XLarge |   1  | 256 | 1.2 sec | 0.7 sec
| P3.8XLarge |   1  | 512 | 0.8 sec | 0.5 sec
| P3.8XLarge |   1  | 1024 | 0.4 sec | 0.4 sec

##### GPU Utilization
| Instance Type | GPUs  | Batch Size | Keras-MXNet (GPU Utilization %) | Keras-TensorFlow (GPU Utilization %)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 9 | 7
| P3.8XLarge |   1  | 128 | 8 | 7
| P3.8XLarge |   1  | 256 | 8 | 7
| P3.8XLarge |   1  | 512 | 8 | 7
| P3.8XLarge |   1  | 1024 | 7 | 8

##### GPU Memory Consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (GPU Memory (MB)) | Keras-TensorFlow (GPU Memory (MB))  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 966.7 | 16135.5
| P3.8XLarge |   1  | 128 | 970.9 | 16135.5
| P3.8XLarge |   1  | 256 | 973.1 | 16135.5
| P3.8XLarge |   1  | 512 | 994.1 | 16135.5
| P3.8XLarge |   1  | 1024 | 987.7 | 16135.5

### Note
Run the file as `python run_sparse_benchmark.py`, by default the benchmark runs for `training` with `10 epochs` and batch size of `512`

### References
MXNet supports sparse data in 2 NDArray formats - CSRNDArray and RowSparseNDArray which are defined in `mxnet.ndarray.sparse` package
For further details on MXNet Sparse NDArray API check [documentation related to MXNet Sparse](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html)

Keras Input layer supports sparse data by setting a boolean placeholder value - check document for [Keras Input layer](https://keras.io/layers/core/#input)
