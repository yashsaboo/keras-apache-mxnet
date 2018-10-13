# Linear Regression Benchmark Results 

## Summary
```
   Results below show the performance comparison of linear regression with MXNet vs Keras-Tensorflow using sparse tensors
```                                                   

### Results
### Training Benchmark
#### Input samples are sparse feature vectors with 10,000 dimensions
### Configuration
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.4                                                      |
| TensorFlow       | v1.11.0                                                     |
| MXNet-mkl         | v1.3.0   

###### Using 10 epochs
#### CPU
##### Speed
###### Note speed calculated using an average of 5 runs
| Instance Type | GPUs  | Batch Size  | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|
| C5.8XLarge |   0  | 64  | 715.4 sec | 676.6 sec
| C5.8XLarge |   0  | 128 | 346.2 sec | 341.1 sec 
| C5.8XLarge |   0  | 256 | 168.9 sec | 165.5 sec
| C5.8XLarge |   0  | 512 | 89.3 sec | 83.8 sec 
| C5.8XLarge |   0  | 1024 | 50.5 sec | 48.6 sec

#### CPU Memory Consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (CPU Memory (MB)) | Keras-TensorFlow (CPU Memory (MB))  |
|-----|-----|-----|-----|-----|
| C5.8XLarge |   0  | 64  | 1331.3 | 1578.2 |
| C5.8XLarge |   0  | 128 | 1387.3 | 1472.1 
| C5.8XLarge |   0  | 256 | 1431.3 | 1511.2  |
| C5.8XLarge |   0  | 512 | 1458.2 | 1506.5 |
| C5.8XLarge |   0  | 1024 | 1247.8 | 1329.1 |

#### GPU
### Configuration
##### Input samples are sparse feature vectors with 10,000 dimensions
| Dataset          | Synthetic(Randomly generated)                                |
| :--------------- | :----------------------------------------------------------- |
| Keras            | v2.2.4                                                      |
| TensorFlow-GPU   | v1.11.0                                                      |
| MXNet-cu90mkl    | v1.3.0   

###### Using 10 epochs
##### Single GPU
##### Speed
###### Note speed calculated using an average of 5 runs
| Instance Type | GPUs  | Batch Size  | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 50.9 sec | 49.9 sec
| P3.8XLarge |   1  | 128 | 34.8 sec | 33.5 sec 
| P3.8XLarge |   1  | 256 | 18.8 sec | 23.3 sec
| P3.8XLarge |   1  | 512 | 12.4 sec | 19.0 sec
| P3.8XLarge |   1  | 1024 | 9.1 sec | 18.8 sec

##### GPU Utilization
| Instance Type | GPUs  | Batch Size | Keras-MXNet (GPU Utilization %) | Keras-TensorFlow (GPU Utilization %)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 10 | 9
| P3.8XLarge |   1  | 128 | 10 | 8
| P3.8XLarge |   1  | 256 | 10 | 8
| P3.8XLarge |   1  | 512 | 9 | 8
| P3.8XLarge |   1  | 1024 | 8 | 8

##### GPU Memory Consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (GPU Memory (MB)) | Keras-TensorFlow (GPU Memory (MB))  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   1  | 64  | 966.7 | 16135.5
| P3.8XLarge |   1  | 128 | 966.7 | 16135.5
| P3.8XLarge |   1  | 256 | 970.9 | 16135.5
| P3.8XLarge |   1  | 512 | 999.6 | 16135.5
| P3.8XLarge |   1  | 1024 | 991.9 | 16135.5

##### Multi-GPU
##### Speed
###### Benchmark results on multi GPU calculated using an average of 5 runs
| Instance Type | GPUs  | Batch Size | Keras-MXNet (Time/Epoch) | Keras-TensorFlow (Time/Epoch)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   2  | 512 | 19 sec | Not supported
| P3.8XLarge |   4  | 1024  | 21.6 sec | Not supported

##### GPU Utilization
###### Reporting average GPU % utilization
| Instance Type | GPUs  | Batch Size | Keras-MXNet (GPU Utilization %) | Keras-TensorFlow (GPU Utilization %)  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |   2  | 512  | 5 | Not supported
| P3.8XLarge |   4  | 128 | 6 | Not supported

##### GPU Memory Consumed
###### Reporting average memory consumed
| Instance Type | GPUs  | Batch Size | Keras-MXNet (GPU Memory (MB)) | Keras-TensorFlow (GPU Memory (MB))  |
|-----|-----|-----|-----|-----|
| P3.8XLarge |  2  | 512 | 972.1  | Not supported
| P3.8XLarge |  4  | 1024 | 969.9 | Not supported

### Note
Run the file as `python run_sparse_benchmark.py`, by default the benchmark runs for `training` with `10 epochs` and batch size of `512`

### References
MXNet supports sparse data in 2 NDArray formats - CSRNDArray and RowSparseNDArray which are defined in `mxnet.ndarray.sparse` package
For further details on MXNet Sparse NDArray API check [documentation related to MXNet Sparse](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html)

Keras Input layer supports sparse data by setting a boolean placeholder value - check document for [Keras Input layer](https://keras.io/layers/core/#input)