# Use sparse data with Keras-MXNet
## Table of Contents
1. [Objective](#objective)
2. [Sparse support overview](#sparse-support-overview)
3. [Slice sparse data](#slice-sparse-mxnet-data)
4. [References](#references)
## Objective
In this tutorial, we give an overview of sparse support in Keras-MXNet and how to prepare sparse data for building 
end-to-end model.
## Sparse support overview
### Sparse tensors
MXNet supports sparse data in 2 formats - [CSRNDArray](https://mxnet.incubator.apache.org/tutorials/sparse/csr.html) & 
[RowSparseNDArray](https://mxnet.incubator.apache.org/tutorials/sparse/row_sparse.html). 

Use placeholder `sparse=True`:
```python
from keras import backend as K
sparse_tensor = K.placeholder((2, 2), sparse=True)
```
Use Input layer for configuring sparse input in [functional model](https://keras.io/getting-started/functional-api-guide/):
```python
from keras.layers import Input
sparse_tensor = Input(shape=(2, 2), sparse=True)
```
With [sequential model](https://keras.io/models/sequential/) sparse tensor can be used as:
```python
from keras.models import Sequential
from keras.layers import InputLayer
model = Sequential()
model.add(InputLayer(input_shape=(2, 2), sparse=True))
```

Both of the above code snippets will create a sparse tensor that will bind the data in `csr` format

### Sparse weight
Keras-MXNet also supports adding sparse weights for training and they can be initialized as follows:
```python
from keras.layers import Input, Dense
input = Input(shape=(3,), sparse=True)
output = Dense(units=1, activation='linear', sparse_weight=True)(input)
```
Using sparse weight is useful in models where we need to invoke 
[dot operation](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html#mxnet.ndarray.sparse.dot) for training.
In this code, setting sparse_weight to `True` will create a `row_sparse` tensor that will be used internally.
### Sparse operators
We have added sparse support for the following operators with MXNet backend:
* sum
* mean
* dot
* concat
* Embedding
For using Embedding layer with sparse data, we need to set flag `sparse_grad` to True
```python
embedding=Embedding(max_features, 128, input_length=10, sparse_grad=True)
```
Please see release notes for v2.2.4.1 for further details
## Slice sparse data
As MXNet sparse arrays do not support reshape operation, it is essential to be able to slice the data for training the 
model.
You can use the following API to slice sparse data in Keras-MXNet:
* `keras.data_utils.prepare_sliced_sparse_data` API.
```python
# ... Assuming you are using sparse dataset as follows:
from scipy import sparse
x_train = sparse.coo_matrix((1000, 1000))
model.fit(x_train, y_train,
          batch_size=32,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
```
In the above code, the train dataset is of shape `(1000,1000)`
For every epoch in training, a batch size of `32` is used and the data in the last batch is reshaped if its size is 
smaller than or equal to the batch size.

For the above model, all the epochs will have batch size of `32`. This will cover `992` samples out of `1000` total 
as `992` is the highest divisible value by `32` in a sample of `1000 `
The last batch will have a batch of size `8` and this will require invocation of `reshape` to reshape the input matrix 
into `(8,1000)` dimension.
As MXNet sparse NDArray does not support reshape, we need to slice excess dimensions from the train data as follows:
```python
x_train = data_utils.prepare_sliced_sparse_data(x_train, batch_size)
```
This will slice the training data into shape `(992,1000)`
Now we can continue training the data without any issue
To summarize, all you have to do is to call `keras.data_utils.prepare_sliced_sparse_data()` and pass the 
sparse training data
## References
1. [MXNet Sparse NDArray Implementation](https://mxnet.incubator.apache.org/_modules/mxnet/ndarray/sparse.html)
2. [Refer sparse linear regression model for further details](https://github.com/awslabs/keras-apache-mxnet/tree/master/benchmark/sparse/linear_regression)