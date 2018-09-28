# Prepare sparse data for training for Keras-MXNet

## Table of Contents

1. [Objective](#objective)
2. [Slice sparse data](#slice-sparse-mxnet-data)
3. [References](#references)

## Objective

In this tutorial, we show how to train a model with sparse data in Keras-MXNet.

As MXNet sparse arrays do not support reshape operation, it is essential to be able to slice the data for training the 
model.
You can use the following API to slice sparse data in Keras-MXNet:
1. Using `keras.data_utils.prepare_sliced_sparse_data` API.

## Slice sparse data
```python
# ... Assuming you are using sparse dataset as follows:
row_ind = np.array([0, 1, 1, 3, 4])
col_ind = np.array([0, 2, 4, 3, 4])
data = np.array([1, 2, 3, 4, 5], dtype=float)
x_train = scipy.sparse.coo_matrix((data, (row_ind, col_ind)))

model.fit(x_train, y_train,
          batch_size=3,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
```

In the above code, the train dataset is of shape `(5,5)`

For every epoch in training, a batch size of `3` is used and the data in the last batch is reshaped if its size is 
smaller than or equal to the batch size.

For the above model, the first epoch will have batch of size `3`. 
Second will have a batch of size `2` and this will require invocation of `reshape`

As MXNet sparse NDArray does not support reshape, we need to slice excess dimensions from the train data as follows:
```python
x_train = data_utils.prepare_sliced_sparse_data(x_train, batch_size)
```

This will slice the training data into shape `(3,5)`
Now we can continue training the data without any issue

To summarize, all you have to do is to call `keras.data_utils.prepare_sliced_sparse_data()` and pass the 
sparse training data


## References
1. [MXNet Sparse NDArray Implementation](https://mxnet.incubator.apache.org/_modules/mxnet/ndarray/sparse.html)
2. [Refer sparse linear regression model for further details](https://github.com/awslabs/keras-apache-mxnet/tree/master/benchmark/sparse/linear_regression)