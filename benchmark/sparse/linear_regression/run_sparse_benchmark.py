"""
Prepare data for running benchmark on sparse linear regression model
"""
from __future__ import print_function

import argparse
import time

import keras_sparse_model
import mxnet as mx
import mxnet_sparse_model
from scipy import sparse

from keras import backend as K
from keras.utils.data_utils import prepare_sliced_sparse_data


def invoke_benchmark(batch_size, epochs):
    feature_dimension = 1000
    train_data = mx.test_utils.rand_ndarray((100000, feature_dimension), 'csr', 0.01)
    target_weight = mx.nd.arange(1, feature_dimension + 1).reshape((feature_dimension, 1))
    train_label = mx.nd.dot(train_data, target_weight)
    eval_data = train_data
    eval_label = mx.nd.dot(eval_data, target_weight)

    train_data = prepare_sliced_sparse_data(train_data, batch_size)
    train_label = prepare_sliced_sparse_data(train_label, batch_size)
    eval_data = prepare_sliced_sparse_data(eval_data, batch_size)
    eval_label = prepare_sliced_sparse_data(eval_label, batch_size)

    print("Running Keras benchmark script on sparse data")
    print("Using Backend: ", K.backend())
    keras_sparse_model.run_benchmark(train_data=sparse.csr_matrix(train_data.asnumpy()),
                                     train_label=sparse.csr_matrix(train_label.asnumpy()),
                                     eval_data=sparse.csr_matrix(eval_data.asnumpy()),
                                     eval_label=sparse.csr_matrix(eval_label.asnumpy()),
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     start=time.time())

    print("Running MXNet benchmark script on sparse data")
    mxnet_sparse_model.run_benchmark(train_data=train_data,
                                     train_label=train_label,
                                     eval_data=eval_data,
                                     eval_label=eval_label,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     start=time.time())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=128,
                        help="Batch of data to be processed for training")
    parser.add_argument("--epochs", default=25,
                        help="Number of epochs to train the model on. Set epochs>=1000 for the best results")
    args = parser.parse_args()

    invoke_benchmark(int(args.batch), int(args.epochs))
