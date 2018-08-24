"""
Prepare data for running benchmark on sparse linear regression model
"""

import argparse

import keras_tf_model
import mxnet as mx
import mxnet_sparse_model
from scipy import sparse


def invoke_benchmark(batch_size, epochs):
    train_data = mx.test_utils.rand_ndarray(shape=(1000, 10), stype='csr')
    train_label = 3 * train_data
    eval_data = mx.test_utils.rand_ndarray(shape=(1000, 10), stype='csr')
    eval_label = 3 * eval_data

    print("Running Keras-TF sparse benchmark script")
    keras_tf_model.run_benchmark(train_data=sparse.csr_matrix(train_data.asnumpy()),
                                 train_label=sparse.csr_matrix(train_label.asnumpy()),
                                 eval_data=sparse.csr_matrix(eval_data.asnumpy()),
                                 eval_label=sparse.csr_matrix(eval_label.asnumpy()),
                                 batch_size=int(batch_size),
                                 epochs=int(epochs))

    print("Running MXNet sparse benchmark script")
    mxnet_sparse_model.run_benchmark(train_data=train_data, train_label=train_label, eval_data=eval_data,
                                     eval_label=eval_label, batch_size=int(batch_size), epochs=int(epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=128,
                        help='Batch of data to be processed for training')
    parser.add_argument('--epochs', default=1000,
                        help='Number of epochs to train the model on. Set epochs>=1000 for the best results')
    args = parser.parse_args()

    invoke_benchmark(args.batch, args.epochs)
