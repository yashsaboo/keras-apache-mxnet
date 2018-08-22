"""
Linear Regression model with sparse synthetic data for MXNet
"""

import time
import mxnet as mx


def run_benchmark(train_data, train_label, eval_data, eval_label, batch_size, epochs):
    train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, shuffle=True,
                                   label_name='lin_reg_label', last_batch_handle='discard')
    eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size=batch_size, shuffle=False,
                                  label_name='lin_reg_label', last_batch_handle='discard')

    X = mx.sym.Variable('data', stype='csr')
    Y = mx.symbol.Variable('lin_reg_label', stype='csr')
    fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=10)
    lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

    model = mx.mod.Module(
        symbol=lro,
        data_names=['data'],
        label_names=['lin_reg_label']
    )

    start = time.time()

    model.fit(train_iter, eval_iter,
              optimizer_params={'learning_rate': 0.1, 'momentum': 0.9},
              num_epoch=epochs,
              eval_metric='mse',
              batch_end_callback=mx.callback.Speedometer(batch_size))

    print("MXNet Sparse Benchmark Results")
    print("Batch Size")
    print(batch_size)
    print('Total Time')
    print(time.time() - start)

    metric = mx.metric.MSE()
    mse = model.score(eval_iter, metric)
    print("Achieved {0:.6f} validation MSE".format(mse[0][1]))

    # Verify results on eval data
    print(model.score(eval_iter, metric))
