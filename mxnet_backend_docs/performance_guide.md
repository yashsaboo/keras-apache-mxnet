# Performance Guide for MXNet Backend

In this document we will discuss some tips to improve your training/inference speed for MXNet Backend

## Image Data Format for CNN

If you have seen this Performance warning, we will show you how to solve it here.

```
UserWarning: MXNet Backend performs best with `channels_first` format. Using `channels_last` will significantly reduce performance due to the Transpose operations. For performance improvement, please use this API`keras.utils.to_channels_first(x_input)`to transform `channels_last` data to `channels_first` format and also please change the `image_data_format` in `keras.json` to `channels_first`.Note: `x_input` is a Numpy tensor or a list of Numpy tensor
```

By default, we have set mxnet backend to use `channels_last` data format same as TensorFlow backend. 
Your existing Keras code should work out of the box for MXNet backend. However, in order to boost your performance, 
channels first data format is the optimal format to use when training on NVIDIA GPUs using cuDNN. Please refer also to [TensorFlow performance guid](https://www.tensorflow.org/performance/performance_guide#data_formats) for detailed explanation.
Therefore, some changes need to be made, and please follow these steps to avoid breaking your code.
#### 1. Change Json Config File

First thing is to change the `keras.json` file in your home directory under `~/.keras` , set `image_data_format` to `channels_first`. This is also how you can switch between backends, refer to [Keras Json Details](https://keras.io/backend/#kerasjson-details)
```
"image_data_format": "channels_first", 
"backend": "mxnet"
```

### 2. Make sure data is channels first format

You need to make sure the data shape you provide to the model are consistent with the `keras.json` configuration. Raw images are in channels last format, so you need to do one time transpose to convert it to channels last before training.
There are two ways to do that:

2.1. Pass your data to [`to_channels_first`](https://github.com/awslabs/keras-apache-mxnet/blob/master/keras/utils/np_utils.py#L55) API for transposing.

2.2. If you use Keras built in dataset, sometimes it converts for you automatically based on your json configuration (e.g. 
[cifar10 dataset](https://github.com/awslabs/keras-apache-mxnet/blob/master/keras/datasets/cifar10.py#L40)), sometimes 
it does not (e.g. [mnist dataset](https://github.com/awslabs/keras-apache-mxnet/blob/master/keras/datasets/mnist.py)).
So always double check your data format before training or you might get a shape miss match error.

### 3. Change `input_shape` argument when constructing model
Remember to pass input shape aligned with your data format when constructing the model. For example, using Conv2D on color images with size
256x256, you will pass:

For channels last format:
```model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256,256,3)))```
For channels first format:
```model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, 256,256)))```

We recommend to pass input_shape base on the shape field of your input tensor so it's always alined with your input data format. For example:
```model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))```

## Try Different MXNet packages

Currently we have different mxnet packages optimized for different platforms. You can choose the right one for your case to serve as backend.

Instead of `pip install mxnet` you can try:
 
`pip install mxnet-mkl` for Intel CPUs:
 
`pip install mxnet-cu90 ` for NVIDIA GPUS with cuda 9
 
`pip install mxnet-cu90-mkl` for both

In addition, there are more configurations you can do if you built mxnet from source, refer to 
[nstallation guide](https://mxnet.incubator.apache.org/install/index.html) and 
[MXNet Performance Guide](https://mxnet.incubator.apache.org/faq/perf.html)


## Profiler
MXNet has a built-in profiler that gives detailed information about execution time at the symbol level. 
This feature complements general profiling tools like nvprof and gprof by summarizing at the operator level, 
instead of a function, kernel, or instruction level. Follow [MXNet Performance Guide](https://mxnet.incubator.apache.org/faq/perf.html)
on how to enable profiler.

## Reference:
1. [MXNet Performance Guide](https://mxnet.incubator.apache.org/faq/perf.html)
2. [TensorFlow Performance Guide](https://www.tensorflow.org/performance/performance_guide#data_formats)

