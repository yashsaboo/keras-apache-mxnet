# Inference on Amazon EIA with Keras-MXNet

## Table of Contents

1. [Objective](#objective)
2. [Prerequisites](#prerequisites)
3. [Step 1 - Download a Pre-trained Keras Model](#step-1---download-a-pre-trained-keras-model)
4. [Step 2 - Load the Keras Model in EIA context](#step-2---load-the-keras-model-in-eia-context)
5. [Step 3 - Run Inference](#step-3---run-inference)
6. [References](#references)

## Objective

In this tutorial, you will use [Keras](https://keras.io/), with [Apache MXNet](https://mxnet.incubator.apache.org/) 
backend, to load a [pre-trained VGG16 Keras model](https://keras.io/applications/#vgg16) and run inference on [Amazon Elastic Inference Accelerator (Amazon EIA)](https://aws.amazon.com/machine-learning/elastic-inference/).

## Prerequisites

1. Keras-MXNet
2. Apache MXNet with EIA support. See [Amazon Elastic Inference for MXNet setup guide](https://docs.aws.amazon.com/elastic-inference/latest/developerguide/ei-python.html)
3. Amazon EC2 instance with EIA. See [Amazon Elastic Inference setup guide](https://docs.aws.amazon.com/elastic-inference/latest/developerguide/setting-up-ei.html)

## Step 1 - Download a Pre-trained Keras Model

For this tutorial, let us download a [pre-trained, on ImageNet, VGG16 Keras Model](https://keras.io/applications/#vgg16) from Keras applications.

```python
from keras.applications.vgg16 import VGG16

# Load a ImageNet Pre-Trained VGG-16
model = VGG16(weights='imagenet', input_shape=(224,224,3))
model.save("imagenet_vgg16.h5")

```

## Step 2 - Load the Keras Model in EIA context

Next, we will load the downloaded pre-trained VGG-16 model in `EIA` context i.e., we let the MXNet backend know that we want to use EIA for predictions.

```python
from keras import backend as K
from keras.models import load_model

# Load the Model in EIA Context
with K.Context("eia"):
    model = load_model("imagenet_vgg16.h5")

```

## Step 3 - Run Inference on EIA

For this tutorial, to demo inference, let us download 2 sample images - Image of a Tusker and a Race car.

```bash
wget -O tusker.jpeg http://www.krugerpark.co.za/images/kt-19-big-tuskers-1.jpg

wget -O racecar.jpeg https://cdn.cnn.com/cnnnext/dam/assets/180130144240-formula-e-car-gen2-front-facing-large-169.jpg

```

Next, we will use the loaded Keras model to run the inference on the 2 sample images. Since, we have loaded the model in `EIA` context, MXNet backend will be using EIA to run inference.

```python
import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

# Prepare Inputs

# Image 1 - A racecar
racecar = image.load_img("racecar.jpeg", target_size=(224, 224))
racecar = image.img_to_array(racecar)
racecar = np.expand_dims(racecar, axis=0)
racecar = preprocess_input(racecar)

# Image 2 - A Tusker
tusker = image.load_img("tusker.jpeg", target_size=(224, 224))
tusker = image.img_to_array(tusker)
tusker = np.expand_dims(tusker, axis=0)
tusker = preprocess_input(tusker)

preds = model.predict(racecar)
print('Prediction:', decode_predictions(preds, top=3)[0])
```

**Expected Output:**

```
Prediction: [('n04037443', 'racer', 0.88579184), ('n04285008', 'sports_car', 0.104770236), ('n02974003', 'car_wheel', 0.008486765)]
```

```python
preds = model.predict(tusker)
print('Prediction:', decode_predictions(preds, top=3)[0])
```

**Expected Output:**

```
Predicted: [('n01871265', 'tusker', 0.63174385), ('n02504458', 'African_elephant', 0.35999662), ('n02504013', 'Indian_elephant', 0.008235933)]
```

You can use `Batch Prediction` as shown below.

```python
batch_input = np.concatenate((racecar, tusker), axis=0)
batch_preds = model.predict_on_batch(batch_input)
for pred in decode_predictions(batch_preds, top=3):
    print('Prediction:', pred)
```

**Expected Output:**

```
Prediction: [('n04037443', 'racer', 0.88579184), ('n04285008', 'sports_car', 0.104770236), ('n02974003', 'car_wheel', 0.008486765)]
Prediction: [('n01871265', 'tusker', 0.63174385), ('n02504458', 'African_elephant', 0.35999662), ('n02504013', 'Indian_elephant', 0.008235933)]
```

**NOTE**
1. Amazon EIA supports `Inference` only. You cannot use a model loaded in `EIA` context for Training.

## References
1. [Keras-MXNet Installation Guide](installation.md)
2. [Apache MXNet](http://mxnet.incubator.apache.org/)
3. [Keras Applications](https://keras.io/applications/)
4. [Amazon Elastic Inference](https://docs.aws.amazon.com/elastic-inference/latest/developerguide/what-is-ei.html)
5. [Amazon Elastic Inference for MXNet setup guide](https://docs.aws.amazon.com/elastic-inference/latest/developerguide/ei-python.html)
6. [Amazon Elastic Inference setup guide](https://docs.aws.amazon.com/elastic-inference/latest/developerguide/setting-up-ei.html)
