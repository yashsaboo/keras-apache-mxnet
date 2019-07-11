import pytest
import keras
import numpy as np
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions


def has_eia():
    if K.backend() != 'mxnet':
        return False

    import mxnet as mx
    try:
        # try to create eia context
        mx.eia()
    except:
        return False

    return True


@pytest.mark.skipif(K.backend() != 'mxnet' or not has_eia(),
                    reason='Inference with AWS EIA is currently supported '
                           'with MXNet backend only. We need to have EIA '
                           'to run Keras predictions on EIA tests.')
def test_prediction_with_eia():
    import mxnet as mx

    # 1. Download and save ImageNet Pre-Trained VGG-16
    model = VGG16(weights='imagenet', input_shape=(224, 224, 3))
    model.save("imagenet_vgg16.h5")

    # 2. Load the Model in EIA Context
    with K.Context("eia"):
        model = keras.models.load_model("imagenet_vgg16.h5")

    # Verify Model is loaded in EIA context
    assert model._context
    assert model._context[0] == mx.eia()

    # 3. Prepare inputs for prediction
    dummy_image1 = np.random.randint(low=0, high=255, size=(224, 224, 3))
    dummy_image1 = np.expand_dims(dummy_image1, axis=0)
    dummy_image1 = preprocess_input(dummy_image1)
    preds = model.predict(dummy_image1)
    assert len(decode_predictions(preds, top=3)[0]) == 3

    # 4. Test batch prediction
    dummy_image2 = np.random.randint(low=0, high=255, size=(224, 224, 3))
    dummy_image2 = np.expand_dims(dummy_image2, axis=0)
    dummy_image2 = preprocess_input(dummy_image2)

    batch_input = np.concatenate((dummy_image1, dummy_image2), axis=0)
    batch_preds = model.predict_on_batch(batch_input)
    assert len(batch_preds) == 2
    for pred in decode_predictions(batch_preds, top=3):
        assert len(pred[0]) == 3


if __name__ == '__main__':
    pytest.main([__file__])
