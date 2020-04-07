import pytest

import tempfile
from keras import backend as K
from keras.layers import Dense, BatchNormalization
from keras.models import load_model, Sequential
from keras.backend import tensorflow_backend as KTF
import warnings

pytestmark = pytest.mark.skipif(K.backend() != "mxnet",
        reason="Testing MXNet context supports only for MXNet backend")


class TestMXNetTfModel(object):
    def test_batchnorm_layer_reload(self):
        # Save a tf backend keras h5 model
        tf_model = KTF.tf.keras.models.Sequential([
            KTF.tf.keras.layers.Dense(10, kernel_initializer="zeros"),
            KTF.tf.keras.layers.BatchNormalization(),
        ])
        tf_model.build(input_shape=(1, 10))
        _, fname = tempfile.mkstemp(".h5")
        tf_model.save(fname)

        # Load from MXNet backend keras
        try:
            mx_model = load_model(fname, compile=False)
        except TypeError:
            warnings.warn("Could not reload from tensorflow backend saved model.")
            assert False

        # Retest with mxnet backend keras save + load
        mx_model_2 = Sequential([
            Dense(10, kernel_initializer="zeros"),
            BatchNormalization(),
        ])
        mx_model_2.build(input_shape=(1, 10))
        _, fname = tempfile.mkstemp(".h5")
        mx_model_2.save(fname)

        try:
            mx_model_3 = load_model(fname, compile=False)
        except TypeError:
            warnings.warn("Could not reload from MXNet backend saved model.")
            assert False


if __name__ == "__main__":
    pytest.main([__file__])
