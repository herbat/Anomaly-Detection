import time
import pickle
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import *

epochs = 5
batch_size = 64
load = True
train = False
fname = "convAE.h5"


class DenseTranspose(Layer):

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 tied_to=None,  # Enter a layer as input to enforce weight-tying
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseTranspose, self).__init__(**kwargs)
        self.units = units
        # We add these two properties to save the tied weights
        self.tied_to = tied_to
        self.tied_weights = self.tied_to.weights
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        # We remove the weights and bias because we do not want them to be trainable

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        # Return the transpose layer mapping using the explicit weight matrices
        output = K.dot(inputs - self.tied_weights[1], K.transpose(self.tied_weights[0]))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def makeDenseModel():
    inputs = Input(shape=(1000,))
    e1 = Dense(units=512, activation='tanh')
    e2 = Dense(units=128, activation='tanh')
    e3 = Dense(units=32,  activation='tanh')

    x = e1(inputs)
    x = e2(x)
    x = e3(x)

    d1 = DenseTranspose(128, tied_to=e3)
    d2 = DenseTranspose(512, tied_to=e2)
    d3 = DenseTranspose(1000, tied_to=e1)

    x = d1(x)
    x = d2(x)
    x = d3(x)

    return Model(inputs, x)


def makeConvModel():

    inputs = Input(shape=(1000,))
    x = Reshape((1000,1,1), input_shape=(1000,))(inputs)
    x = Conv2D(16, (12,1), activation='tanh', strides=4, padding='valid')(x)
    x = Conv2DTranspose(16, (12,1), activation='tanh', strides=4, padding='valid')(x)
    x = Lambda(lambda x: x[:, :, 0])(x)
    x = Conv1D(1, 7, activation='tanh', strides=1, padding='same')(x)
    x = Flatten()(x)
    return Model(inputs, x)


# load or create model
# autoencoder = makeDenseModel()
# if load: autoencoder.load_weights(fname)
autoencoder = load_model(fname) if load else makeConvModel()
autoencoder.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])
eeg_train = np.asarray(pickle.load(open('trainingset.pkl', 'rb')))
start = time.time()


if train:
    eeg_train = np.asarray(pickle.load(open('trainingset.pkl', 'rb')))
    eeg_test = np.asarray(pickle.load(open('testingset.pkl', 'rb')))
    autoencoder.fit(eeg_train, eeg_train,
                    epochs=epochs,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(eeg_test, eeg_test))
else:
    eeg_anomaly = np.asarray(pickle.load(open('anomalyset.pkl', 'rb')))
    # eeg_anomaly = np.expand_dims(eeg_anomaly[0], axis=0)
    print(autoencoder.test_on_batch(eeg_anomaly, eeg_anomaly))


# TESTING

# data = K.variable(np.expand_dims(eeg_train[0], axis=0))
# print(np.shape(data))
# print(K.int_shape(autoencoder(data)))

stop = time.time()
print(stop - start)
if train: autoencoder.save(fname)
