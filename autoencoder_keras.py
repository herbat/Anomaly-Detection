import time
import pickle
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import load_model, Model

epochs = 1
batch_size = 64
load = True
train = False
model_fname = 'rnnAE.h5'
train_fname = 'trajectory_data/training_rand1.p'
test_fname  = 'trajectory_data/testing_rand1.p'
anom_fname  = 'trajectory_data/anomaly_rand1.p'


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


def intermediate_loss(x):
    return K.mean(K.square(x[0] - x[1] + 1000), axis=-1)


def generator(from_list_x):

    total_size = len(from_list_x)

    while True:

        for i in range(0,total_size):
            yield np.expand_dims(np.array(from_list_x[i]), axis=0), np.array([0])


def make_dense_model():
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


def make_conv_model():

    inputs = Input(shape=(1000,))
    x = Reshape((1000,1,1), input_shape=(1000,))(inputs)
    x = Conv2D(16, (12,1), activation='tanh', strides=4, padding='valid')(x)
    x = Conv2DTranspose(16, (12,1), activation='tanh', strides=4, padding='valid')(x)
    x = Lambda(lambda x: x[:, :, 0])(x)
    x = Conv1D(1, 7, activation='tanh', strides=1, padding='same')(x)
    x = Flatten()(x)
    return Model(inputs, x)


def make_rnn_model():

    inputs = Input(shape=(None, 2))
    gru = GRU(units=128)(inputs)
    x = Reshape((128, 1, 1), input_shape=(128,))(gru)
    x = Conv2D(16, (12, 1), activation='tanh', strides=4, padding='valid')(x)
    x = Conv2DTranspose(16, (12, 1), activation='tanh', strides=4, padding='valid')(x)
    x = Lambda(lambda x: x[:, :, 0])(x)
    x = Conv1D(1, 7, activation='tanh', strides=1, padding='same')(x)
    x = Flatten()(x)
    x = Lambda(intermediate_loss, output_shape=(1, ), name='in_loss')([gru, x])
    return Model(inputs, x)


def plot_trajectory(_c):
    for t in _c:
        x = []
        y = []
        for i, j in t:
            x.append(i)
            y.append(j)
        plt.plot(x, y)

    plt.axis([0, 200, 0, 200])
    plt.show()


# load or create model
# autoencoder = makeDenseModel()
# if load: autoencoder.load_weights(fname)
autoencoder = load_model(model_fname, custom_objects={'<lambda>': lambda y_true, y_pred: y_pred}) if load else make_rnn_model()
autoencoder.compile(loss={'in_loss': lambda y_true, y_pred: y_pred}, optimizer='adadelta', metrics=['accuracy'])
start = time.time()

if train:
    training = pickle.load(open(test_fname, 'rb'))
    autoencoder.fit_generator(generator(training),
                              steps_per_epoch = len(training),
                              epochs=epochs,
                              shuffle=True)
else:
    anomaly = pickle.load(open(anom_fname, 'rb'))[np.random.randint(500)]
    plot_trajectory([anomaly])
    anomaly = np.expand_dims(np.array(anomaly), axis=0)
    print(autoencoder.test_on_batch(anomaly, [0]))


# TESTING

# data = np.asarray(pickle.load(open(test_fname, 'rb')))
# data = K.variable(np.expand_dims(data[0], axis=0))
# print(np.shape(data))
# print(K.eval(autoencoder(data)))

stop = time.time()
print(stop - start)
if train: autoencoder.save(model_fname)