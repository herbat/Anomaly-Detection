import time
import pickle
import random
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model

epochs = 300
batch_size = 64
load = True
train = False

def makeModel():
    model = Sequential([
        #encoder
        Dense(units=512, activation='tanh', input_dim=1000),
        Dense(units=128, activation='tanh'),
        Dense(units=64, activation='tanh'),
        #decoder
        Dense(units=128, activation='tanh'),
        Dense(units=512, activation='tanh'),
        Dense(units=1000, activation='tanh')
    ])
    return model


# load or create model
autoencoder = load_model('autoenc1.h5') if load else makeModel()
autoencoder.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])

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


stop = time.time()
# print(stop - start)
autoencoder.save('autoenc1.h5')