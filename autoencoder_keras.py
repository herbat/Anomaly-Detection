import time
import pickle
import random
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model

epochs = 1000
batch_size = 64
load = False

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


#load or create model
autoencoder = load_model('autoenc1.h5') if load else makeModel()

eeg_data = np.asarray(pickle.load(open('trainingset.pkl', 'rb')))

autoencoder.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])
start = time.time()
autoencoder.fit(eeg_data, eeg_data, epochs=epochs, batch_size=64, shuffle=True)
stop = time.time()
print(stop - start)
autoencoder.save('autoenc1.h5')