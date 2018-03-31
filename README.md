# Anomaly Detection

This semester's Individual Study is going to be the basis of my degree thesis, and since my goal is to study artifical intelligence and neural network further down the line, I chose an interesting application of AI: anomaly detection. For this task, I start with autoencoders and then explore further. 

## First stage: framework testing

### General info 

The dataset I used was a randomly generated set of 5000 recordings, every one of which had 1000 datapoints. These are all sinusoids in the alpha range of EEG. 
With both PyTorch and Keras, I did 100 epochs with 64 for batch size and a `1e-4` learning rate. I used adam optimizer, tanh activation, and the same architecture of densely connected layers. 

### PyTorch

Last semester I gained some knowledge in deep learning with PyTorch, so I quickly wrote an autoencoder, to compare it with other frameworks. Since PyTorch has a special Dataloader library, it's not straightforward to use your own dataset, which isn't a good thing. 

When tested on my dataset, 100 epochs ran for **220 seconds**, which means **2.2s/epoch**.

### Keras

This library was completely new for me, but not more than half an hour was enough for me to get a working autoencoder. A little more reading, and my comparable setup was finished. The data loading is a breeze, the code is much clearer and easier to debug. The metrics are presented perfectly as well.

The same test, using tensorflow backend took **130 seconds**, which is a huge improvement **(69% better)**. I also ran 1000 epochs run, at the end of which the loss was `0.0291` (mean squared error). When tested with comlpetely different data, the loss was `0.103`. This model was ready for further testing. 

### Conclusion

After this test, I decided to keep using Keras as the main framework for this project. Meanwhile, I will learn tensorflow better, but since Keras is sufficient in almost all cases except serious research or industrial applications, it will not be necessary. 

## Anomaly detection 

The first test to do is to use an autoencoder and feed it "anomaly data", which can be anything from a completely different thing or the same sinusoidal waves with little modifications. For this experiment, I will change the frequency and amplitude substantially in anomaly recordings, and in other recordings, insert peaks similar to that of an epileptic brain. 

### First try: frequency and amplitude change

After the training was done, I ran a test on anomalous recordings. This test was not very successful, the normal test batch showed a constant `0.051` loss while the anomalies showed a varying loss about `0.081`. This means a **60% increase** in loss, but needs 500 recordings to evaluate properly. On single recordings, the loss was sometimes as low as `0.058`. 

