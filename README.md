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

After the training was done, I ran a test on anomalous recordings. This test was very successful, the normal test batch showed a constant `0.051` loss while the anomalies showed a varying loss about `0.381`. 
In conclusion, the current model can differentiate quite easily between simulated alpha and beta brainwaves.

### Tied weights vs. free weights

Next, I tried tying the weights of the layers as is a convention in autoencoders. This made the model symmetric: the weights of the encoding layers were shared by the decoding layers, which is supposed to make learning faster. My experiments showed that the model learned the with the same speed(relative to epochs), and the computing time wasn't always faster. This might be due to the usage of custom layers for the weight mirroring.

### Convolutional architecture

To prepare for more complex datasets, I engineered a model only using convolutional layers. I tried a lot of things, including tied weights, 3 layer deep encoder/decoder, maxpooling and upsampling, but all failed miserably. This led me to keeping it simple, and trying a six-layer convolution only model. While this model wasn't a very good performer, I realized that by adding fewer layers, I can make the learning easier and faster. My final version is a two-layer autoencoder, which is **very** bad at data compression(it actually inflates the data), but is perfectly useful for anomaly detection. The testing dataset had an average of `0.0084` loss, while the anomaly dataset had an average of `0.061` loss, which is a sevenfold increase compared to the normal samples. 

After a bit of tuning, I arrived at the following architecture:
- 1D convolution, kernel:12, stride:3, channels:16
- 1D transpose convolution with the same parameters
- 1D convolution to transform back to the original data shape

This model had an average testing loss of `0.0088`, and anomaly loss of `0.087`, differing by an order of magnitude. Although this could surely be improved, I decided to take on other datasets, and tweak the model according to my findings.

## New data: Trajectories

As the network got quite good anomaly to normal loss ratios on random generated EEG data, it was time to try it on other datasets. Trajectories were chosen because it has real world applications. 

### Recording trajectories from video data

To gather realistic data, I decided to run an object detection network on the [UCSD Anomaly Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm), which consists of 98 videos, all 200 frames in length. For the object detection, I chose [YOLO](https://pjreddie.com/darknet/yolo/), a complex convolutional neural network, which was pretrained for object detection. Since YOLO is written in C, and built on DarkNet, a neural network framework - also written in C -, I used a DarkNet API for python called [LightNet](https://github.com/explosion/lightnet). I implemented a tracking algorithm, which is not very sophisticated, but usable. Unfortunately, these pictures are `373 × 248` in resolution, and this caused the network to be somewhat unreliable. Since my tracking algorithm wasn't refined for such scenarios, it failed in reliably recording the trajectories properly (which you can see in the following image). This led me to another approach. ![fail](https://github.com/herbat/Anomaly-Detection/blob/master/failure.png) 

### Randomly generated trajectories

Since most human trajectories can easily be modelled with a vector and some "noise" - a little variation in direction -, the model is quite simple, and straightforward to implement. The resulting training dataset has 10 000 random generated records, with variable length trajectories and no anomalies. 

### Recurrent layer before convnet autoencoder

To handle the variable-length data, I built a recurrent layer on top of the autoencoder showed before. This layer consists of 128 GRU cells, and shares the training loss function of the autoencoder. 

When I first tried training it, it didn't pass the anomaly detection test, as it had very small differences between the loss on anomalous records and normal ones. 






