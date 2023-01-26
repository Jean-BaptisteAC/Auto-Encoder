# Auto-Encoder
This project is about the construction of a Variable Auto Encoder (VAE) using the basic MNIST-Fashion data set. 
The goal of an Auto Encoder is to encode an image within a very low-dimentional space in order to really grasp the specifications of the data set features.

Autoencoders can be used for anomaly classification, dimensionnality reduction, preprocessing, and for merging different images. 


## Architecture of the Model: 

The following figure describes the model that we constructed for our Variable Auto Encoder (VAE). 

![Turn](Ressources/File.drawio.png)

**The written sizes are corresponding to the output format of data after each layer. The sizes underneath Convolution and Transpose Convolution Layers are the kernel sizes**

The encoder is mapped on the left side while the decoder is on the right side of the image. In between, the intermediate Z-vector is of size 2. We decided for this project that a dimension of 2 will be enough to capture all the subtilties of the dataset: we indeed selected the Fashion-MNIST data set for its simple features.


For creating data points, we used the sampling method from the normal distribution where the first coordinate of the Z vector is the mean and the second coordinate the variance. We also used the Kl_Loss for the encoder and the binary cross-entropy for the decoder that we added together in order to compute the total loss of our VAE.


## Results: 

The following figure represents the Z-vector subspace after the model learned the data of the train set. 

![Turn](Ressources/plot.png)

What we can see here, is that the transitions between different classes of Fashion objects are pretty smooth, which is the purpose of the VAE because want to be able to interpolate images. However, we can spot several “holes” in the subspace, and some of the regions are overlapping. Overlapping regions means that we might have trouble to differentiate between separate picture classes. And holes in the Z-vector subspace directly result in pictures that we can’t recognize because the decoder is not able to construct a proper image out of it.













```
Code here
```

![Turn](Ressources/plot.png)


- List A
- List B
- List C


:red_car: **AI**
