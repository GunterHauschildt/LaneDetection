# Lane Detection
Prototype for a lane detection algorithm.

## Results <a name="results"></a>
https://github.com/user-attachments/assets/41e098be-34e8-4407-a402-02519ff6cf86

## Discussion <a name="discussion">
An image segmentation neural network is used to predict lane markers.  
Classical image processing and machine learning tecniques are used to cleanup those lane markers, ultimately giving a left-right sorted list that describes these lane markers using simply y=mx+b.

### Data
Kaggle's TuSimple dataset is used.  
https://www.kaggle.com/datasets/manideep1108/tusimple  
The data is annoted such that up to 4 lane markers are given: up to 2 to left and up to 2 to the right.  

### Image Segmentation Neural Network <a name="image_segmentations">  
The image segmentation neural network uses (pretrained) ResNet50 as the encoder and a pix2pix GAN as the decoder.  
It is similar to https://www.tensorflow.org/tutorials/images/segmentation  
100 epochs were used but the best validation accuraccy achieved after 20.  

### Post Processing  
The mask returned from the image segmentation NN must be converted to order lane marker descriptors.  
To describe a lane marker's line, using y=mx+b will be used and thus the lane marker needs only m, b.  
Generally, the NN gives noisy disconnected segments for each lane marker.  

#### Morphology Open, Close, Smoothing, Thinning
The mask as return from the image segmentation NN is first cleaned up by opening, closing, and then smoothing the resulting contours.  
The result is 'curvy-but-close' 2D areas. The 'curvy-but-close' 2D areas are converted to 'curvy-but-close' 1D contours.  
If small, these 1D contours are discarded, otherwise they are assumed to be a straight line and their extremes used to define line segments.  


#### Calibration  
Line segment descriptors, m, b from y=mx+b, when and only when the car is centered into its are pushed into KMeans.
Kmeans output is then sorted from left to right (increasing m) so we can sort the lane segments in their approoriate lane,
Note that 'x' is vertical and 'y' is horizontal. This eliminates the possiblity of infinite slope -
you can't drive perpendicular to the lane.

#### Kalman Filtering  
Each lane marker is kalman filtered.  

#### Lane Deperature  
Lane departure is now a simple matter of observing where the lane marker lies. In this example, only 'm' is required to detect departure.  
