# **Traffic Sign Recognition** 

## Writeup

This report describes the development of a Neural Network for the purpose of identifying Traffic Signs from the 'German Traffic Sign' database. The original data is sourced from: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/sign_labels.png "Sign Labels"
[image2]: ./writeup_images/sign_types.png "Sign Types"
[image3]: ./writeup_images/hist_eq.png "Histogram Equalization"
[image4]: ./writeup_images/data_augment.png "Augmented Data"
[image5]: ./sign_images/sign11_small.png "Traffic Sign 1"
[image6]: ./sign_images/sign12_small.png "Traffic Sign 2"
[image7]: ./sign_images/sign14_small.png "Traffic Sign 3"
[image8]: ./sign_images/sign25_small.png "Traffic Sign 4"
[image9]: ./sign_images/sign33_small.png "Traffic Sign 5"

## Rubric Points

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

This file is the project deliverable writeup! The Jupyter notebook implementing this project is available at [project code](https://github.com/ajhsutton/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set that extracts one image for each class in the set.

![Visualization][image1]

The following image shows a selection of sign images along with their label.


![Sign Labels][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Greyscale images were found to be sufficient for this project. All images in the training set were for converted to greyscale, and quantized to int8. Histogram equalization was used to normalize each image. The figure below shows the output result of the greyscale conversion with (2nd row) and without (3rd row) histogram equalization.

![Greyscale and Equalization][image3]

Data augmentation was performed for the data set, where 5 images were generated for each greyscale image in the training set.

Augmentation used openCV function to perform data augmentation by adding:
 * Random angle between (-10°,10°),
 * Random scale between 90% - 110%,
 * Random rotation center (within the image bounds).

This functionality used openCV's 'getRotationMatrix2D' and 'warpAffine' functions.

Here is an example of a set of random image permutations that were used to augment the data set:

![Data Augmentation][image4]

The Augmented training data set has 173995 images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   					| 
| L1: Convolution 5x5	| 1x1 stride, same padding, 				 	|
| 		RELU			|												|
| 		Max pooling		| 2x2 stride,  outputs 14x14x24 				|
| L2: Convolution 5x5	| 1x1 stride, same padding, 				 	|
| 		RELU			|												|
| 		Max pooling		| 2x2 stride,  outputs 5x5x32 					|
| L3: Convolution 3x3	| 1x1 stride, same padding, 				 	|
| 		RELU			| 1x1 stride, outputs 3x3x64 					|
| Flatten				| output 1x576									|
| Fully connected w RELU| output 1x128 									|
| Fully connected		| output 1x43 									|
| Softmax				| Classifier Output (43 Classes					|
 
Variables were initialized using a truncated normal distribution with mean 0 and sigma = 0.2. 

#### 3. Describe how you trained your model. 

Training Hyperparameters:
* Optimizer: Adam
* Number of Epochs : 30
* Batch Size: 128
* Learning Rate: 0.0002

Dropout was implemented in all layers, using a 50% dropout probability when training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

An iterative approach to network architecture development was used. The initial architecture was selected to match the LeNet architecture. Model parameters were adjusted to achieve the final result of a 96

My final model results were:
* training set accuracy of 94.5%
* validation set accuracy of 96%
* test set accuracy of 93.8%

If an iterative approach was chosen. Multiple architecture variable were experimented with including:
* Variable itialization distirbution: convergence of the network training was found to be highly sensitive to specification of the variable standard deviation use to initialize the network. For example, increasing the standard deviation to 0.5 caused the network to fail to converge.
* Learning rate: the learning rate was initialized as 0.001, however this needed to be decreased by 1/10 to allow repeatible convergence. This parameter was also found to be sensitive to batch size.
* 1x1 Convolution: Experimentation with adding a 1x1 convolution layer failed to improve performance.
* 3 Convolutional layers: the 5x, 5x, 3x convolutional layer structure was found to converge quickly and gave good results.
* 2 Fully-Connected output layers: Experimentation showed that 2 fully connected output layers were required for convergence, and that the first FC layer needed significantly more entries than the output-FC layer (which has 43 outputs, one for each class).
* Dropout was utlized to manage overfitting. The close agreement between train, validation and test acuracy indicates an absence of overfitting in the final architecture. Early architectures were found to suffer from overfitting to the training set, where the training accuracy would plateau while the loss continued to decrease.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Sign 1][image5] ![Sign 2][image6] ![Sign 3][image7] 
![Sign 4][image8] ![Sign 5][image9]

These images are clear and should be realtively easy to classify due to the clear visibliity and excellent lighting conditions.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

The model was able to correctly predict all 5 new traffic sign images, giving an accuracy of 100%. This compares favorably to the accuracy on the test set, and likely results from the 'quality' of the images (ie. clearly visible with good lighting).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The notebook displays the predictions and associated probabilitys for the top-five most-likely signs. Due to the high prediction probability, the other class predictions have very low probaility.
