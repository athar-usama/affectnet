# affectnet
Facial Expression Recognition (FER) and computing Valence &amp; Arousal through Transfer Learning on AffectNet dataset. Includes the entire source code for data preprocessing, model training, analysis, and visualization.

## Contents
1. [ Abstract ](#abs)
2. [ Methodology Diagram ](#m_dig)
3. [ Introduction ](#intro)
4. [ Setup Instructions ](#setup)
5. [ Usage Instructions ](#usage)
6. [ Quantitative Results ](#quant_res)
7. [ Visual Results ](#vis_res)
8. [ Author ](#auth)

<a name="abs"></a>
## Abstract
An image classification approach using Bag of Visual Words with SIFT features and two classifiers, Support Vector Machine (SVM) and Random Forest. The purpose of this work is to compare the performance of both classifiers in classifying two datasets; objects_dataset & flowers_dataset. Experimental results demonstrate that both classifiers have decent accuracy in classifying images. The results also indicate that the performance of the classifiers is highly dependent on the choice of parameters as well as the feature extractor. Overall, the proposed approach shows promising results and can be used as a basis for more complex image classification tasks.

<a name="m_dig"></a>
## Methodology Diagram
![methodology_diagram_2](https://user-images.githubusercontent.com/41828100/224599760-55d05b82-441c-44e5-9aa4-03dfcc501a04.png)

<a name="intro"></a>
## Introduction
Image classification is a fundamental problem in computer vision that involves categorizing images into different classes or categories. The task of image classification has numerous applications in various fields such as medical diagnosis, object recognition, face recognition, and autonomous vehicles. In recent years, many approaches have been proposed for image classification, including deep learning-based approaches such as Convolutional Neural Networks (CNNs), which have shown remarkable performance in various tasks. However, these approaches require large amounts of data and computational resources to train and optimize the models.

In this work, I proposed an image classification approach that uses Bag of Visual Words with SIFT features and two classifiers, Support Vector Machine (SVM) and Random Forest, to classify two datasets. The Bag of Visual Words (BoVW) is a popular feature extraction and representation method used in computer vision for image classification. BoVW divides an image into small regions and extracts relevant features from each region. The extracted features are then quantized into a predefined vocabulary of visual words, which are used to represent the image as a histogram. The BoVW method has been widely used in various image classification tasks, including object recognition and scene recognition.

SIFT (Scale-Invariant Feature Transform) is a feature detector and descriptor commonly used in computer vision for extracting robust and invariant features from images.  SIFT algorithm extracts key points and descriptors from an image, which can be used for matching, object recognition, and image classification. SIFT is known for its robustness to changes in scale, rotation, and illumination, making it suitable for various visual classification tasks.

I have also compared the performance of SVM and Random Forest classifiers in the given datasets using Bag of Visual Words. SVM is a popular machine learning that constructs a hyperplane separating the data into different classes by maximizing the margin between the hyperplane and the nearest data points of each class. Random Forest is an ensemble learning method which constructs multiple decision trees and combines their predictions to classify the data. Random Forest is known for its high accuracy and robustness to noise and outliers.

<a name="setup"></a>
## Setup Instructions
These are instructions for setting up the environment for running this code.

The code performs image classification using Bag of Visual Words and SIFT features. It uses two datasets: objects dataset and flowers dataset. The classifiers used are Support Vector Machine (SVM) and Random Forest Classifier.

### Requirements:
To run this code, you will need:

1- Python</br>
2- OpenCV</br>
3- scikit-learn

There are 2 methods for running this code. Let us take a brief look at each one of them.

### Method 1 (On Cloud):

Download the Python notebook and open it up inside Google Colaboratory. The datasets will be automatically mound via Google Drive.

Just run the notebook and wait for the classifiers to train themselves. The objects dataset section takes around 5 minutes whereas the flowers dataset section takes around 50 minutes for training. Trained classifiers are also saved and dumped in a file with JobLib. You can use them for prospective image classification tasks on similar datasets.

### Method 2 (On Device):

Clone this repository to your local machine with the following command:</br>
<pre>git clone https://github.com/athar-usama/bovw.git</pre>

Install Python and Jupyter Notebook. You can download them from the official websites:
#### Python: https://www.python.org/downloads/
#### Jupyter Notebook: https://jupyter.org/install

Install OpenCV and scikit-learn libraries using pip:</br>
<pre>pip install opencv-python scikit-learn</pre>

Launch Jupyter Notebook from the command line:</br>
<pre>jupyter notebook</pre>

Open the .ipynb file in Jupyter Notebook and run the code.

<a name="usage"></a>
## Usage Instructions
The code is designed to work with two datasets: objects dataset and flowers dataset. Both of these datasets have already been uploaded on a publicly accessible Google Drive folder. That folder gets downloaded from inside the Google Colaboratory notebook. Therefore, there is no need to download it locally.

All you need is to run the code and wait for the classifiers to be trained.

<a name="quant_res"></a>
## Quantitative Results

### Objects Dataset

#### SVM Confusion Matrix
![cm_o_svm](https://user-images.githubusercontent.com/41828100/224597979-7bcf5950-34a2-4fcd-ad6f-67bdfa4eb2f3.png)

#### SVM Class-Wise Results
![obj_svm](https://user-images.githubusercontent.com/41828100/224598383-1dbdb29b-1166-4c87-8bc0-718e8a14b006.jpg)

#### Random Forest Confusion Matrix
![cm_o_rf](https://user-images.githubusercontent.com/41828100/224598055-f6220691-3829-428b-9bf5-2ce4ee4c8b1c.png)

#### Random Forest Class-Wise Results
![obj_rf](https://user-images.githubusercontent.com/41828100/224598404-ae4cd296-3848-4687-a929-a6aaebdd4bcf.jpg)

#### Model Accuracy
![obj_acc](https://user-images.githubusercontent.com/41828100/224598501-72f07f72-8ad7-4088-a920-8c4580ef5ec8.jpg)

### Flowers Dataset

#### SVM Confusion Matrix
![cm_f_svm](https://user-images.githubusercontent.com/41828100/224598592-59660663-185a-442a-a707-8582fc66814b.png)

#### SVM Class-Wise Results
![flow_svm](https://user-images.githubusercontent.com/41828100/224598730-269fa690-6704-4f91-ac1a-d722b979dd50.jpg)

#### Random Forest Confusion Matrix
![cm_f_rf](https://user-images.githubusercontent.com/41828100/224598601-dd41b78b-80de-4bce-9bfa-c0290c4cd242.png)

#### Random Forest Class-Wise Results
![flow_rf](https://user-images.githubusercontent.com/41828100/224598752-30b1bb35-900f-4db0-9cd9-5329ea233d71.jpg)

#### Model Accuracy
![flow_acc](https://user-images.githubusercontent.com/41828100/224598774-6551a5ac-a069-41aa-baf0-e973cd9e4109.jpg)

<a name="vis_res"></a>
## Visual Results

### Correctly Classified Object
![cor_obj](https://user-images.githubusercontent.com/41828100/224599052-f8b9c44c-18c8-4734-a2f0-0ab2b4b5cfc4.jpg)

### Incorrectly Classified Object
![incor_obj](https://user-images.githubusercontent.com/41828100/224599073-d4b7c3bc-59e4-45cb-8236-301b8dd1f7bd.jpg)

### Correctly Classified Flower
![cor_flo](https://user-images.githubusercontent.com/41828100/224599088-1b4de0a1-d186-4d93-819b-0e76c0698ee7.jpg)

### Incorrectly Classified Flower
![incor_flo](https://user-images.githubusercontent.com/41828100/224599117-d3f13abc-669b-4d35-b95c-9f6df001ab52.jpg)

<a name="auth"></a>
## Author
Usama Athar atharusama99@gmail.com
